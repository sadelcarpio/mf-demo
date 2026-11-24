"""The five models from the deck, in PyTorch.

Every trainer returns a `Result` holding snapshots of the user and item vectors
taken during training. Those snapshots are what the app animates. In each
snapshot, the score is the dot product of a user vector and an item vector
(plus biases where the model has them).

  model      user vector                     item vector                  loss
  ---------  ------------------------------  ---------------------------  --------------------------
  MF         row of U (table)                row of V (table)             MSE on observed ratings
  LogMF      row of U (table)                row of V (table)             BCE on rating > threshold
  Softmax    encoder(user feats + history)   row of W (pre-softmax table) CE vs 1/n target, /tau
  Two-tower  encoder(user feats + history)   encoder(item feats), L2-norm in-batch softmax B=4, /tau, logQ
  FM         sum of user-feature v's         sum of item-feature v's      MSE or BCE (pointwise)
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Dataset

torch.set_num_threads(2)
N_SNAPSHOTS = 40


@dataclass
class Result:
    epochs: list[int] = field(default_factory=list)
    user_snaps: list[np.ndarray] = field(default_factory=list)
    item_snaps: list[np.ndarray] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    scores: np.ndarray | None = None  # final (n_users, n_items) ranking scores
    metrics: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    def snap(self, epoch: int, U: torch.Tensor, V: torch.Tensor) -> None:
        self.epochs.append(epoch)
        self.user_snaps.append(U.detach().cpu().numpy().copy())
        self.item_snaps.append(V.detach().cpu().numpy().copy())


def _snap_epochs(n_epochs: int) -> set[int]:
    return set(np.unique(np.linspace(0, n_epochs, N_SNAPSHOTS + 1).round().astype(int)))


# --------------------------------------------------------------------------- metrics
def recall_at_k(ds: Dataset, scores: np.ndarray, k: int = 5, threshold: float = 3) -> float:
    """Held-out positives found in the top-k. Games the user already rated in train are excluded."""
    s = scores.copy()
    s[ds.train.user_id, ds.train.item_id] = -np.inf
    topk = np.argsort(-s, axis=1)[:, :k]
    rec = []
    for u, g in ds.test[ds.test.rating > threshold].groupby("user_id"):
        pos = set(g.item_id)
        rec.append(len(pos & set(topk[u])) / min(k, len(pos)))
    return float(np.mean(rec)) if rec else float("nan")


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


# --------------------------------------------------------------------------- encoders
class UserEncoder(nn.Module):
    """concat[user_id, platform, age, favorite_category, mean(history items)] -> MLP -> R^k."""

    def __init__(self, ds: Dataset, k: int, d: int = 8, hidden: int = 32,
                 use_id: bool = True, use_history: bool = True):
        super().__init__()
        self.use_id, self.use_history = use_id, use_history
        self.feats = {n: torch.as_tensor(np.array(v)) for n, v in ds.user_feats.items()}
        self.id_emb = nn.Embedding(ds.n_users, d)
        self.feat_embs = nn.ModuleDict({n: nn.Embedding(len(v), d) for n, v in ds.user_vocab.items()})
        self.hist_emb = nn.Linear(ds.n_items, d, bias=False)  # bag-of-items embedding
        n_in = d * (len(self.feat_embs) + use_id + use_history)
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, uids: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        parts = [self.feat_embs[n](self.feats[n][uids]) for n in self.feat_embs]
        if self.use_id:
            parts.append(self.id_emb(uids))
        if self.use_history:
            parts.append(self.hist_emb(history / history.sum(1, keepdim=True).clamp(min=1)))
        return self.mlp(torch.cat(parts, 1))


class ItemEncoder(nn.Module):
    """concat[item_id, mean(tag embeddings), mode] -> MLP -> R^k."""

    def __init__(self, ds: Dataset, k: int, d: int = 8, hidden: int = 32, use_id: bool = True):
        super().__init__()
        self.use_id = use_id
        self.tags = torch.as_tensor(ds.item_tags)
        self.tags = self.tags / self.tags.sum(1, keepdim=True)
        self.mode = torch.as_tensor(np.array(ds.item_mode))
        self.id_emb = nn.Embedding(ds.n_items, d)
        self.tag_emb = nn.Linear(len(ds.tag_vocab), d, bias=False)
        self.mode_emb = nn.Embedding(len(ds.mode_vocab), d)
        self.mlp = nn.Sequential(nn.Linear(d * (2 + use_id), hidden), nn.ReLU(), nn.Linear(hidden, k))

    def forward(self, iids: torch.Tensor) -> torch.Tensor:
        parts = [self.tag_emb(self.tags[iids]), self.mode_emb(self.mode[iids])]
        if self.use_id:
            parts.append(self.id_emb(iids))
        return self.mlp(torch.cat(parts, 1))


# --------------------------------------------------------------------------- 1 & 2: MF
def train_mf(ds: Dataset, k: int = 2, epochs: int = 300, lr: float = 0.05, l2: float = 0.05,
             logistic: bool = False, threshold: float = 3, biases: bool = False,
             solver: str = "gd", sweeps: int = 15, init_std: float = 0.1, seed: int = 0) -> Result:
    """Classic MF, R_hat = U V^T, trained only on observed entries.
    logistic=False: MSE against 1-5 stars.  logistic=True: BCE against y = [rating > threshold].

    solver="gd":  full-batch gradient descent with Adam. Each epoch is one step on the loss over
                  all observed ratings.
    solver="als": alternating least squares. Each sweep solves every user row in closed form with V
                  fixed, then every item row with U fixed. Both solvers minimize the same objective
                  (the L2 term per row scales with its number of ratings, as in ALS-WR)."""
    torch.manual_seed(seed)
    u = torch.as_tensor(ds.train.user_id.to_numpy().copy())
    i = torch.as_tensor(ds.train.item_id.to_numpy().copy())
    r = torch.as_tensor(ds.train.rating.to_numpy().copy(), dtype=torch.float32)
    y = (r > threshold).float() if logistic else r

    U = nn.Parameter(torch.randn(ds.n_users, k) * init_std)
    V = nn.Parameter(torch.randn(ds.n_items, k) * init_std)
    bu = nn.Parameter(torch.zeros(ds.n_users))
    bi = nn.Parameter(torch.zeros(ds.n_items))
    b0 = nn.Parameter(torch.tensor(float(y.mean())) if biases else torch.tensor(0.0))
    params = [U, V] + ([bu, bi, b0] if biases else [])
    opt = torch.optim.Adam(params, lr=lr)

    def predict(uu, ii):
        s = (U[uu] * V[ii]).sum(1)
        return s + bu[uu] + bi[ii] + b0 if biases else s

    res = Result()
    res.snap(0, U, V)
    if solver == "als":
        if logistic:
            raise ValueError("ALS needs squared loss: logistic MF has no closed-form row update")
        with torch.no_grad():
            _als(ds, u, i, r, U, V, bu, bi, b0, l2, biases, sweeps, res, predict)
    else:
        snaps = _snap_epochs(epochs)
        for ep in range(1, epochs + 1):
            s = predict(u, i)  # full batch: every observed train rating, every step
            data_loss = F.binary_cross_entropy_with_logits(s, y) if logistic else F.mse_loss(s, y)
            reg = l2 * ((U[u] ** 2).sum(1) + (V[i] ** 2).sum(1)).mean()
            opt.zero_grad()
            (data_loss + reg).backward()
            opt.step()
            res.loss.append(data_loss.item())
            if ep in snaps:
                res.snap(ep, U, V)

    with torch.no_grad():
        full = U @ V.T
        if biases:
            full = full + bu[:, None] + bi[None, :] + b0
        res.scores = full.numpy()
        tu = torch.as_tensor(ds.test.user_id.to_numpy().copy())
        ti = torch.as_tensor(ds.test.item_id.to_numpy().copy())
        tr = ds.test.rating.to_numpy()
        st = predict(tu, ti).numpy()
        if logistic:
            ty = (tr > threshold).astype(int)
            res.metrics["test accuracy"] = float(((st > 0) == ty).mean())
            res.metrics["test AUC"] = _auc(ty, st)
            # (score, label) for the sigmoid plot, as on the logistic MF slide
            res.extras["train_scores"] = predict(u, i).numpy()
            res.extras["train_labels"] = y.numpy()
        else:
            res.metrics["train RMSE"] = float(np.sqrt(res.loss[-1]))
            res.metrics["test RMSE"] = float(np.sqrt(((st - tr) ** 2).mean()))
    res.metrics["test recall@5"] = recall_at_k(ds, res.scores, threshold=threshold)
    return res


def _solve_rows(M, T, Fx, reg, reg_mask):
    """Ridge regression for every row at once: for row r with observed columns c (M[r, c] = 1),
    minimize sum_c (T[r, c] - <x_r, Fx[c]>)^2 + reg[r] * ||x_r * reg_mask||^2.
    Closed form: x_r = (Fx_c^T Fx_c + reg[r] diag(reg_mask))^-1 Fx_c^T T[r, c]."""
    A = torch.einsum("rc,cd,ce->rde", M, Fx, Fx) + reg[:, None, None] * torch.diag(reg_mask)
    b = torch.einsum("rc,cd->rd", M * T, Fx)
    return torch.linalg.solve(A, b)


def _als(ds, u, i, r, U, V, bu, bi, b0, l2, biases, sweeps, res, predict):
    """Alternating least squares on the same objective as the GD path:
    sum_obs (r - pred)^2 + l2 * sum_obs (|U_u|^2 + |V_i|^2)  ==  per-row L2 of l2 * n_ratings(row).
    Biases (if on) are solved jointly with the embedding as an extra unregularized column.
    The global mean b0 stays fixed. Rows with no ratings (cold start) have nothing to fit and
    keep their initial values, as in the GD path."""
    k = U.shape[1]
    R = torch.zeros(ds.n_users, ds.n_items)
    M = torch.zeros(ds.n_users, ds.n_items)
    R[u, i], M[u, i] = r, 1.0
    n_u, n_i = M.sum(1), M.sum(0)
    wu, wi = n_u > 0, n_i > 0  # rows that have data
    ones_u, ones_i = torch.ones(ds.n_users, 1), torch.ones(ds.n_items, 1)
    mask = torch.cat([torch.ones(k), torch.zeros(1)]) if biases else torch.ones(k)

    for s in range(1, sweeps + 1):
        # users: fix V (and item biases), solve each user's row
        Fx = torch.cat([V, ones_i], 1) if biases else V
        T = R - (b0 + bi[None, :]) if biases else R
        x = _solve_rows(M[wu], T[wu], Fx, l2 * n_u[wu], mask)
        U[wu] = x[:, :k]
        if biases:
            bu[wu] = x[:, k]
        res.snap(s - 0.5, U, V)

        # items: fix U (and user biases), solve each item's row
        Fx = torch.cat([U, ones_u], 1) if biases else U
        T = R.T - (b0 + bu[None, :]) if biases else R.T
        x = _solve_rows(M.T[wi], T[wi], Fx, l2 * n_i[wi], mask)
        V[wi] = x[:, :k]
        if biases:
            bi[wi] = x[:, k]
        res.snap(s, U, V)
        res.loss.append(F.mse_loss(predict(u, i), r).item())


# --------------------------------------------------------------------------- 3: softmax
def train_softmax(ds: Dataset, k: int = 8, epochs: int = 300, lr: float = 0.01,
                  temperature: float = 1.0, threshold: float = 3, l2: float = 1e-4,
                  use_id: bool = True, use_history: bool = True, seed: int = 0) -> Result:
    """u = f(user features) ;  z = (W u + b) / tau ;  p = softmax(z) over ALL items.
    Target q = y / n over the user's positives (1/n target slide).
    Each epoch the positives are randomly split in half: one half is fed as history,
    the other is the target. Otherwise the encoder could copy history into the output."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    P = torch.as_tensor(ds.train_matrix(threshold))  # positives
    users = torch.nonzero(P.sum(1) > 0).squeeze(1)
    all_users = torch.arange(ds.n_users)

    enc = UserEncoder(ds, k, use_id=use_id, use_history=use_history)
    W = nn.Parameter(torch.randn(ds.n_items, k) * 0.1)
    b = nn.Parameter(torch.zeros(ds.n_items))
    opt = torch.optim.Adam(list(enc.parameters()) + [W, b], lr=lr, weight_decay=l2)

    res = Result()
    snaps = _snap_epochs(epochs)

    def snapshot(ep):
        with torch.no_grad():
            res.snap(ep, enc(all_users, P), W)  # serving: full history

    snapshot(0)
    for ep in range(1, epochs + 1):
        Pu = P[users]
        split = (torch.rand(Pu.shape, generator=gen) < 0.5) & (Pu > 0)
        target = Pu * ~split
        empty = target.sum(1) == 0  # fewer than 2 positives: use all of them as target, no history
        target[empty] = Pu[empty]
        hist = Pu * split
        hist[empty] = 0
        q = target / target.sum(1, keepdim=True)

        z = (enc(users, hist) @ W.T + b) / temperature
        loss = -(q * F.log_softmax(z, 1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        res.loss.append(loss.item())
        if ep in snaps:
            snapshot(ep)

    with torch.no_grad():
        z = (enc(all_users, P) @ W.T + b) / temperature
        res.scores = z.numpy()
        res.extras["probs"] = F.softmax(z, 1).numpy()
    res.metrics["test recall@5"] = recall_at_k(ds, res.scores, threshold=threshold)
    return res


# --------------------------------------------------------------------------- 4: two-tower
def train_two_tower(ds: Dataset, k: int = 8, epochs: int = 40, lr: float = 0.005,
                    temperature: float = 0.1, threshold: float = 3, batch_size: int = 4,
                    logq: bool = True, use_id: bool = True, use_history: bool = True,
                    l2: float = 1e-4, seed: int = 0) -> Result:
    """In-batch softmax: s_ij = <u_i, v_j> / tau - log q_j, with L2-normalized towers.
    Diagonal = positives, off-diagonal = free negatives. If two pairs in a batch share
    the same item, that off-diagonal entry is a false negative and gets masked out."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    P = torch.as_tensor(ds.train_matrix(threshold))
    pu, pi = torch.nonzero(P, as_tuple=True)  # positive (user, item) pairs
    counts = torch.bincount(pi, minlength=ds.n_items).float()
    log_q = torch.log(counts.clamp(min=1) / counts.sum())  # q ~ item popularity

    ut = UserEncoder(ds, k, use_id=use_id, use_history=use_history)
    it = ItemEncoder(ds, k, use_id=use_id)
    opt = torch.optim.Adam(list(ut.parameters()) + list(it.parameters()), lr=lr, weight_decay=l2)
    all_users, all_items = torch.arange(ds.n_users), torch.arange(ds.n_items)

    def towers(uids, hist, iids):
        return F.normalize(ut(uids, hist), dim=1), F.normalize(it(iids), dim=1)

    res = Result()
    snaps = _snap_epochs(epochs)

    def snapshot(ep):
        with torch.no_grad():
            res.snap(ep, *towers(all_users, P, all_items))

    snapshot(0)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(pu), generator=gen)
        ep_loss = []
        for s in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[s:s + batch_size]
            bu, bi = pu[idx], pi[idx]
            hist = P[bu].clone()
            hist[torch.arange(len(bu)), bi] = 0  # never feed the target item as history
            u, v = towers(bu, hist, bi)
            logits = u @ v.T / temperature
            if logq:
                logits = logits - log_q[bi][None, :]
            same = (bi[:, None] == bi[None, :]) & ~torch.eye(len(bi), dtype=torch.bool)
            logits = logits.masked_fill(same, -1e9)
            loss = F.cross_entropy(logits, torch.arange(len(bi)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(loss.item())
        res.loss.append(float(np.mean(ep_loss)))
        if ep in snaps:
            snapshot(ep)

    with torch.no_grad():
        U, V = towers(all_users, P, all_items)
        res.scores = (U @ V.T).numpy()  # serving: cosine, no logQ
        # one example batch, shown as a 4x4 matrix like the slide
        idx = torch.randperm(len(pu), generator=gen)[:batch_size]
        bu, bi = pu[idx], pi[idx]
        hist = P[bu].clone()
        hist[torch.arange(len(bu)), bi] = 0
        u, v = towers(bu, hist, bi)
        logits = u @ v.T / temperature - (log_q[bi][None, :] if logq else 0)
        res.extras["batch"] = dict(users=bu.numpy(), items=bi.numpy(), logits=logits.numpy(),
                                   probs=F.softmax(logits, 1).numpy())
        res.extras["item_q"] = (counts / counts.sum()).numpy()
    res.metrics["test recall@5"] = recall_at_k(ds, res.scores, threshold=threshold)
    return res


# --------------------------------------------------------------------------- 5: FM
def _fm_design(ds: Dataset, use_ids: bool):
    """One column per feature value. Returns user-part X_u (n_users, F), item-part X_i (n_items, F)
    and column labels. The x of a (user, item) pair is X_u[u] + X_i[i]."""
    blocks_u, blocks_i, names = [], [], []

    def onehot(codes, n):
        return np.eye(n, dtype=np.float32)[codes]

    if use_ids:
        blocks_u.append(np.eye(ds.n_users, dtype=np.float32))
        names += [f"user_id={u}" for u in range(ds.n_users)]
    for n, vocab in ds.user_vocab.items():
        blocks_u.append(onehot(ds.user_feats[n], len(vocab)))
        names += [f"{n}={v}" for v in vocab]
    n_u = sum(b.shape[1] for b in blocks_u)
    if use_ids:
        blocks_i.append(np.eye(ds.n_items, dtype=np.float32))
        names += [f"item_id={t}" for t in ds.games.title]
    blocks_i.append(ds.item_tags)
    names += [f"tag={t}" for t in ds.tag_vocab]
    blocks_i.append(onehot(ds.item_mode, len(ds.mode_vocab)))
    names += [f"mode={m}" for m in ds.mode_vocab]
    n_i = sum(b.shape[1] for b in blocks_i)

    Xu = np.concatenate([np.concatenate(blocks_u, 1), np.zeros((ds.n_users, n_i), np.float32)], 1)
    Xi = np.concatenate([np.zeros((ds.n_items, n_u), np.float32), np.concatenate(blocks_i, 1)], 1)
    return torch.as_tensor(Xu), torch.as_tensor(Xi), names


def train_fm(ds: Dataset, k: int = 4, epochs: int = 300, lr: float = 0.03, l2: float = 0.01,
             logistic: bool = False, threshold: float = 3, use_ids: bool = True,
             init_std: float = 0.1, seed: int = 0) -> Result:
    """y_hat(x) = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j.
    For the plot, apply the "FM = two towers with linear encoders" slide:
    e_u = sum_{a in user feats} x_a v_a ,  e_i = sum_{b in item feats} x_b v_b ."""
    torch.manual_seed(seed)
    Xu, Xi, names = _fm_design(ds, use_ids)
    n_f = Xu.shape[1]
    u = torch.as_tensor(ds.train.user_id.to_numpy().copy())
    i = torch.as_tensor(ds.train.item_id.to_numpy().copy())
    r = torch.as_tensor(ds.train.rating.to_numpy().copy(), dtype=torch.float32)
    y = (r > threshold).float() if logistic else r

    w0 = nn.Parameter(torch.tensor(float(y.mean()) if not logistic else 0.0))
    w = nn.Parameter(torch.zeros(n_f))
    V = nn.Parameter(torch.randn(n_f, k) * init_std)
    opt = torch.optim.Adam([w0, w, V], lr=lr)

    def fm(X):
        xv = X @ V
        pair = 0.5 * (xv ** 2 - (X ** 2) @ (V ** 2)).sum(1)  # O(nk) trick for sum_{i<j}
        return w0 + X @ w + pair

    X = Xu[u] + Xi[i]
    res = Result()
    snaps = _snap_epochs(epochs)
    res.snap(0, Xu @ V, Xi @ V)
    for ep in range(1, epochs + 1):
        s = fm(X)
        data_loss = F.binary_cross_entropy_with_logits(s, y) if logistic else F.mse_loss(s, y)
        reg = l2 * ((X @ (V ** 2)).sum(1).mean() + (X @ w ** 2).mean())  # only active features
        opt.zero_grad()
        (data_loss + reg).backward()
        opt.step()
        res.loss.append(data_loss.item())
        if ep in snaps:
            res.snap(ep, Xu @ V, Xi @ V)

    with torch.no_grad():
        uu, ii = torch.meshgrid(torch.arange(ds.n_users), torch.arange(ds.n_items), indexing="ij")
        res.scores = fm(Xu[uu.flatten()] + Xi[ii.flatten()]).reshape(ds.n_users, ds.n_items).numpy()
        tu = ds.test.user_id.to_numpy()
        ti = ds.test.item_id.to_numpy()
        tr = ds.test.rating.to_numpy()
        st = res.scores[tu, ti]
        if logistic:
            ty = (tr > threshold).astype(int)
            res.metrics["test accuracy"] = float(((st > 0) == ty).mean())
            res.metrics["test AUC"] = _auc(ty, st)
        else:
            res.metrics["train RMSE"] = float(np.sqrt(res.loss[-1]))
            res.metrics["test RMSE"] = float(np.sqrt(((st - tr) ** 2).mean()))
        res.extras["feature_names"] = names
        res.extras["feature_V"] = V.detach().numpy()
        res.extras["feature_w"] = w.detach().numpy()
    res.metrics["test recall@5"] = recall_at_k(ds, res.scores, threshold=threshold)
    return res


TRAINERS = {
    "mf": train_mf,
    "logmf": lambda ds, **kw: train_mf(ds, logistic=True, **kw),
    "softmax": train_softmax,
    "two_tower": train_two_tower,
    "fm": train_fm,
}
