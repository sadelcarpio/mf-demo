"""Project user and item vectors to 2-D and draw them as an animated plotly figure."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import CATEGORIES, Dataset

# Categorical slots validated with the dataviz palette checker. Five hues in one scatter
# can't pass the all-pairs checks, so each category also gets its own marker shape,
# items get direct labels, and every point has a hover tooltip.
COLORS = dict(zip(CATEGORIES, ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100"]))
SYMBOLS = dict(zip(CATEGORIES, ["circle", "square", "diamond", "triangle-up", "pentagon"]))
INK, MUTED = "#52514e", "#8a8984"


def project(user_snaps, item_snaps, method: str = "PCA", normalize: bool = False, seed: int = 0):
    """Return lists of 2-D user/item coordinates, one entry per snapshot.

    PCA is fit once, on the final snapshot (users and items together), and every
    earlier snapshot is projected with that same basis. The axes stay fixed, so the
    animation shows the points moving and not the basis rotating.
    t-SNE has no reusable map, so it is run on every snapshot. It runs forward in time, and each
    frame starts from the previous frame's layout (no early exaggeration), so points only move
    as much as their neighborhoods change. Running backward from the final frame would carry
    the trained clusters into the early, still-random snapshots."""
    def prep(x):
        if normalize:
            x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)
        return x

    U = [prep(u) for u in user_snaps]
    V = [prep(v) for v in item_snaps]
    nu = U[-1].shape[0]
    final = np.vstack([U[-1], V[-1]])

    if method == "t-SNE":
        from sklearn.manifold import TSNE
        metric = "cosine" if normalize else "euclidean"
        us, vs, prev = [], [], None
        for u, v in zip(U, V):
            X = np.vstack([u, v])
            if prev is None:  # first snapshot: a regular t-SNE
                tsne = TSNE(2, perplexity=15, init="pca", random_state=seed, metric=metric)
            else:  # later snapshots: warm start from the previous layout
                tsne = TSNE(2, perplexity=15, init=prev, random_state=seed, metric=metric,
                            early_exaggeration=1.0, max_iter=300, learning_rate=200)
            prev = tsne.fit_transform(X)
            us.append(prev[:nu])
            vs.append(prev[nu:])
        return us, vs, 1.0

    mean = final.mean(0)
    if final.shape[1] <= 2:
        comps = np.eye(final.shape[1], 2)
        explained = 1.0
    else:
        _, s, vt = np.linalg.svd(final - mean, full_matrices=False)
        comps = vt[:2].T
        explained = float((s[:2] ** 2).sum() / (s ** 2).sum())
    return [(u - mean) @ comps for u in U], [(v - mean) @ comps for v in V], explained


def _short(title: str) -> str:
    return title if len(title) <= 18 else title[:16] + "…"


def embedding_figure(ds: Dataset, user_xy, item_xy, epochs, focus_user: int | None = None,
                     liked: dict | None = None, height: int = 620, title: str = "") -> go.Figure:
    """Animated scatter of users (small, no labels) and items (large, outlined, labeled).

    For focus_user, the figure also highlights that user and draws lines to the games
    they liked, so you can watch those games move toward the user as training runs."""
    likes = ds.likes.likes.reindex(ds.users.user_id).to_numpy()
    cold_u = set(ds.cold_users.tolist())
    cold_i = set(ds.cold_items.tolist())

    def user_hover(u):
        r = ds.users.iloc[u]
        if u in cold_u:
            head = "COLD START (no ratings)"
        else:
            head = f"likes {likes[u]} most (avg {ds.likes.likes_avg[u]:.1f}★)"
        return (f"<b>user {u}</b> · {head}<br>features: {r.platform} · {r.age_group} · "
                f"favorite: {r.favorite_category}")

    def item_hover(i):
        g = ds.games.iloc[i]
        tag = " · NEW RELEASE (no ratings)" if i in cold_i else ""
        return f"<b>{g.title}</b>{tag}<br>{g.category}<br>tags: {g.tags.replace('|', ', ')}"

    def traces(f):
        uxy, ixy = user_xy[f], item_xy[f]
        out = []
        for c in CATEGORIES:
            idx = np.flatnonzero(likes == c)
            out.append(go.Scatter(
                x=uxy[idx, 0], y=uxy[idx, 1], mode="markers", name=f"users who like {c}",
                legendgroup=c, showlegend=True,
                marker=dict(size=9, color=COLORS[c], opacity=0.75, symbol=SYMBOLS[c]),
                hovertext=[user_hover(u) for u in idx], hoverinfo="text"))
        idx = np.array(sorted(cold_u), dtype=int)
        out.append(go.Scatter(
            x=uxy[idx, 0], y=uxy[idx, 1], mode="markers", name="cold-start users (no ratings)",
            legendgroup="cold", showlegend=True,
            marker=dict(size=10, symbol="circle-open", color=INK, line=dict(width=2, color=INK)),
            hovertext=[user_hover(u) for u in idx], hoverinfo="text"))
        for c in CATEGORIES:
            idx = np.flatnonzero(ds.games.category.to_numpy() == c)
            out.append(go.Scatter(
                x=ixy[idx, 0], y=ixy[idx, 1], mode="markers+text", name=f"games · {c}",
                legendgroup=c, showlegend=True,
                marker=dict(size=19, color=["white" if i in cold_i else COLORS[c] for i in idx],
                            symbol=SYMBOLS[c], line=dict(width=2, color="#0b0b0b")),
                text=[_short(t) for t in ds.games.title.iloc[idx]], textposition="top center",
                textfont=dict(size=11, color=INK),
                hovertext=[item_hover(i) for i in idx], hoverinfo="text"))
        # focus user + lines to the games they liked
        lx, ly = [], []
        if focus_user is not None and liked:
            for i in liked.get(focus_user, []):
                lx += [uxy[focus_user, 0], ixy[i, 0], None]
                ly += [uxy[focus_user, 1], ixy[i, 1], None]
        out.append(go.Scatter(x=lx, y=ly, mode="lines", name="liked by focus user",
                              line=dict(color=INK, width=1.5, dash="dot"), hoverinfo="skip",
                              showlegend=focus_user is not None))
        fx = [uxy[focus_user, 0]] if focus_user is not None else []
        fy = [uxy[focus_user, 1]] if focus_user is not None else []
        out.append(go.Scatter(x=fx, y=fy, mode="markers", name=f"focus: user {focus_user}",
                              marker=dict(size=22, symbol="star", color="#0b0b0b",
                                          line=dict(width=2, color="white")),
                              hovertext=[user_hover(focus_user)] if fx else [], hoverinfo="text",
                              showlegend=focus_user is not None))
        # legend-only key for the new game
        out.append(go.Scatter(x=[None], y=[None], mode="markers", name="white = new game (no ratings)",
                              legendgroup="key", marker=dict(size=15, symbol="circle", color="white",
                                                             line=dict(width=2, color="#0b0b0b"))))
        return out

    n = len(user_xy)
    allxy = np.vstack([np.vstack([u, v]) for u, v in zip(user_xy, item_xy)])
    lo, hi = allxy.min(0), allxy.max(0)
    pad = (hi - lo) * 0.08 + 1e-6
    xr, yr = [lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]

    fig = go.Figure(data=traces(n - 1))  # start on the trained state
    if n > 1:
        fig.frames = [go.Frame(data=traces(f), name=str(epochs[f])) for f in range(n)]
        steps = [dict(method="animate", label=str(epochs[f]),
                      args=[[str(epochs[f])], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                   transition=dict(duration=0))])
                 for f in range(n)]
        fig.update_layout(
            sliders=[dict(active=n - 1, steps=steps, currentvalue=dict(prefix="epoch "),
                          pad=dict(t=40), len=0.9, x=0.1)],
            updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0, xanchor="left",
                              yanchor="top", pad=dict(t=45, r=10), buttons=[
                    dict(label="▶ replay training", method="animate",
                         args=[None, dict(frame=dict(duration=180, redraw=True), fromcurrent=False,
                                          transition=dict(duration=120, easing="linear"))]),
                    dict(label="❚❚", method="animate",
                         args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
                ])])
    fig.update_layout(
        title=title, height=height, hovermode="closest", margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(range=xr, showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False,
                   title="component 1"),
        yaxis=dict(range=yr, showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False,
                   title="component 2"),
        legend=dict(groupclick="toggleitem", font=dict(size=11), tracegroupgap=6,
                    title=dict(text="<b>color + shape = category</b><br>"
                                    "small = user, colored by the category<br>they rate highest<br>"
                                    "large outlined = game<br>"
                                    "<i>click an entry to hide it</i>", font=dict(size=11))),
    )
    return fig


def liked_items(ds: Dataset, threshold: float = 3) -> dict[int, list[int]]:
    pos = ds.train[ds.train.rating > threshold]
    return pos.groupby("user_id").item_id.apply(list).to_dict()


def topk_table(ds: Dataset, scores: np.ndarray, user: int, k: int = 5) -> pd.DataFrame:
    """Top-k unseen games for a user, plus how the user rated them in the held-out set (if at all)."""
    s = scores[user].copy()
    seen = ds.train.item_id[ds.train.user_id == user].to_numpy()
    s[seen] = -np.inf
    top = np.argsort(-s)[:k]
    held = ds.test[ds.test.user_id == user].set_index("item_id").rating
    return pd.DataFrame({
        "game": ds.games.title.iloc[top].to_numpy(),
        "category": ds.games.category.iloc[top].to_numpy(),
        "score": scores[user, top].round(3),
        "held-out rating": [str(held[i]) + "★" if i in held.index else "–" for i in top],
    })
