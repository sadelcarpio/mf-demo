"""From matrix factorization to two towers: an interactive comparison.

Run:  uv run streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mfdemo import data, models, viz

st.set_page_config(page_title="MF → Two-tower demo", page_icon="🎮", layout="wide")

PAGES = {
    "dataset": "📦 Dataset",
    "mf": "1 · MF (regression)",
    "logmf": "2 · Logistic MF",
    "softmax": "3 · Softmax over all items",
    "two_tower": "4 · Two-tower (in-batch)",
    "fm": "5 · Factorization machine",
    "compare": "⚖️ Compare all",
}

COSINE_DEFAULT = {"softmax", "two_tower"}

DEFAULTS = {
    "mf": dict(k=3, epochs=300, lr=0.05, l2=0.2, biases=False),
    "logmf": dict(k=3, epochs=300, lr=0.05, l2=0.05, threshold=3),
    "softmax": dict(k=3, epochs=300, lr=0.01, temperature=1.0, threshold=3, use_id=True, use_history=True),
    "two_tower": dict(k=3, epochs=40, lr=0.01, temperature=0.1, threshold=3, logq=True,
                      use_id=True, use_history=True),
    "fm": dict(k=4, epochs=300, lr=0.03, l2=0.1, logistic=False, threshold=3, use_ids=True),
}


@st.cache_resource
def get_data() -> data.Dataset:
    return data.load()


@st.cache_data(show_spinner="Training…", max_entries=64)
def train(name: str, params: tuple) -> models.Result:
    return models.TRAINERS[name](get_data(), **dict(params))


@st.cache_data(show_spinner="Projecting…", max_entries=64)
def projected(name: str, params: tuple, method: str, normalize: bool):
    r = train(name, params)
    return viz.project(r.user_snaps, r.item_snaps, method, normalize)


ds = get_data()


def likes_of(u: int) -> str:
    """Category the user rates highest on average (used for colors and labels)."""
    return ds.likes.likes[u] or "cold start"


def users_by_likes() -> np.ndarray:
    """User ids grouped by liked category (in CATEGORIES order), cold-start users last."""
    rank = {c: i for i, c in enumerate(data.CATEGORIES)}
    return np.array(sorted(ds.users.user_id, key=lambda u: (rank.get(ds.likes.likes[u], 99), u)))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎮 MF → Two-tower")
    page = st.radio("Model", list(PAGES), format_func=PAGES.get, label_visibility="collapsed")
    st.divider()
    st.subheader("Embedding plot")
    method = st.radio("2-D projection", ["PCA", "t-SNE"], horizontal=True,
                      help="PCA is fit on the final epoch and reused for every snapshot, so the "
                           "animation is honest. t-SNE runs on every snapshot, each starting "
                           "from the previous layout. It keeps local neighborhoods but not global "
                           "distances, and its axes have no meaning.")
    # softmax: only the direction of u matters, and its norm dwarfs W's rows; two-tower is already unit-norm
    normalize = st.checkbox("Cosine view (L2-normalize vectors first)", value=page in COSINE_DEFAULT,
                            key=f"normalize_{page}", disabled=page == "compare",
                            help="Scores are dot products. Normalizing plots directions only, so "
                                 "closeness in the plot matches the score more closely. On by default for "
                                 "softmax, where the encoder output u grows much longer than W's rows. "
                                 "The compare page uses each model's default.")
    def user_label(u):
        if u is None:
            return "none"
        if u in ds.cold_users:
            return f"user {u} · cold start (no ratings)"
        return f"user {u} · likes {likes_of(u)}"

    focus = st.selectbox("Focus user", [None] + ds.users.user_id.tolist(), index=1,
                         format_func=user_label,
                         help="Drawn as a ★, with dotted lines to the games they rated above "
                              "the threshold in train.")

# ------------------------------------------------------------------ helpers
FORMULAS = {
    "mf": (r"\hat A = UV^\top,\qquad \mathcal L=\sum_{(u,i)\in obs}(A_{ui}-U_uV_i^\top)^2"
           r"+\lambda(\lVert U_u\rVert^2+\lVert V_i\rVert^2)"),
    "logmf": (r"\hat y_{ui}=\sigma(U_uV_i^\top),\quad \mathcal L=-\sum_{obs}[y\log\hat y+(1-y)\log(1-\hat y)],"
              r"\quad \frac{\partial\mathcal L}{\partial V_i}=(\hat y_{ui}-y_{ui})U_u"),
    "softmax": (r"u=f(x_u),\quad z_i=\frac{\langle u,w_i\rangle+b_i}{\tau},\quad p=\mathrm{softmax}(z),"
                r"\quad \mathcal L_u=-\sum_i \tfrac{y_{ui}}{n_u}\log p_{ui}"),
    "two_tower": (r"s_{ij}=\frac{\langle u_i,v_j\rangle}{\tau}-\log q_j,\qquad"
                  r"\mathcal L=-\frac1B\sum_i\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}},\quad"
                  r"\lVert u\rVert=\lVert v\rVert=1"),
    "fm": (r"\hat y(x)=w_0+\sum_i w_ix_i+\sum_{i<j}\langle v_i,v_j\rangle x_ix_j"
           r"\;=\;c_u+b_i+\langle e_u,e_i\rangle,\quad e_u=\sum_{a\in U}x_av_a,\;e_i=\sum_{b\in I}x_bv_b"),
}

WHAT_IS_PLOTTED = {
    "mf": "**Users** are rows of the table U; **games** are rows of V. Both are plain lookups, as on the slides.",
    "logmf": "Same tables as MF. Only the loss changes: the label is now *liked or not* instead of the star count.",
    "softmax": "**Users** are the encoder output u = f(user_id, platform, age, favorite, history); "
               "**games** are the rows of the pre-softmax weight matrix W (the bias b is not plotted).",
    "two_tower": "**Users** are the L2-normalized user tower output and **games** the L2-normalized item tower "
                 "output (item_id, tags, mode). The score is cosine similarity / τ.",
    "fm": "Following the *FM = two towers with linear encoders* slide: **users** are e_u = Σ v over active "
          "user features (user_id, platform, age, favorite) and **games** are e_i = Σ v over item_id, tags, mode.",
}


def controls(name: str) -> tuple:
    """Hyperparameter form. Returns params as a hashable tuple for caching."""
    d = DEFAULTS[name]
    p = {}
    with st.form(f"form_{name}", border=True):
        c = st.columns(4)
        p["k"] = c[0].select_slider("Embedding dim k", [2, 3, 4, 8, 16, 32], value=d["k"])
        p["epochs"] = c[1].select_slider("Epochs", [10, 20, 40, 80, 150, 300, 600, 1000],
                                         value=d["epochs"])
        p["lr"] = c[2].select_slider("Learning rate", [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1],
                                     value=d["lr"])
        if "l2" in d:

            p["l2"] = c[3].select_slider("L2 λ", [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0], value=d["l2"])
        if "temperature" in d:
            p["temperature"] = c[3].select_slider(
                "Temperature τ", [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0], value=d["temperature"])
        c = st.columns(4)
        if "threshold" in d:
            p["threshold"] = c[0].slider("Positive if rating >", 1, 4, d["threshold"])
        if "biases" in d:
            p["biases"] = c[1].checkbox("Add biases (μ + b_u + b_i)", d["biases"],
                                        help="Not on the slides. Biases absorb 'this user rates "
                                             "everything high', which leaves U and V free to encode taste.")
        if "logistic" in d:
            p["logistic"] = c[1].toggle("Logistic loss (rating > threshold)", d["logistic"])
        if "use_ids" in d:
            p["use_ids"] = c[2].checkbox("Include user_id / item_id features", d["use_ids"],
                                         help="Off = pure content model. Every user and game is just the "
                                              "sum of its feature embeddings.")
        if "use_id" in d:
            p["use_id"] = c[1].checkbox("user_id embedding", d["use_id"])
            p["use_history"] = c[2].checkbox("history (liked games)", d["use_history"])
        if "logq" in d:
            p["logq"] = c[3].checkbox("logQ correction", d["logq"])
            c[0].number_input("Batch size B", value=4, disabled=True,
                              help="Fixed at 4, as on the slide: each batch is a 4×4 score matrix.")
        st.form_submit_button("Train with these settings", type="primary")
    return tuple(sorted(p.items()))


def metric_row(r: models.Result) -> None:
    cols = st.columns(len(r.metrics))
    for c, (k, v) in zip(cols, r.metrics.items()):
        c.metric(k, f"{v:.3f}")


def loss_chart(r: models.Result, ylabel: str) -> None:
    st.caption(f"Training loss ({ylabel})")
    st.line_chart(pd.DataFrame({"epoch": np.arange(1, len(r.loss) + 1), "loss": r.loss}),
                  x="epoch", y="loss", height=180)


def embedding_section(name: str, params: tuple, threshold: float) -> models.Result:
    r = train(name, params)
    uxy, ixy, explained = projected(name, params, method, normalize)
    epochs = r.epochs
    left, right = st.columns([5, 2])
    with left:
        fig = viz.embedding_figure(ds, uxy, ixy, epochs, focus_user=focus,
                                   liked=viz.liked_items(ds, threshold))
        st.plotly_chart(fig, width="stretch")
        k = dict(params)["k"]
        note = (f"PCA of k={k} → 2 (explains {explained:.0%} of the final variance)"
                if method == "PCA" else "t-SNE, warm-started frame to frame (only local neighborhoods are meaningful)")
        st.caption(f"{note}. Press **▶ replay training** to animate the saved snapshots of the last run, or drag the epoch slider. "
                   "Small markers are users and large outlined ones are games. Color and shape "
                   "show the category; a user gets the category they rate highest. Open gray "
                   "markers are cold-start users and the white game is a new release; none of "
                   "them have ratings.")
    with right:
        st.markdown(WHAT_IS_PLOTTED[name])
        metric_row(r)
        loss_chart(r, {"mf": "MSE", "logmf": "BCE", "softmax": "cross-entropy",
                       "two_tower": "in-batch CE", "fm": "BCE" if dict(params).get("logistic") else "MSE"}[name])
        if focus is not None:
            st.caption(f"Top-5 unseen games for **user {focus}**")
            st.dataframe(viz.topk_table(ds, r.scores, focus), hide_index=True, width="stretch")
    return r


def explainer(name: str, text: str) -> None:
    with st.expander("How this model works (from the slides)", expanded=False):
        st.latex(FORMULAS[name])
        st.markdown(text)


# ------------------------------------------------------------------ pages
st.header(PAGES[page])

if page == "dataset":
    st.markdown(
        f"A small synthetic **snapshot** of videogame ratings (no timestamps): "
        f"**{ds.n_users} users**, **{ds.n_items} games** and **{len(ds.ratings)} ratings** "
        f"({len(ds.ratings) / (ds.n_users * ds.n_items):.0%} of the matrix). "
        f"{len(ds.test)} ratings are held out for evaluation. "
        f"Generated by `data/generate_data.py` (seeded).")
    st.info(
        "**Plot colors:** each user is colored by the game category they rate highest on average, "
        "computed from the ratings below. It's only a label for the plots, not a model input.\n\n"
        f"**Cold start:** users {', '.join(map(str, ds.cold_users))} have features but no ratings, and "
        f"*{', '.join(ds.games.title.iloc[ds.cold_items])}* is a new release with no ratings.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Games (item features)")
        st.dataframe(ds.games, hide_index=True, width="stretch", height=400)
    with c2:
        st.subheader("Users (user features)")
        st.dataframe(ds.users, hide_index=True, width="stretch", height=400)

    st.subheader("Rating matrix")
    order = users_by_likes()
    games_order = ds.games.sort_values("category")
    M = np.full((ds.n_users, ds.n_items), np.nan)
    M[ds.ratings.user_id, ds.ratings.item_id] = ds.ratings.rating
    M = M[order][:, games_order.item_id]
    fig = go.Figure(go.Heatmap(
        z=M, x=games_order.title, y=[f"u{u} · {likes_of(u)}" for u in order], colorscale="Blues",
        zmin=1, zmax=5, hovertemplate="%{y}<br>%{x}<br>%{z}★<extra></extra>", xgap=1, ygap=1,
        colorbar=dict(title="★")))
    fig.update_layout(height=900, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Rows are grouped by the category each user rates highest, columns by game category. The block-diagonal "
               "structure is what the models have to discover. Blank cells are unobserved.")
    c = st.columns(3)
    c[0].bar_chart(ds.ratings.rating.value_counts().sort_index(), height=200)
    c[0].caption("Rating distribution")
    c[1].bar_chart(ds.ratings.groupby("user_id").size().value_counts().sort_index(), height=200)
    c[1].caption("Ratings per user")
    c[2].bar_chart(ds.ratings.merge(ds.games, on="item_id").groupby("title").size().sort_values(), height=200)
    c[2].caption("Ratings per game (popularity)")

elif page == "mf":
    params = controls("mf")
    r = embedding_section("mf", params, threshold=3)
    explainer("mf", """
The loss uses **observed entries only**. Each user is a row of U and each game a row of V. Training
pulls a user toward the games they rated highly and pushes them away from the ones they rated low.

**What to try**
- **k = 2**: no PCA needed, so you see the raw space. Every user and game is a 2-D arrow, just like
  the *open-world level / action level* example on the slides.
- **Raise k**: train RMSE drops, but test RMSE stops improving and then rises. That's overfitting on
  ~450 ratings.
- **Cold start**: the open markers (users with no ratings) and the white *Silent Hill f* never move.
  The loss has no gradient for them, so only L2 acts and pulls them toward the origin.
- **Biases**: without them, U·V has to produce the absolute star level (≈3), so everything leans in a
  shared "mean" direction. With biases on, the embeddings only need to encode taste.
""")
    with st.expander("Predicted rating matrix vs observed", expanded=False):
        order = users_by_likes()
        gord = ds.games.sort_values("category").item_id.to_numpy()
        obs = np.full((ds.n_users, ds.n_items), "", dtype=object)
        obs[ds.train.user_id, ds.train.item_id] = [f"{x}★" for x in ds.train.rating]
        obs[ds.test.user_id, ds.test.item_id] = [f"({x})" for x in ds.test.rating]
        fig = go.Figure(go.Heatmap(
            z=r.scores[order][:, gord], x=ds.games.title.iloc[gord], y=[f"u{u} · {likes_of(u)}" for u in order],
            text=obs[order][:, gord], texttemplate="%{text}", colorscale="Blues", zmin=1, zmax=5,
            hovertemplate="%{y}<br>%{x}<br>predicted %{z:.2f}<br>observed %{text}<extra></extra>",
            xgap=1, ygap=1))
        fig.update_layout(height=1000, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Color is the prediction U·Vᵀ for every cell. Text shows train ratings (★) and held-out "
                   "ratings (in parentheses). Every cell gets a prediction, including unobserved ones.")

elif page == "logmf":
    params = controls("logmf")
    thr = dict(params)["threshold"]
    r = embedding_section("logmf", params, threshold=thr)
    explainer("logmf", f"""
Same model, but the label is now **y = 1 if rating > {thr}**. Ratings ≤ {thr} are *real* negatives,
because this dataset has explicit ratings. With implicit feedback (clicks, plays), unobserved cells
are not negatives, so negatives would have to be sampled.

**Same gradient as RMSE**: the residual (ŷ − y) multiplies the embedding. The only change is that ŷ
goes from linear to σ(s). Compare the space with page 1: positives pull users toward games just
like before, but a 1★ and a 3★ rating now push equally hard.
""")
    s, y = r.extras["train_scores"], r.extras["train_labels"]
    xs = np.linspace(min(s.min(), -6), max(s.max(), 6), 200)
    jitter = np.random.default_rng(0).normal(0, 0.02, len(y))
    fig = go.Figure([
        go.Scatter(x=xs, y=1 / (1 + np.exp(-xs)), mode="lines", name="σ(s)",
                   line=dict(color="#2a78d6", width=2)),
        go.Scatter(x=s[y == 0], y=y[y == 0] + jitter[y == 0], mode="markers", name=f"disliked (≤{thr}★)",
                   marker=dict(color="#52514e", size=7, opacity=0.6)),
        go.Scatter(x=s[y == 1], y=y[y == 1] + jitter[y == 1], mode="markers", name=f"liked (>{thr}★)",
                   marker=dict(color="#1baf7a", size=7, opacity=0.6, symbol="diamond")),
    ])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Train pairs on the sigmoid",
                      xaxis_title="s = U_u · V_i", yaxis_title="ŷ = σ(s)")
    st.plotly_chart(fig, width="stretch")

elif page == "softmax":
    params = controls("softmax")
    p = dict(params)
    r = embedding_section("softmax", params, threshold=p["threshold"])
    explainer("softmax", """
The user is now a **function of features** (user_id, platform, age, favorite category, bag of liked
games) instead of a table row. Every game is a row of **W**, and all 20 games **compete** in one
softmax. A user's vector moves toward their positives and away from wherever the model currently
puts probability mass: −∇ᵤL = 𝔼_q[w] − 𝔼_p[w].

**Temperature τ** divides the logits. A small τ gives a sharp, peaky p, which pushes positives and
negatives far apart. A large τ gives a flat p and weak gradients, so the space stays mushy.

**The plot uses cosine view by default.** The softmax only cares about the direction of u, so
the encoder output grows far longer than W's rows. In a raw-scale plot every game would bunch up at the
origin. Untick *cosine view* in the sidebar to see this.

**History leakage**: during training, each user's positives are randomly split each epoch, half as
input history and half as target, so the encoder can't simply copy its input.

**Cold start (users)**: the open markers have no history and an untrained user_id, yet they land near
their stated favorite category thanks to the side features. Games are still table rows, so the new
release (white) stays near where it started.
""")
    if focus is not None:
        probs = r.extras["probs"][focus]
        o = np.argsort(-probs)
        liked = set(viz.liked_items(ds, p["threshold"]).get(focus, []))
        fig = go.Figure(go.Bar(
            x=probs[o], y=ds.games.title.iloc[o], orientation="h",
            marker=dict(color=[viz.COLORS[c] for c in ds.games.category.iloc[o]]),
            text=[("✓ liked  " if i in liked else "") + f"{probs[i]:.0%}" for i in o],
            textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        ent = float(-(probs * np.log(probs + 1e-12)).sum())
        fig.update_layout(height=520, margin=dict(l=10, r=40, t=40, b=10),
                          yaxis=dict(autorange="reversed"),
                          title=f"p = softmax(z) for user {focus} · entropy {ent:.2f} "
                                f"(uniform = {np.log(ds.n_items):.2f})")
        st.plotly_chart(fig, width="stretch")
        st.caption("Change τ and retrain to see the distribution sharpen or flatten. ✓ marks games the user "
                   "liked in train. Those are fed as history, so they score high.")

elif page == "two_tower":
    params = controls("two_tower")
    p = dict(params)
    r = embedding_section("two_tower", params, threshold=p["threshold"])
    explainer("two_tower", """
Both sides are now **encoders**: games get an item tower over item_id, tags and play mode. Each batch
holds **B = 4 (user, positive game) pairs**. The diagonal of the 4×4 score matrix holds the positives
and the 12 off-diagonal cells are free negatives. Vectors are **L2-normalized**, so the score is
cosine/τ and everything lives on a unit sphere. With k = 3, the 2-D projection shows that sphere
seen from above: users and games sit on a ring.

**logQ**: popular games show up as in-batch negatives more often, so without the correction they get
pushed away from everyone. Subtracting log q_j cancels that sampling bias. Toggle it and watch where
popular games end up.

**Temperature**: with normalized vectors, τ sets how hard the model separates. τ = 1 caps logits at
±1 and the space barely organizes. τ ≈ 0.05–0.2 is typical.

**Cold start (both sides)**: the white new release lands among its horror/story neighbors because the
item tower sees its tags.

*Masking*: with only 20 games, two pairs in a batch often share the same game. That off-diagonal cell
would be a false negative, so it's masked out.
""")
    b = r.extras["batch"]
    ulab = [f"u{u}" for u in b["users"]]
    ilab = [viz._short(ds.games.title.iloc[i]) for i in b["items"]]
    c1, c2 = st.columns(2)
    for col, key, ttl in [(c1, "logits", "s_ij = cos/τ − log q_j"), (c2, "probs", "row softmax")]:
        z = b[key]
        fig = go.Figure(go.Heatmap(z=z, x=[f"v: {t}" for t in ilab], y=ulab, colorscale="Blues",
                                   text=np.round(z, 2), texttemplate="%{text}", showscale=False,
                                   xgap=2, ygap=2))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), title=f"Example batch: {ttl}",
                          yaxis=dict(autorange="reversed"))
        col.plotly_chart(fig, width="stretch")
    st.caption("A fresh batch of 4 positive pairs scored by the trained towers. Diagonal = positives, "
               "off-diagonal = in-batch negatives. A well-trained model puts most of each row's mass "
               "on its diagonal cell.")

elif page == "fm":
    params = controls("fm")
    p = dict(params)
    r = embedding_section("fm", params, threshold=p["threshold"])
    explainer("fm", """
Every **feature value** (user_id=7, platform=PS5, tag=horror, mode=multi, …) gets its own vector
v. The prediction sums the dot products over all pairs of active features. Grouping the pairs into
user-side and item-side gives ŷ = c_u + b_i + ⟨e_u, e_i⟩. So FM is a two-tower model whose towers
are **sums**, trained **pointwise** (MSE or BCE) like MF.

- **Special case**: keep only user_id and item_id and you get MF with biases.
- **Cold start on both sides**: the new release is placed by Σ v(tag=horror, story, stealth). A new
  user is placed by Σ v(platform, age, favorite). Uncheck *Include user_id / item_id* to get a pure
  content model in which users with identical features collapse to the same point.
""")
    names, Vf = r.extras["feature_names"], r.extras["feature_V"]
    keep = [i for i, n in enumerate(names) if not n.startswith(("user_id=", "item_id="))]
    norms = np.linalg.norm(Vf[keep], axis=1)
    o = np.argsort(-norms)
    fig = go.Figure(go.Bar(x=[names[keep[i]] for i in o], y=norms[o], marker_color="#2a78d6",
                           hovertemplate="%{x}<br>‖v‖ = %{y:.3f}<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                      title="Side-feature embedding norms ‖v‖ (how much each feature moves e_u / e_i)")
    st.plotly_chart(fig, width="stretch")

elif page == "compare":
    st.markdown("All five models trained with their **default** settings, shown at their final epoch. "
                "Each model's page has its own controls and the training animation.")
    rows, figs = [], {}
    for name in ["mf", "logmf", "softmax", "two_tower", "fm"]:
        params = tuple(sorted(DEFAULTS[name].items()))
        r = train(name, params)
        uxy, ixy, _ = projected(name, params, method, name in COSINE_DEFAULT)
        figs[name] = viz.embedding_figure(ds, uxy[-1:], ixy[-1:], [r.epochs[-1]], focus_user=focus,
                                          liked=viz.liked_items(ds, DEFAULTS[name].get("threshold", 3)),
                                          height=460, title=PAGES[name])
        rows.append({"model": PAGES[name], **{k: round(v, 3) for k, v in r.metrics.items()}})
    st.dataframe(pd.DataFrame(rows).set_index("model"), width="stretch")
    st.caption("recall@5: of each user's held-out liked games (> 3★), the share found in the top 5 among "
               "games not rated in train. It's comparable across all five models. RMSE and AUC only "
               "apply to the pointwise models.")
    st.markdown(" ".join(
        f"<span style='color:{viz.COLORS[c]}'>■</span> {c} &nbsp;" for c in data.CATEGORIES)
        + "<br><small>Small markers = users (colored by the category they rate highest) · large outlined = games · "
          "open = cold-start user · white = new game · ★ = focus user</small>", unsafe_allow_html=True)
    names = list(figs)
    for i in range(0, len(names), 2):
        cols = st.columns(2)
        for c, n in zip(cols, names[i:i + 2]):
            figs[n].update_layout(showlegend=False)
            c.plotly_chart(figs[n], width="stretch")
    st.subheader("From MF to two-tower: what changes?")
    st.dataframe(pd.DataFrame([
        ["MF", "embedding table", "embedding table", "RMSE / logistic (pointwise)", "No", "No"],
        ["FM", "Σ of feature embeddings", "Σ of feature embeddings", "RMSE / logistic (pointwise)", "Yes*", "Yes*"],
        ["Softmax", "encoder over user features", "table (rows of W)", "softmax CE (listwise)", "Yes*", "No"],
        ["Two-tower", "encoder over user features", "encoder over item features", "sampled softmax + logQ",
         "Yes*", "Yes*"],
    ], columns=["Model", "User repr.", "Item repr.", "Loss", "Cold start (user)", "Cold start (item)"]),
        hide_index=True, width="stretch")
    st.caption("* Only when the new user or item comes with features. What is the embedding a function of? "
               "table → sum → encoder. What loss shapes the space? pointwise → listwise → contrastive.")
