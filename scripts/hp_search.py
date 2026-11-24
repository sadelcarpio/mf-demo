"""Small grid search for demo defaults that produce clean clusters in the 2-D plot.

Run:  uv run python scripts/hp_search.py [model ...]

For every config the search trains with 2 seeds and projects the final epoch the way
the app shows it by default (PCA; cosine view for softmax and two-tower). It then
measures, in that 2-D view:

  purity@3    for each user with ratings: the share of their 3 nearest games that are
              in the category they rate highest
  silhouette  how tightly games group by category (cold game excluded)
  recall@5    held-out positives found in the top 5 (from the full model scores)

Selection: drop configs whose recall@5 is more than 0.05 below the model's best, then
rank by purity@3 + silhouette.
"""

import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COSINE = {"softmax", "two_tower"}
SEEDS = [0, 1]

GRIDS = {
    "mf": dict(k=[2, 3, 4, 8], l2=[0.05, 0.1, 0.2, 0.5], lr=[0.03, 0.05, 0.1],
               epochs=[150, 300, 600], biases=[False, True]),
    "logmf": dict(k=[2, 3, 4, 8], l2=[0.01, 0.05, 0.1, 0.2], lr=[0.03, 0.05, 0.1],
                  epochs=[150, 300, 600], threshold=[3]),
    "softmax": dict(k=[2, 3, 4, 8], temperature=[0.2, 0.5, 1.0, 2.0], lr=[0.003, 0.01, 0.03],
                    epochs=[150, 300, 600], threshold=[3], use_id=[True], use_history=[True]),
    "two_tower": dict(k=[2, 3, 4, 8], temperature=[0.05, 0.1, 0.2, 0.3], lr=[0.003, 0.005, 0.01],
                      epochs=[20, 40, 80], threshold=[3], logq=[True], use_id=[True], use_history=[True]),
    "fm": dict(k=[2, 3, 4, 8], l2=[0.01, 0.05, 0.1, 0.2], lr=[0.01, 0.03, 0.1],
               epochs=[150, 300, 600], logistic=[False], threshold=[3], use_ids=[True]),
}


def cluster_scores(ds, U2, V2):
    from sklearn.metrics import silhouette_score

    likes = ds.likes.likes.reindex(ds.users.user_id).to_numpy()
    cats = ds.games.category.to_numpy()
    warm = [u for u in range(ds.n_users) if likes[u] is not None]
    d = np.linalg.norm(U2[warm, None, :] - V2[None, :, :], axis=2)
    nearest = np.argsort(d, axis=1)[:, :3]
    purity = float(np.mean([(cats[nearest[j]] == likes[u]).mean() for j, u in enumerate(warm)]))
    keep = np.setdiff1d(np.arange(ds.n_items), ds.cold_items)
    sil = float(silhouette_score(V2[keep], cats[keep])) if np.ptp(V2[keep]) > 0 else -1.0
    return purity, sil


def run(job):
    import torch

    torch.set_num_threads(1)
    from mfdemo import data, models, viz

    name, params = job
    ds = data.load()
    out = []
    for seed in SEEDS:
        r = models.TRAINERS[name](ds, seed=seed, **params)
        u, v, _ = viz.project(r.user_snaps[-1:], r.item_snaps[-1:], "PCA", name in COSINE)
        purity, sil = cluster_scores(ds, u[-1], v[-1])
        out.append((purity, sil, r.metrics["test recall@5"]))
    p, s, rec = np.mean(out, axis=0)
    return {"model": name, **params, "purity@3": p, "silhouette": s, "recall@5": rec}


def main(names):
    jobs = [(n, dict(zip(GRIDS[n], combo))) for n in names for combo in itertools.product(*GRIDS[n].values())]
    print(f"{len(jobs)} configs x {len(SEEDS)} seeds")
    with Pool(12) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    out_dir = Path(__file__).parent
    for n in names:
        d = df[df.model == n].dropna(axis=1, how="all").copy()
        d = d[d["recall@5"] >= d["recall@5"].max() - 0.05]
        d["score"] = d["purity@3"] + d["silhouette"]
        d = d.sort_values("score", ascending=False)
        print(f"\n=== {n}  (best recall@5 {df[df.model == n]['recall@5'].max():.3f})")
        print(d.head(8).round(3).to_string(index=False))
    df.to_csv(out_dir / "hp_results.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or list(GRIDS))
