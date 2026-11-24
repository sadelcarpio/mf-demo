"""Load the CSVs and turn them into the tensors and index maps the models need."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATEGORIES = ["Action-Adventure", "RPG", "Horror", "Competitive", "Family"]


@dataclass
class Dataset:
    games: pd.DataFrame
    users: pd.DataFrame
    likes: pd.DataFrame  # per user: category they rate highest on average (plot coloring only)
    ratings: pd.DataFrame  # all observed ratings
    train: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    # categorical side features as integer codes
    user_feats: dict[str, np.ndarray]  # name -> (n_users,) codes
    user_vocab: dict[str, list[str]]
    item_tags: np.ndarray  # (n_items, n_tags) multi-hot
    tag_vocab: list[str]
    item_mode: np.ndarray  # (n_items,) codes
    mode_vocab: list[str]
    cold_users: np.ndarray
    cold_items: np.ndarray

    def train_matrix(self, threshold: float | None = None) -> np.ndarray:
        """Dense (n_users, n_items) matrix of train ratings (0 = unobserved),
        or a 0/1 matrix of positives (rating > threshold) if a threshold is given."""
        m = np.zeros((self.n_users, self.n_items), np.float32)
        r = self.train.rating.to_numpy(np.float32)
        if threshold is not None:
            r = (r > threshold).astype(np.float32)
        m[self.train.user_id, self.train.item_id] = r
        return m


def _likes(users: pd.DataFrame, games: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """For coloring users in the plots: the game category each user rates highest on average
    (ties go to the category with more ratings). Users with no ratings get None."""
    per_cat = (ratings.merge(games[["item_id", "category"]], on="item_id")
               .groupby(["user_id", "category"]).rating.agg(["mean", "size"]).reset_index()
               .sort_values(["user_id", "mean", "size"], ascending=[True, False, False])
               .drop_duplicates("user_id"))
    out = users[["user_id"]].merge(per_cat, on="user_id", how="left")
    out = out.rename(columns={"category": "likes", "mean": "likes_avg"})[["user_id", "likes", "likes_avg"]]
    out["likes"] = out["likes"].astype(object).where(out["likes"].notna(), None)
    return out.set_index("user_id")


def load(test_frac: float = 0.15, seed: int = 0) -> Dataset:
    games = pd.read_csv(DATA_DIR / "games.csv")
    users = pd.read_csv(DATA_DIR / "users.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    likes = _likes(users, games, ratings)

    # Hold out ~15% of each warm user's ratings for evaluation (at least 5 stay in train).
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, g in ratings.groupby("user_id"):
        n_test = min(int(round(len(g) * test_frac)), len(g) - 5)
        if n_test > 0:
            test_idx += list(rng.choice(g.index, n_test, replace=False))
    test = ratings.loc[sorted(test_idx)]
    train = ratings.drop(index=test.index)

    user_feats, user_vocab = {}, {}
    for col in ["platform", "age_group", "favorite_category"]:
        vocab = sorted(users[col].unique()) if col != "favorite_category" else CATEGORIES
        user_vocab[col] = list(vocab)
        user_feats[col] = users[col].map({v: i for i, v in enumerate(vocab)}).to_numpy()

    tag_vocab = sorted({t for tags in games.tags for t in tags.split("|")})
    item_tags = np.array(
        [[t in tags.split("|") for t in tag_vocab] for tags in games.tags], np.float32
    )
    mode_vocab = sorted(games["mode"].unique())
    item_mode = games["mode"].map({v: i for i, v in enumerate(mode_vocab)}).to_numpy()

    rated_users = set(ratings.user_id)
    rated_items = set(ratings.item_id)
    return Dataset(
        games=games,
        users=users,
        likes=likes,
        ratings=ratings,
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        n_users=len(users),
        n_items=len(games),
        user_feats=user_feats,
        user_vocab=user_vocab,
        item_tags=item_tags,
        tag_vocab=tag_vocab,
        item_mode=item_mode,
        mode_vocab=mode_vocab,
        cold_users=np.array([u for u in users.user_id if u not in rated_users]),
        cold_items=np.array([i for i in games.item_id if i not in rated_items]),
    )
