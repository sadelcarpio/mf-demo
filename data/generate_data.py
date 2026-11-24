"""Generate the small synthetic videogame dataset used by the demo.

Run:  uv run python data/generate_data.py

Writes games.csv, users.csv and ratings.csv next to this file. Deterministic (fixed seed).

How it is built
---------------
* 20 games in 5 categories, each with 2-3 tags and a play mode.
* 60 users with features: platform, age group and favorite category. Internally
  the generator gives each user a taste profile (one of the 5 categories) to
  produce ratings that have structure. The profile is not saved; it only drives
  the ratings and correlates with the features.
* Warm users rate 7-12 games. They mostly pick games they like, as people do, but
  they also rate some they don't, which gives real 1-2 star negatives.
* Cold start: the last 4 users have features but no ratings, and the game
  "Silent Hill f" (a new release) has tags but no ratings. MF cannot place them in
  the space. The feature-based models (softmax, two-tower, FM) can.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_USERS = 60
N_COLD_USERS = 4
OUT = Path(__file__).parent

CATEGORIES = ["Action-Adventure", "RPG", "Horror", "Competitive", "Family"]

# title, category, tags, mode
GAMES = [
    ("God of War Ragnarök", "Action-Adventure", "action|adventure|story", "single"),
    ("Ghost of Tsushima", "Action-Adventure", "action|adventure|open_world", "single"),
    ("The Last of Us Part II", "Action-Adventure", "action|story|stealth", "single"),
    ("Uncharted 4", "Action-Adventure", "action|adventure|story", "single"),
    ("The Witcher 3", "RPG", "rpg|open_world|story", "single"),
    ("Elden Ring", "RPG", "rpg|action|open_world", "both"),
    ("Cyberpunk 2077", "RPG", "rpg|open_world|shooter", "single"),
    ("Baldur's Gate 3", "RPG", "rpg|strategy|story", "both"),
    ("Resident Evil 4", "Horror", "horror|action|shooter", "single"),
    ("Dead Space", "Horror", "horror|shooter|story", "single"),
    ("Alan Wake 2", "Horror", "horror|story|adventure", "single"),
    ("Silent Hill f", "Horror", "horror|story|stealth", "single"),  # cold item
    ("Call of Duty: MW III", "Competitive", "shooter|multiplayer|action", "multi"),
    ("Valorant", "Competitive", "shooter|multiplayer|strategy", "multi"),
    ("EA Sports FC 24", "Competitive", "sports|multiplayer", "both"),
    ("Rocket League", "Competitive", "sports|racing|multiplayer", "multi"),
    ("Mario Kart 8 Deluxe", "Family", "racing|party|multiplayer", "both"),
    ("Zelda: Tears of the Kingdom", "Family", "adventure|open_world|puzzle", "single"),
    ("Animal Crossing", "Family", "cozy|simulation", "both"),
    ("Stardew Valley", "Family", "cozy|simulation|rpg", "both"),
]
COLD_ITEM = "Silent Hill f"

# How much each taste segment likes each tag (unlisted tags are 0).
SEGMENT_TASTE = {
    "Action-Adventure": {"action": 1.0, "adventure": 1.0, "story": 0.7, "stealth": 0.6, "open_world": 0.4},
    "RPG": {"rpg": 1.2, "open_world": 0.9, "story": 0.6, "strategy": 0.4},
    "Horror": {"horror": 1.4, "story": 0.5, "shooter": 0.3, "stealth": 0.4},
    "Competitive": {"multiplayer": 1.1, "shooter": 0.9, "sports": 0.9, "racing": 0.5, "strategy": 0.3},
    "Family": {"cozy": 1.2, "party": 1.0, "puzzle": 0.8, "simulation": 0.8, "racing": 0.4, "adventure": 0.3},
}
DISLIKES = {  # mild negative tastes make the 1-2 star ratings meaningful
    "Action-Adventure": {"cozy": -0.6, "sports": -0.4},
    "RPG": {"sports": -0.6, "party": -0.4},
    "Horror": {"cozy": -0.7, "party": -0.5, "sports": -0.4},
    "Competitive": {"story": -0.4, "cozy": -0.6, "puzzle": -0.3},
    "Family": {"horror": -1.0, "shooter": -0.6},
}
PLATFORM_BY_SEGMENT = {  # P(PS5, Xbox, PC, Switch)
    "Action-Adventure": [0.6, 0.15, 0.2, 0.05],
    "RPG": [0.3, 0.2, 0.45, 0.05],
    "Horror": [0.45, 0.25, 0.25, 0.05],
    "Competitive": [0.25, 0.3, 0.4, 0.05],
    "Family": [0.05, 0.05, 0.1, 0.8],
}
AGE_BY_SEGMENT = {  # P(13-17, 18-24, 25-34, 35+)
    "Action-Adventure": [0.1, 0.3, 0.4, 0.2],
    "RPG": [0.1, 0.35, 0.4, 0.15],
    "Horror": [0.1, 0.4, 0.35, 0.15],
    "Competitive": [0.4, 0.4, 0.15, 0.05],
    "Family": [0.3, 0.15, 0.2, 0.35],
}
PLATFORMS = ["PS5", "Xbox", "PC", "Switch"]
AGES = ["13-17", "18-24", "25-34", "35+"]


def main() -> None:
    rng = np.random.default_rng(SEED)

    games = pd.DataFrame(GAMES, columns=["title", "category", "tags", "mode"])
    games.insert(0, "item_id", np.arange(len(games)))
    all_tags = sorted({t for tags in games.tags for t in tags.split("|")})

    # item tag matrix, normalized so 2-tag and 3-tag games are comparable
    tag_mat = np.array([[t in tags.split("|") for t in all_tags] for tags in games.tags], float)
    tag_mat /= np.sqrt(tag_mat.sum(1, keepdims=True))
    quality = rng.normal(0, 0.25, len(games))  # some games are just better

    users, ratings = [], []
    segments = np.array(CATEGORIES)[np.arange(N_USERS) % len(CATEGORIES)]
    rng.shuffle(segments)
    for uid, seg in enumerate(segments):
        fav = seg if rng.random() < 0.7 else rng.choice([c for c in CATEGORIES if c != seg])
        users.append(
            dict(
                user_id=uid,
                platform=rng.choice(PLATFORMS, p=PLATFORM_BY_SEGMENT[seg]),
                age_group=rng.choice(AGES, p=AGE_BY_SEGMENT[seg]),
                favorite_category=fav,
                segment=seg,
            )
        )
        if uid >= N_USERS - N_COLD_USERS:
            continue  # cold-start user: no interactions

        taste = np.array([SEGMENT_TASTE[seg].get(t, 0) + DISLIKES[seg].get(t, 0) for t in all_tags])
        taste = taste + rng.normal(0, 0.25, len(all_tags))  # personal twist
        affinity = tag_mat @ taste + quality

        candidates = games.index[games.title != COLD_ITEM].to_numpy()
        n_rated = rng.integers(7, 13)
        # people mostly play games they expect to like, plus a few they don't
        p = np.exp(1.2 * affinity[candidates])
        p = 0.75 * p / p.sum() + 0.25 / len(candidates)
        rated = rng.choice(candidates, size=n_rated, replace=False, p=p)
        for iid in rated:
            z = 2.2 * (affinity[iid] - 0.4) + rng.normal(0, 0.45)
            stars = int(np.clip(np.round(3 + 1.3 * z), 1, 5))
            ratings.append(dict(user_id=uid, item_id=int(iid), rating=stars))

    users = pd.DataFrame(users)
    ratings = pd.DataFrame(ratings).sort_values(["user_id", "item_id"])

    games.to_csv(OUT / "games.csv", index=False)
    users.drop(columns="segment").to_csv(OUT / "users.csv", index=False)
    ratings.to_csv(OUT / "ratings.csv", index=False)

    print(f"{len(games)} games, {len(users)} users, {len(ratings)} ratings")
    print("rating distribution:", ratings.rating.value_counts().sort_index().to_dict())
    print(f"positives (>3): {(ratings.rating > 3).mean():.0%}")
    print(ratings.merge(users, on="user_id").merge(games, on="item_id")
          .pivot_table(index="segment", columns="category", values="rating").round(1))


if __name__ == "__main__":
    main()
