"""Apply/embeddings — shared recommender evaluation + popularity baseline.

Every model (KGE, GNN, baseline) is judged the same way, so the numbers compare:
  - candidates = anime with >=1 train interaction (rated or watched)
  - for each user, rank ALL candidates, after masking what the user already
    interacted with in train (and in valid, when scoring test)  -> "filtered"
  - Recall@K, NDCG@K, HR@K, averaged over users (not edges; heavy users would
    otherwise dominate)

A model plugs in via `score_fn(user_names: list[str]) -> np.ndarray` of shape
(len(user_names), len(candidates)), higher = more likely.

Run (popularity baseline):  python src/recsys_eval.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

APPLY = Path(__file__).resolve().parent.parent
SPLIT = APPLY / "data" / "generated" / "split"
RESULTS = APPLY / "data" / "generated" / "results"
GEN_MODELS = APPLY / "data" / "generated" / "models"

# user->anime relations. Both count as "the user touched it" for candidates and
# masking; only `rated` is a prediction target (valid/test hold `rated` only).
INTERACTIONS = ("rated", "watched")


def load_split(split_dir: Path = SPLIT) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(split_dir / f"{name}.tsv", sep="\t",
                          names=["h", "r", "t"], dtype=str)
        for name in ("train", "valid", "test")
    }


def subset_split(split: dict[str, pd.DataFrame], n_users: int,
                 seed: int = 26) -> dict[str, pd.DataFrame]:
    """Keep `n_users` random users' interactions (+ all content edges) -- for fast trials.

    Whole users, same reason as build_triples.py: a profile is the unit of signal.
    """
    train = split["train"]
    users = train[train.r.isin(INTERACTIONS)].h.unique()
    keep = set(np.random.default_rng(seed).choice(users, n_users, replace=False))
    is_inter = train.r.isin(INTERACTIONS)
    out = {"train": train[~is_inter | train.h.isin(keep)]}
    for name in ("valid", "test"):
        out[name] = split[name][split[name].h.isin(keep)]
    return out


class Evaluator:
    def __init__(self, split: dict[str, pd.DataFrame], k: int = 10):
        self.k = k
        train = split["train"]
        inter = train[train.r.isin(INTERACTIONS)]

        self.candidates = sorted(set(inter.t))
        self.cand_idx = {a: i for i, a in enumerate(self.candidates)}

        # everything a user already touched in train -> masked out of the ranking
        self.seen = defaultdict(set)
        for u, a in zip(inter.h, inter.t):
            self.seen[u].add(self.cand_idx[a])

        self.valid = self._targets(split["valid"])
        self.test = self._targets(split["test"])

    def _targets(self, df: pd.DataFrame) -> dict[str, set[int]]:
        out = defaultdict(set)
        for u, a in zip(df.h, df.t):
            if a in self.cand_idx:  # split already dropped cold-start anime
                out[u].add(self.cand_idx[a])
        return dict(out)

    def evaluate(self, score_fn: Callable[[list[str]], np.ndarray],
                 on: str = "test", batch: int = 512) -> dict:
        targets = self.test if on == "test" else self.valid
        users = sorted(targets)
        k = self.k
        discount = 1.0 / np.log2(np.arange(2, k + 2))

        recall, ndcg, hit = [], [], []
        for s in range(0, len(users), batch):
            ub = users[s:s + batch]
            scores = np.asarray(score_fn(ub), dtype=np.float32).copy()
            for row, u in enumerate(ub):
                mask = list(self.seen.get(u, ()))
                if on == "test":  # valid positives are known by test time
                    mask += list(self.valid.get(u, ()))
                scores[row, mask] = -np.inf

            # top-k without a full sort: argpartition, then order just those k
            top = np.argpartition(-scores, k, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
            top = np.take_along_axis(top, order, axis=1)

            for i, u in enumerate(ub):
                tgt = targets[u]
                rel = np.array([t in tgt for t in top[i]], dtype=float)
                n_hit = rel.sum()
                # min(k, |tgt|): a user with 3 test items can score recall 1.0
                recall.append(n_hit / min(k, len(tgt)))
                ideal = discount[:min(k, len(tgt))].sum()
                ndcg.append(float((rel * discount).sum() / ideal))
                hit.append(float(n_hit > 0))

        return {f"recall@{k}": float(np.mean(recall)),
                f"ndcg@{k}": float(np.mean(ndcg)),
                f"hr@{k}": float(np.mean(hit)),
                "users": len(users)}


def save_result(name: str, metrics: dict, extra: dict | None = None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(
        json.dumps({"model": name, **metrics, **(extra or {})}, indent=2))


def main() -> None:
    split = load_split()
    ev = Evaluator(split)

    # popularity = train interaction count per candidate; same list for everyone
    train = split["train"]
    pop = train[train.r.isin(INTERACTIONS)].t.value_counts()
    a = np.array([pop.get(c, 0) for c in ev.candidates], dtype=np.float32)

    metrics = ev.evaluate(lambda users: np.tile(a, (len(users), 1)))
    print("popularity:", metrics)
    save_result("popularity", metrics)


if __name__ == "__main__":
    main()
