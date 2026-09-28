"""Apply/embeddings — per-user train/valid/test split of the flat triples.

Rule (target relation = `rated`; everything else -- content + `watched` -- stays in train):
  - per user, shuffle their `rated` edges (seeded, users in sorted order)
  - test  = max(1, floor(10% of n)) edges
  - valid = max(1, floor(10% of n)) edges, or 0 if the user has a single rating
  - the rest -> train
  - valid/test edges whose anime has no train interaction are dropped (cold start:
    a shallow model has no row for them, so they can't be scored fairly)
Per-user, not a random edge split: every test user keeps ~89% of their profile in train.

Reproducibility note: the split in data/generated/split/ was made with this rule by an
earlier version of this script that was lost; the random draw can't be recovered, so
that split is shipped as data. This script regenerates an equivalent split
(same rule, different draw) -- see Apply/notes.md for the check that results hold.

Run:  python src/make_split.py --out data/generated/split_regen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_eval import APPLY, INTERACTIONS

TRIPLES = APPLY / "data" / "generated" / "triples.tsv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=APPLY / "data" / "generated" / "split")
    p.add_argument("--frac", type=float, default=0.1, help="share of each user's ratings per holdout set")
    p.add_argument("--seed", type=int, default=26)
    p.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = p.parse_args()
    if (args.out / "train.tsv").exists() and not args.force:
        raise SystemExit(f"{args.out} already holds a split (the shipped one?) -- use --out or --force")

    full = pd.read_csv(TRIPLES, sep="\t", names=["h", "r", "t"], dtype=str)
    rated = full[full.r == "rated"]
    rng = np.random.default_rng(args.seed)

    test_idx, valid_idx = [], []
    for _, g in rated.groupby("h", sort=True):
        n = len(g)
        k = max(1, int(n * args.frac))
        perm = g.index.values[rng.permutation(n)]
        test_idx += list(perm[:k])
        valid_idx += list(perm[k:k + (k if n >= 2 else 0)])

    held = set(test_idx) | set(valid_idx)
    train = full[~full.index.isin(held)]
    inter_anime = set(train[train.r.isin(INTERACTIONS)].t)

    stats = {"seed": args.seed, "valid_frac": args.frac, "test_frac": args.frac,
             "target_relation": "rated"}
    out = {"train": train}
    for name, idx in (("valid", valid_idx), ("test", test_idx)):
        df = full.loc[sorted(idx)]
        keep = df.t.isin(inter_anime)
        stats[f"{name}_cold_start_dropped"] = int((~keep).sum())
        out[name] = df[keep]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(args.out / f"{name}.tsv", sep="\t", header=False, index=False)
        stats[f"{name}_triples"] = len(df)
    stats["train_target_edges"] = int((train.r == "rated").sum())
    stats["users"] = int(out["test"].h.nunique())
    stats["candidate_anime"] = len(inter_anime)
    (args.out / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
