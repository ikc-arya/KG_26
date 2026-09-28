"""Apply/embeddings — look inside the trained models, by title.

Metrics say *how well*; this says *what* each model learned:
  1. nearest-neighbour anime for a few well-known titles
       DistMult: cosine similarity (score is a dot product -> direction matters)
       TransE:   euclidean distance (score is a distance -> position matters)
  2. one user's history vs each model's top-10, with test hits marked
  3. the same behaviour aggregated over ALL users -- a single example can mislead
     (user_66835 made TransE look niche-obsessed; the aggregate says otherwise)

Run after train_kge.py for both models:  python src/probe_embeddings.py
Writes data/generated/results/probe.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from recsys_eval import APPLY, GEN_MODELS, INTERACTIONS, RESULTS, Evaluator, load_split
from train_kge import DistMult, TransE

PROBE_ANIME = [1, 1535, 5680, 199]  # Cowboy Bebop, Death Note, K-On!, Spirited Away
N = 5


def load_model(name: str):
    ck = torch.load(GEN_MODELS / f"{name}.pt")
    a = ck["args"]
    n_ent, n_rel = len(ck["ents"]), len(ck["rels"])
    if a.get("model", "distmult") == "transe":  # early checkpoints predate --model
        m = TransE(n_ent, n_rel, a["dim"], a["init_std"], a["gamma"])
    else:
        m = DistMult(n_ent, n_rel, a["dim"], a["init_std"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, {e: i for i, e in enumerate(ck["ents"])}, ck["rels"].index("rated")


def main() -> None:
    titles = pd.read_csv(APPLY / "data" / "anime.csv", usecols=["mal_id", "title"])
    title = dict(zip("anime_" + titles.mal_id.astype(str), titles.title))
    ev = Evaluator(load_split())
    cands = ev.candidates
    models = {n: load_model(n) for n in ("distmult", "transe")}
    out = ["# Embedding probe\n"]

    # ---- 1. nearest neighbours ----
    out.append(f"## Nearest-neighbour anime (top {N})\n")
    for mid in PROBE_ANIME:
        q = f"anime_{mid}"
        out.append(f"**{title[q]}**\n")
        out.append("| DistMult (cosine) | TransE (euclidean) |\n|---|---|")
        cols = []
        for name, (m, e2i, _) in models.items():
            with torch.no_grad():
                E = m.ent(torch.tensor([e2i[a] for a in cands]))
                v = m.ent(torch.tensor(e2i[q]))
                if name == "transe":
                    sim = -(E - v).norm(dim=1)
                else:
                    sim = torch.nn.functional.cosine_similarity(E, v[None], dim=1)
            order = [cands[i] for i in sim.argsort(descending=True).tolist() if cands[i] != q]
            cols.append([title.get(a, a) for a in order[:N]])
        out += [f"| {d} | {t} |" for d, t in zip(*cols)]
        out.append("")

    # ---- 2. one user, end to end ----
    rng = np.random.default_rng(26)
    ok = [u for u in sorted(ev.test) if 50 <= len(ev.seen[u]) <= 150 and len(ev.test[u]) >= 5]
    u = ok[rng.integers(len(ok))]
    uid = int(u.split("_")[1])
    ratings = pd.read_csv(APPLY / "data" / "rating.csv")
    ratings = ratings[ratings.user_id == uid]
    seen = {cands[i] for i in ev.seen[u]}
    hist = ratings[("anime_" + ratings.anime_id.astype(str)).isin(seen)]
    hist = hist.sort_values("rating", ascending=False).head(10)
    test = {cands[i] for i in ev.test[u]}

    out.append(f"## One user: `{u}` ({len(seen)} train interactions, {len(test)} test anime)\n")
    out.append("**Their 10 highest-rated (train):** " + "; ".join(
        f"{title.get(f'anime_{a}', a)} ({r})" for a, r in zip(hist.anime_id, hist.rating)) + "\n")
    out.append("| # | DistMult top-10 | TransE top-10 |\n|---|---|---|")
    cols = []
    for name, (m, e2i, r_rated) in models.items():
        with torch.no_grad():
            s = m.score_all(torch.tensor([e2i[u]]), r_rated,
                            torch.tensor([e2i[a] for a in cands]))[0]
        s[list(ev.seen[u]) + list(ev.valid.get(u, ()))] = -float("inf")
        top = [cands[i] for i in s.argsort(descending=True)[:10].tolist()]
        cols.append([("✅ " if a in test else "") + title.get(a, a) for a in top])
    out += [f"| {i + 1} | {d} | {t} |" for i, (d, t) in enumerate(zip(*cols))]
    out.append("\n✅ = in this user's held-out test set.\n")

    # ---- 3. aggregate over all test users ----
    train = load_split()["train"]
    pop = train[train.r.isin(INTERACTIONS)].t.value_counts()
    poprank = {a: i for i, a in enumerate(pop.index)}  # 0 = most interacted
    hentai = set(train[(train.r == "hasGenre") & (train.t == "Hentai")].h)
    logpop = np.log1p(np.array([pop.get(a, 0) for a in cands]))
    users = sorted(ev.test)
    hu = [i for i, x in enumerate(users) if any(cands[j] in hentai for j in ev.seen[x])]
    out.append(f"## All {len(users):,} test users\n")
    out.append("| | DistMult | TransE |\n|---|---|---|")
    rows = {k: [] for k in ("pop", "top100", "distinct", "h5", "hshare", "corr")}
    for name, (m, e2i, r_rated) in models.items():
        with torch.no_grad():
            C = torch.tensor([e2i[a] for a in cands])
            S = m.score_all(torch.tensor([e2i[x] for x in users]), r_rated, C)
            norm = m.ent(C).norm(dim=1).numpy()
        for i, x in enumerate(users):
            S[i, list(ev.seen[x]) + list(ev.valid.get(x, ()))] = -float("inf")
        top = S.topk(10, dim=1).indices.numpy()
        ranks = np.array([[poprank[cands[j]] for j in row] for row in top])
        h = np.array([[cands[j] in hentai for j in row] for row in top])
        rows["pop"].append(f"{np.median(ranks):.0f}")
        rows["top100"].append(f"{np.mean(ranks < 100):.0%}")
        rows["distinct"].append(f"{len(set(top.ravel())):,}")
        rows["h5"].append(f"{np.mean(h.sum(1) >= 5):.1%}")
        rows["hshare"].append(f"{h[hu].mean():.0%}")
        rows["corr"].append(f"{np.corrcoef(logpop, norm)[0, 1]:+.2f}")
    labels = {"pop": "median popularity rank of recs (0 = most popular)",
              "top100": "share of recs from the 100 most popular",
              "distinct": "distinct anime recommended to anyone",
              "h5": "users with >=5 Hentai-genre recs",
              "hshare": f"Hentai share of top-10, the {len(hu)} users with any in history",
              "corr": "corr(log popularity, item vector norm)"}
    out += [f"| {labels[k]} | {v[0]} | {v[1]} |" for k, v in rows.items()]

    text = "\n".join(out)
    (RESULTS / "probe.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
