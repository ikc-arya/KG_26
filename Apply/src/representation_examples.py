"""Apply/embeddings — five concrete examples of how nodes/edges are represented.

Read straight from the trained checkpoints (portfolio section 3.1):
  1. an anime node (DistMult)              vector excerpt, norm, nearest anime
  2. a user node (DistMult)                vector excerpt, norm, top recommendations
  3. the `rated` relation (DistMult)       a diagonal scaling -> symmetric by construction
  4. the same user in TransE               a point u + r_rated; distances to liked vs random anime
  5. a never-rated anime (LightGCN+content) embedded only through genre/studio neighbours

Run after train_kge.py (both models) and train_gnn.py --content:
    python src/representation_examples.py      -> data/generated/results/representation.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from probe_embeddings import load_model
from recsys_eval import APPLY, INTERACTIONS, RESULTS, Evaluator, load_split
from train_gnn import load_final_embeddings

USER, ANIME = "user_66835", "anime_1"  # the probe's user; Cowboy Bebop


def vec(v: torch.Tensor, n: int = 6) -> str:
    return "[" + ", ".join(f"{x:+.2f}" for x in v[:n].tolist()) + ", …]"


@torch.no_grad()
def main() -> None:
    torch.manual_seed(26)
    titles = pd.read_csv(APPLY / "data" / "anime.csv", usecols=["mal_id", "title"])
    title = dict(zip("anime_" + titles.mal_id.astype(str), titles.title))
    split = load_split()
    ev = Evaluator(split)
    cands = ev.candidates
    out = ["# Representation examples\n"]

    dm, e2i, r_rated = load_model("distmult")
    C = dm.ent(torch.tensor([e2i[a] for a in cands]))

    # 1. anime node
    a = dm.ent(torch.tensor(e2i[ANIME]))
    sim = F.cosine_similarity(C, a[None], dim=1)
    nn = [cands[i] for i in sim.argsort(descending=True).tolist() if cands[i] != ANIME][:3]
    out.append(f"**1. Anime node — {title[ANIME]} (DistMult, 128-d)**: {vec(a)} ‖·‖ = {a.norm():.2f}; "
               f"nearest by cosine: {', '.join(title[x] for x in nn)}\n")

    # 2. user node
    u = dm.ent(torch.tensor(e2i[USER]))
    s = dm.score_all(torch.tensor([e2i[USER]]), r_rated, torch.tensor([e2i[x] for x in cands]))[0]
    s[list(ev.seen[USER]) + list(ev.valid.get(USER, ()))] = -float("inf")
    top = [cands[i] for i in s.topk(3).indices.tolist()]
    out.append(f"**2. User node — {USER} (DistMult)**: {vec(u)} ‖·‖ = {u.norm():.2f}; a *direction*: "
               f"top-3 unseen anime along it: {', '.join(title[x] for x in top)}\n")

    # 3. relation `rated`
    r = dm.rel.weight[r_rated]
    fwd = dm.score(torch.tensor(e2i[USER]), torch.tensor(r_rated), torch.tensor(e2i[ANIME]))
    bwd = dm.score(torch.tensor(e2i[ANIME]), torch.tensor(r_rated), torch.tensor(e2i[USER]))
    out.append(f"**3. Relation `rated` (DistMult)**: diagonal {vec(r)} → score = Σ uᵢ·rᵢ·aᵢ. "
               f"score({USER}, rated, {title[ANIME]}) = {fwd:.3f}, reversed = {bwd:.3f} → symmetric by construction\n")

    # 4. same user in TransE
    te, e2i_t, r_t = load_model("transe")
    p = te.ent(torch.tensor(e2i_t[USER])) + te.rel.weight[r_t]
    liked = [cands[i] for i in ev.seen[USER]]
    others = [x for x in cands if x not in set(liked)]
    d_like = (te.ent(torch.tensor([e2i_t[x] for x in liked])) - p).norm(dim=1)
    rnd = np.random.default_rng(26).choice(others, len(liked), replace=False)
    d_rand = (te.ent(torch.tensor([e2i_t[x] for x in rnd])) - p).norm(dim=1)
    out.append(f"**4. Same user in TransE**: a *point* u + r_rated = {vec(p)}; distance to their "
               f"{len(liked)} train anime {d_like.mean():.2f} ± {d_like.std():.2f} vs random anime "
               f"{d_rand.mean():.2f} (γ = {te.gamma}) → closer, but no anime sits *on* the point\n")

    # 5. never-rated anime in LightGCN + content
    E, n2i = load_final_embeddings("lightgcn_k3_content", split)
    nodes = list(n2i)
    train = split["train"]
    genres = train[train.r == "hasGenre"].groupby("h").t.apply(sorted).to_dict()
    inter = set(train[train.r.isin(INTERACTIONS)].t)
    cold = sorted(x for x in nodes if x.startswith("anime_") and x not in inter
                  and len(genres.get(x, [])) >= 3 and x in title)
    c = cold[np.random.default_rng(26).integers(len(cold))]
    W = F.normalize(E[[n2i[x] for x in cands]], dim=1)
    nn = [cands[i] for i in (W @ F.normalize(E[n2i[c]], dim=0)).topk(3).indices.tolist()]
    out.append(f"**5. Never-rated anime — {title[c]} {genres[c]} (LightGCN + content)**: no rating edge, "
               f"embedded only via its genre/studio neighbours; nearest rated anime: "
               + "; ".join(f"{title[x]} {genres.get(x, [])}" for x in nn) + "\n")

    text = "\n".join(out)
    (RESULTS / "representation.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
