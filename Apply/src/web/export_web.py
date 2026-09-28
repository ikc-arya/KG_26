"""Aninext web UI — export the recommenders to static files for GitHub Pages.

GitHub Pages serves static files only: no Python, no torch. The heavy parts run here once;
the cheap re-rank (score, z-score, +boost per preferred tag, avoid mask) runs in the
browser, the same formula as `recommend.rerank`.
  - ML side:    per model, the anime vectors + each user's query vector -> data/<model>/*.bin
                (the page loads a model only when it is picked)
  - logic side: SPARQL over the KG -> each anime's genre/theme/demographic tags -> anime.json
  - history:    every anime a user already rated/watched (train + valid + test) -> known.bin

Every model is reduced to one of three scoring kinds, so the page needs no model code:
  dot  score = q . a          LightGCN (q = user vector), DistMult (q = user * r_rated)
  l2   score = -||q - a||     TransE (q = user + r_rated; the model's gamma is a constant shift)
  pop  score = train count    popularity baseline (same list for everyone)
A new user's q is the mean of the liked anime's vectors: LightGCN's own propagation rule,
and for DistMult / TransE the point that scores those anime highest ("more like these").

Writes <repo>/docs/ (index.html copied from this folder + data/), then self-checks: a NumPy
mirror of the JS on the exported files must give the same top-10 as the Python models, and
for LightGCN k3 the same as recommend.py on both README examples.

Run:  python Apply/src/web/export_web.py              # data + page
      python Apply/src/web/export_web.py --html-only  # just re-copy index.html
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB.parent))  # import the pipeline modules from Apply/src/

import torch  # noqa: E402

from evolve_kg import PREFIX  # noqa: E402
from probe_embeddings import load_model  # noqa: E402
from recommend import TAGS, load_content, matches, rerank, tag_iris  # noqa: E402
from recsys_eval import INTERACTIONS, Evaluator, load_split  # noqa: E402
from train_gnn import load_final_embeddings  # noqa: E402

DOCS = WEB.parents[2] / "docs"
DATA = DOCS / "data"
DEFAULT = "lightgcn_k3"
IMG_CDN = "https://cdn.myanimelist.net/images/anime/"  # = CDN in index.html

# key (= checkpoint / results name), label, one-line description shown in the UI
MODELS = [
    ("popularity", "Popularity", "Most-watched first, the same for everyone"),
    ("lightgcn_k0", "MF", "Matrix factorisation: one vector per user and per anime"),
    ("lightgcn_k3", "LightGCN", "Graph neural network over who watched what"),
    ("lightgcn_k3_content", "LightGCN + KG", "GNN that also sees the KG's genres, studios and other links"),
    ("distmult", "DistMult", "KG embedding: user × rated × anime"),
    ("transe", "TransE", "KG embedding: user + rated lands near the anime"),
]

# the two README examples: (user, liked, prefer, avoid, boost)
EXAMPLES = [("user_66835", None, ["Romance", "Comedy"], ["Hentai"], 1.0),
            (None, ["Cowboy Bebop", "Samurai Champloo", "Trigun"], ["SciFi"], [], 0.5)]


def image_paths() -> dict[str, str]:
    """MAL id -> cover image path, e.g. '4/19644' for .../images/anime/4/19644.jpg.

    Display only: the image URL comes straight from anime.csv, it is not in the KG (the
    anime IRIs do not resolve). The page adds the CDN prefix and .jpg back, which keeps
    anime.json small. MAL's placeholder icon (no cover) is left out."""
    out: dict[str, str] = {}
    with (WEB.parents[1] / "data" / "anime.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):  # first row per id wins, as in build_triples.load_anime
            url = r["image_jpg_url"]
            if r["mal_id"] not in out and url.startswith(IMG_CDN) and url.endswith(".jpg"):
                out[r["mal_id"]] = url[len(IMG_CDN):-len(".jpg")]
    return out


def copy_page() -> None:
    DOCS.mkdir(exist_ok=True)
    for f in [WEB / "index.html", *WEB.glob("bg.*"), *WEB.glob("*.svg")]:  # page + background + logo
        shutil.copy(f, DOCS / f.name)
    (DOCS / ".nojekyll").touch()  # serve files as-is, no Jekyll build


def model_vectors(key: str, split, ev, cands: list[str], users: list[str]) -> dict:
    """-> {kind, A (anime x d), Q (users x d)} or {kind: pop, s}, all float32 numpy,
    plus `ref(user)`: the model's own scores for one user (for the self-check)."""
    if key == "popularity":  # as in recsys_eval.main
        train = split["train"]
        pop = train[train.r.isin(INTERACTIONS)].t.value_counts()
        s = np.array([pop.get(c, 0) for c in cands], dtype=np.float32)
        return {"kind": "pop", "s": s, "ref": lambda u: s}
    if key.startswith("lightgcn"):
        E, n2i = load_final_embeddings(key, split)
        E = E.detach()
        Ac = E[[n2i[a] for a in cands]]
        return {"kind": "dot", "A": Ac.numpy(), "Q": E[[n2i[u] for u in users]].numpy(),
                "ref": lambda u: (Ac @ E[n2i[u]]).numpy()}
    m, e2i, r = load_model(key)
    with torch.no_grad():
        ent, rel = m.ent.weight, m.rel.weight[r]
        A = ent[[e2i[a] for a in cands]]
        U = ent[[e2i[u] for u in users]]
        cand_t = torch.tensor([e2i[a] for a in cands])
        ref = lambda u: m.score_all(torch.tensor([e2i[u]]), r, cand_t)[0].detach().numpy()  # noqa: E731
        if key == "transe":
            return {"kind": "l2", "A": A.numpy(), "Q": (U + rel).numpy(), "ref": ref}
        return {"kind": "dot", "A": A.numpy(), "Q": (U * rel).numpy(), "ref": ref}


def js_scores(v: dict, q: np.ndarray | None) -> np.ndarray:
    """NumPy mirror of the page's scoring (float64 over the exported float32 arrays)."""
    if v["kind"] == "pop":
        return v["s"].astype(np.float64)
    A = v["A"].astype(np.float64)
    return A @ q if v["kind"] == "dot" else -np.sqrt(((A - q) ** 2).sum(1))


def top(x: np.ndarray, k: int = 10) -> list[int]:
    return list(np.argsort(-x, kind="stable")[:k])


def export() -> None:
    split = load_split()
    ev = Evaluator(split)
    g = load_content()
    _, n2i = load_final_embeddings(DEFAULT, split)

    cands = [a for a in ev.candidates if a in n2i]
    c2i = {a: i for i, a in enumerate(cands)}
    assert cands == ev.candidates, "candidate order must match the evaluator's (known.bin uses it)"
    assert len(cands) < 2 ** 16, "known.bin stores anime indices as uint16"
    title = {str(r.a).split("#")[1]: str(r.t) for r in g.query(PREFIX + "SELECT ?a ?t WHERE { ?a :title ?t }")}

    # ---- logic side: tag vocabulary + each candidate's tags (same pattern as recommend.matches)
    vocab: list[dict] = []
    t2i: dict[str, int] = {}
    for r in sorted(g.query(PREFIX + """
            SELECT ?t ?l ?C WHERE { ?t rdfs:label ?l . ?t a ?C . VALUES ?C { :Genre :Theme :Demographic } }"""),
            key=lambda r: str(r.l).lower()):
        if str(r.t) not in t2i:
            t2i[str(r.t)] = len(vocab)
            vocab.append({"label": str(r.l), "class": str(r.C).split("#")[1]})
    tags: list[set[int]] = [set() for _ in cands]
    for r in g.query(PREFIX + f"SELECT ?a ?t WHERE {{ ?a {TAGS} ?t }}"):
        a = str(r.a).split("#")[1]
        if a in c2i and str(r.t) in t2i:
            tags[c2i[a]].add(t2i[str(r.t)])

    # ---- history: everything the KG knows each user touched (the CLI's `known`)
    users = sorted(x for x in n2i if x.startswith("user_"))
    known: dict[str, set[int]] = {u: set(ev.seen.get(u, ())) for u in users}
    for part in ("valid", "test"):
        for u, a in zip(split[part].h, split[part].t):
            if u in known and a in c2i:
                known[u].add(c2i[a])
    offsets = np.cumsum([0] + [len(known[u]) for u in users]).tolist()

    # ---- write shared files
    if DATA.exists():
        shutil.rmtree(DATA)  # pure build output of this script
    DATA.mkdir(parents=True)
    np.concatenate([sorted(known[u]) for u in users]).astype("<u2").tofile(DATA / "known.bin")
    img = image_paths()
    (DATA / "anime.json").write_text(json.dumps({
        "tags": vocab,
        "anime": [{"id": (mid := a.removeprefix("anime_")), "title": title.get(a, a), "img": img.get(mid, ""),
                   "tags": sorted(tags[i])} for i, a in enumerate(cands)],
    }, ensure_ascii=False, separators=(",", ":")))
    (DATA / "users.json").write_text(json.dumps({"ids": users, "offsets": offsets}, separators=(",", ":")))

    # ---- per model: vectors + self-check against the model's own scores
    kmask = {}
    for u in users[:1] + ["user_66835"]:
        kmask[u] = np.zeros(len(cands), bool)
        kmask[u][sorted(known[u])] = True
    no, off = np.zeros(len(cands)), np.zeros(len(cands), bool)  # no preferences, nothing avoided
    meta_models, vecs = [], {}
    for key, label, desc in MODELS:
        v = vecs[key] = model_vectors(key, split, ev, cands, users)
        out = DATA / key
        out.mkdir()
        if v["kind"] == "pop":
            v["s"].astype("<f4").tofile(out / "pop.bin")
            dim = 0
        else:
            v["A"].astype("<f4").tofile(out / "anime.bin")
            v["Q"].astype("<f4").tofile(out / "user.bin")
            dim = int(v["A"].shape[1])
        meta_models.append({"key": key, "label": label, "desc": desc, "kind": v["kind"], "dim": dim})
        for u, m in kmask.items():
            q = None if v["kind"] == "pop" else v["Q"][users.index(u)].astype(np.float64)
            a = top(rerank(v["ref"](u), m, no, off, 1.0)[0])
            b = top(rerank(js_scores(v, q), m, no, off, 1.0)[0])
            assert a == b, f"{key} {u}: {[cands[i] for i in a]} != {[cands[i] for i in b]}"
        print(f"self-check ok: {key} ({v['kind']}, dim {dim}) top-10 == model scores")

    (DATA / "meta.json").write_text(json.dumps({
        "default": DEFAULT, "models": meta_models, "n_anime": len(cands), "n_users": len(users),
    }, indent=1))
    size = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file()) / 1e6
    print(f"exported {len(cands):,} anime, {len(users):,} users, {len(vocab)} tags, "
          f"{len(MODELS)} models -> {DATA} ({size:.1f} MB)")

    # ---- self-check: default model through the page's full path == recommend.py (torch + SPARQL)
    E, n2i = load_final_embeddings(DEFAULT, split)
    E = E.detach()
    v = vecs[DEFAULT]
    by_title = {t.lower(): a for a, t in title.items()}
    for user, liked, prefer, avoid, boost in EXAMPLES:
        # recommend.py path
        if user:
            kn = {cands[i] for i in known[user]}
            u = E[n2i[user]]
        else:
            kn = {by_title[t.lower()] for t in liked}
            u = E[[n2i[a] for a in kn]].mean(0)
        with torch.no_grad():
            s = (E[[n2i[a] for a in cands]] @ u).numpy()
        pm, am = matches(g, tag_iris(g, prefer)), matches(g, tag_iris(g, avoid))
        z, tuned = rerank(s, np.array([a in kn for a in cands]), np.array([pm.get(a, 0) for a in cands]),
                          np.array([a in am for a in cands], dtype=bool), boost)
        # page path: exported arrays + tag index only
        m = np.zeros(len(cands), bool)
        if user:
            m[sorted(known[user])] = True
            q = v["Q"][users.index(user)].astype(np.float64)
        else:
            idx = [c2i[by_title[t.lower()]] for t in liked]
            m[idx] = True
            q = v["A"][idx].astype(np.float64).mean(0)
        pset = {t2i[t] for t in tag_iris(g, prefer)}  # the UI hands over these indices directly
        aset = {t2i[t] for t in tag_iris(g, avoid)}
        pref = np.array([len(tags[i] & pset) for i in range(len(cands))])
        amask = np.array([bool(tags[i] & aset) for i in range(len(cands))])
        z2, tuned2 = rerank(js_scores(v, q), m, pref, amask, boost)
        for name, x, y in (("ML only", z, z2), ("tuned", tuned, tuned2)):
            assert top(x) == top(y), f"{user or liked} {name}: {top(x)} != {top(y)}"
        print(f"self-check ok: {user or ', '.join(liked)} (top-10 ML only + tuned match recommend.py)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--html-only", action="store_true", help="only re-copy index.html into docs/")
    args = p.parse_args()
    copy_page()
    if not args.html_only:
        export()
    print(f"page -> {DOCS / 'index.html'}   preview: python Apply/src/web/serve.py")


if __name__ == "__main__":
    main()
