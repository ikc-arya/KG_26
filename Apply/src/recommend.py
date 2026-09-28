"""Apply/service — recommendations tuned by the user's stated preferences.

The proposal's problem: "recommendations around my taste, with tuned preferences along
with watch history". Two halves, one from each side of the KG:
  - watch history -> ML: LightGCN (k=3) scores every anime for the user
  - preferences   -> logic: SPARQL over the KG finds which anime carry the preferred /
                     avoided genres, themes or demographics; scores are re-ranked with that
Re-rank: z-score the ML scores over all candidates, add `--boost` per matched preference,
drop anything with an avoided tag. Known anime (already rated/watched) are never shown.

A brand-new user ("my taste") needs no retraining: `--liked "A,B,C"` builds the user vector
as the mean of those anime's embeddings -- LightGCN's own propagation rule (a user's
layer-1 vector is the normalised sum of the anime it touches).

Run:  python src/recommend.py --user user_66835 --prefer Romance,Comedy --avoid Hentai
      python src/recommend.py --liked "Cowboy Bebop,Samurai Champloo,Trigun" --prefer SciFi
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from rdflib import Graph

from build_triples import slug
from evolve_kg import GEN, PREFIX
from recsys_eval import Evaluator, load_split
from train_gnn import load_final_embeddings

TAGS = "(:hasGenre|:hasTheme|:hasDemographic)"


def load_content() -> Graph:
    lines = (GEN / "abox.ttl").read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("#### Reified ratings"))
    g = Graph()
    g.parse(data="\n".join(lines[:start]), format="turtle")
    return g


def tag_iris(g: Graph, names: list[str]) -> list[str]:
    known = {str(r.l): str(r.t) for r in g.query(PREFIX + """
        SELECT ?t ?l WHERE { ?t rdfs:label ?l . ?t a ?C . VALUES ?C { :Genre :Theme :Demographic } }""")}
    by_slug = {slug(l): t for l, t in known.items()}
    out = []
    for n in names:
        if slug(n) not in by_slug:
            raise SystemExit(f"unknown tag {n!r}; e.g. {sorted(known)[:15]} ...")
        out.append(by_slug[slug(n)])
    return out


def matches(g: Graph, tags: list[str]) -> dict[str, int]:
    """anime -> number of the given tags it carries (logic side)."""
    if not tags:
        return {}
    values = " ".join(f"<{t}>" for t in tags)
    rows = g.query(PREFIX + f"""
        SELECT ?a (COUNT(DISTINCT ?t) AS ?n) WHERE {{ ?a {TAGS} ?t . VALUES ?t {{ {values} }} }} GROUP BY ?a""")
    return {str(r.a).split("#")[1]: int(r.n) for r in rows}


def rerank(s: np.ndarray, known: np.ndarray, pref: np.ndarray, avoid: np.ndarray,
           boost: float) -> tuple[np.ndarray, np.ndarray]:
    """ML scores -> (z, tuned). `known`/`avoid` are bool masks over candidates, `pref` the
    number of preferred tags each candidate carries. Known anime and avoided tags -> -inf.
    The web UI (src/web/) runs this same formula in JS; export_web.py checks it against this."""
    s = s.copy()
    s[known] = -np.inf
    ok = np.isfinite(s)
    z = np.full_like(s, -np.inf)
    z[ok] = (s[ok] - s[ok].mean()) / s[ok].std()
    tuned = z + boost * pref
    tuned[avoid] = -np.inf
    return z, tuned


def main() -> None:
    p = argparse.ArgumentParser()
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--user", help="existing watcher, e.g. user_66835")
    who.add_argument("--liked", help='new user: comma-separated titles, e.g. "Cowboy Bebop,Trigun"')
    p.add_argument("--prefer", default="", help="genres/themes/demographics to boost, comma-separated")
    p.add_argument("--avoid", default="", help="genres/themes/demographics to exclude")
    p.add_argument("--boost", type=float, default=1.0, help="z-score added per matched preference")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    split = load_split()
    ev = Evaluator(split)
    E, n2i = load_final_embeddings("lightgcn_k3", split)
    g = load_content()
    title = {str(r.a).split("#")[1]: str(r.t) for r in g.query(PREFIX + "SELECT ?a ?t WHERE { ?a :title ?t }")}
    cands = [a for a in ev.candidates if a in n2i]

    # ---- the user vector: watch history (existing) or a few liked titles (new)
    if args.user:
        # everything the KG already knows about this user: train + valid + test ratings
        known = {ev.candidates[i] for i in ev.seen.get(args.user, ())}
        for part in ("valid", "test"):
            known |= set(split[part][split[part].h == args.user].t)
        u = E[n2i[args.user]]
        who_txt = f"{args.user} ({len(known)} known anime)"
    else:
        by_title = {t.lower(): a for a, t in title.items()}
        wanted = [t.strip() for t in args.liked.split(",") if t.strip()]
        missing = [t for t in wanted if t.lower() not in by_title or by_title[t.lower()] not in n2i]
        if missing:
            raise SystemExit(f"not found among rated anime: {missing}")
        known = {by_title[t.lower()] for t in wanted}
        u = E[[n2i[a] for a in known]].mean(0)
        who_txt = f"new user who liked: {', '.join(wanted)}"

    with torch.no_grad():
        s = (E[[n2i[a] for a in cands]] @ u).numpy()

    # ---- the preferences: logic side
    prefer = tag_iris(g, [x for x in args.prefer.split(",") if x.strip()])
    avoid = tag_iris(g, [x for x in args.avoid.split(",") if x.strip()])
    pm, am = matches(g, prefer), matches(g, avoid)
    z, tuned = rerank(s, np.array([a in known for a in cands]),
                      np.array([pm.get(a, 0) for a in cands]),
                      np.array([a in am for a in cands], dtype=bool), args.boost)

    tags: dict[str, list[str]] = {}
    for r in g.query(PREFIX + f"SELECT ?a ?l WHERE {{ ?a {TAGS} ?t . ?t rdfs:label ?l }}"):
        tags.setdefault(str(r.a).split("#")[1], []).append(str(r.l))

    base_top = list(np.argsort(-z)[:args.k])
    tuned_top = list(np.argsort(-tuned)[:args.k])
    base_rank = {j: r for r, j in enumerate(np.argsort(-z))}

    def share(top, m):
        return sum(cands[j] in m for j in top) / len(top)

    print(f"\n# Recommendations for {who_txt}")
    print(f"prefer {args.prefer or '-'} | avoid {args.avoid or '-'} | boost {args.boost}\n")
    print("| # | ML only (watch history) | tuned (history + preferences) | ML rank | tags |\n|---|---|---|---|---|")
    for r in range(args.k):
        b, t = cands[base_top[r]], cands[tuned_top[r]]
        print(f"| {r + 1} | {title.get(b, b)} | {title.get(t, t)} | {base_rank[tuned_top[r]] + 1} | {', '.join(sorted(tags.get(t, [])))} |")
    if prefer:
        print(f"\npreferred tags in top-{args.k}: {share(base_top, pm):.0%} ML only -> {share(tuned_top, pm):.0%} tuned")
    if avoid:
        print(f"avoided tags in top-{args.k}:   {share(base_top, am):.0%} ML only -> {share(tuned_top, am):.0%} tuned")


if __name__ == "__main__":
    main()
