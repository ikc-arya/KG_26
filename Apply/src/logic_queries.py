"""Apply/logic — five rules / queries over the real ABox (portfolio section 4.1, LO2).

Loads ontology.ttl + a slice of abox.ttl: ALL anime content + vocabulary nodes, and the
reified ratings of a seeded sample of users (full 2.5M-triple graph is too slow for
rdflib SPARQL; content is global, so the content-side answers are exact).

  1. OWL-RL typing       (creates edges)   domain/range axioms type nodes on a small slice
  2. `likes` rule        (creates edges)   CONSTRUCT  user likes anime  <-  rating >= 8
  3. recursive path      (recursion)       organisations reachable from Sunrise via co-work chains
  4. explanation query   (logic + ML)      why DistMult's top pick fits user_66835: shared nodes
                                           with anime the user likes (uses rule 2's edges)
  5. integrity checks    (correction)      constraints the generated KG must satisfy

Run:  python src/logic_queries.py            -> data/generated/results/logic.md
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import owlrl
from rdflib import RDF, RDFS, Graph, Namespace

from build_triples import CORP_SUFFIX
from recsys_eval import RESULTS

SRC = Path(__file__).resolve().parent
ABOX = SRC.parent / "data" / "generated" / "abox.ttl"
A = Namespace("http://kg26.example/anime#")
PREFIX = "PREFIX : <http://kg26.example/anime#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
USER = "user_66835"
N_USERS = 200


def load_slice(n_users: int, seed: int = 26) -> Graph:
    """Content + vocab sections whole; rating lines only for sampled users (+ USER)."""
    lines = ABOX.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("#### Reified ratings"))
    users = sorted({m.group(1) for l in lines[start:] if (m := re.match(r":(user_\d+) ", l))})
    keep = set(np.random.default_rng(seed).choice(users, n_users, replace=False)) | {USER}
    body = lines[:start] + [l for l in lines[start:] if (m := re.match(r":(user_\d+) ", l)) and m.group(1) in keep]
    g = Graph()
    g.parse(SRC / "ontology.ttl", format="turtle")
    g.parse(data="\n".join(body), format="turtle")
    return g


def q(g: Graph, body: str):
    return list(g.query(PREFIX + body))


def name(x) -> str:
    return str(x).split("#")[-1]


def main() -> None:
    out = ["# Logic-based representation — rules and queries on the real ABox\n"]
    t0 = time.time()
    g = load_slice(N_USERS)
    out.append(f"Slice: {len(g):,} triples (all content + ratings of {N_USERS + 1} users), loaded in {time.time() - t0:.0f}s.\n")

    # 1. OWL-RL typing on a small real slice (full closure is infeasible in rdflib)
    small = Graph()
    small.parse(SRC / "ontology.ttl", format="turtle")
    for s, p, o in g.triples((A.anime_1, None, None)):
        small.add((s, p, o))
    for r in q(g, f"SELECT ?r WHERE {{ ?r :ratingOf :anime_1 }} LIMIT 20"):
        for t in g.triples((r[0], None, None)):
            small.add(t)
        for t in g.triples((None, A.rated, r[0])):
            small.add(t)
    before = {x for x in small.subjects(RDF.type, None)}
    n0 = len(small)
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(small)
    typed = lambda cls: sorted({name(s) for s in small.subjects(RDF.type, A[cls])})
    out.append("## 1. Domain/range typing (OWL-RL, creates `rdf:type` edges)\n")
    out.append(f"Slice = ontology (incl. its 3-anime sample ABox) + Cowboy Bebop + 20 of its real ratings: {n0} asserted → {len(small):,} triples after closure. "
               f"Nothing in the data says who is a Watcher or what is an Organization; the axioms derive it:\n")
    out.append(f"- Watcher (domain of `rated`): {len(typed('Watcher'))} users, e.g. {typed('Watcher')[:3]}")
    out.append(f"- Rating (domain of `ratingOf`): {len(typed('Rating'))} nodes")
    out.append(f"- Organization (range of `animatedBy`/`producedBy`/`licensedBy`): {typed('Organization')}")
    out.append(f"- Work (via `Anime ⊑ Work`): {typed('Work')}\n")

    # 2. CONSTRUCT rule: likes
    likes = g.query(PREFIX + """
        CONSTRUCT { ?u :likes ?a }
        WHERE { ?u :rated ?r . ?r :ratingOf ?a ; :ratingValue ?v . FILTER(?v >= 8) }""")
    n_like = 0
    for t in likes:
        g.add(t)
        n_like += 1
    n_rated = len(q(g, "SELECT ?r WHERE { ?r :ratingOf ?a }"))
    out.append("## 2. Rule: `likes` (CONSTRUCT, creates edges)\n")
    out.append("```sparql\nCONSTRUCT { ?u :likes ?a }\nWHERE { ?u :rated ?r . ?r :ratingOf ?a ; :ratingValue ?v . FILTER(?v >= 8) }\n```")
    out.append(f"→ {n_like:,} new `likes` edges from {n_rated:,} reified ratings; `-1` (watched, unscored) never qualifies. "
               f"A direct user→anime edge again: the same collapse the flat ML file does, here as an explicit, inspectable rule.\n")

    # 3. recursive: organisations reachable from Sunrise via chains of shared anime
    role = "(:animatedBy|:producedBy|:licensedBy)"
    t0 = time.time()
    reach = q(g, f"SELECT DISTINCT ?o WHERE {{ :Sunrise (^{role}/{role})+ ?o }}")
    direct = q(g, f"SELECT DISTINCT ?o WHERE {{ :Sunrise ^{role}/{role} ?o }}")
    n_org = len(set(g.subjects(RDF.type, A.Organization)))
    out.append("## 3. Recursive exploration: the co-work network from Sunrise\n")
    out.append(f"```sparql\nSELECT DISTINCT ?o WHERE {{ :Sunrise (^{role}/{role})+ ?o }}\n```")
    out.append(f"→ direct collaborators (1 hop): {len(direct):,}; reachable through arbitrarily long chains: "
               f"{len(reach):,} of {n_org:,} organisations ({len(reach) / n_org:.0%}), in {time.time() - t0:.1f}s. "
               f"The industry is one big connected component — which is exactly why shared organisation nodes give embeddings/GNNs paths to follow.\n")

    # 4. explanation: why DistMult's top pick for USER fits them
    pick = q(g, 'SELECT ?a WHERE { ?a :title "One Piece Film: Strong World" }')[0][0]
    rows = q(g, f"""
        SELECT ?via (COUNT(DISTINCT ?liked) AS ?n) WHERE {{
          :{USER} :likes ?liked . FILTER(?liked != <{pick}>)
          ?liked ?p ?via . <{pick}> ?p ?via .
          FILTER(?p IN (:hasGenre, :hasTheme, :animatedBy, :producedBy, :hasSource, :hasDemographic))
        }} GROUP BY ?via ORDER BY DESC(?n)""")
    n_liked = len(q(g, f"SELECT ?a WHERE {{ :{USER} :likes ?a }}"))
    out.append(f"## 4. Explaining an ML prediction: why *One Piece Film: Strong World* for {USER}\n")
    out.append(f"DistMult's top recommendation (a true positive, in the user's test set). Query: nodes shared between it and the {n_liked} anime the user `likes` (rule 2):\n")
    out += [f"- `{name(r[0])}` — shared with {r[1]} liked anime" for r in rows[:6]]
    out.append("\nThe embedding gives a score; the graph gives the reasons.\n")

    # 5. integrity constraints
    checks = {
        "Rating without exactly one `ratingOf`":
            "SELECT ?r WHERE { ?u :rated ?r . OPTIONAL { ?r :ratingOf ?a } } GROUP BY ?r HAVING (COUNT(?a) != 1)",
        "Rating pointing at an anime with no content (orphan)":
            "SELECT ?r WHERE { ?r :ratingOf ?a . FILTER NOT EXISTS { ?a :title ?t } }",
        "Rating without a value":
            "SELECT ?r WHERE { ?u :rated ?r . FILTER NOT EXISTS { ?r :ratingValue ?v } }",
    }
    out.append("## 5. Integrity constraints (correction)\n")
    out.append("| constraint violated | count |\n|---|---|")
    for label, body in checks.items():
        out.append(f"| {label} | {len(q(g, body))} |")
    junk = [name(o) for o in set(g.subjects(RDF.type, A.Organization))
            if CORP_SUFFIX.match(str(g.value(o, RDFS.label)))]
    out.append(f"| Organization that is only a corporate suffix (junk hub, e.g. `Inc.`) | {len(junk)} |")
    # anime are never asserted `a :Anime` in the ABox -- that type only exists after reasoning
    # (domain axioms). Written with `?a a :Anime`, this check silently returns 0 on raw data.
    vacuous = q(g, "SELECT ?a WHERE { ?a a :Anime . FILTER NOT EXISTS { ?a :hasGenre ?x } }")
    no_genre = q(g, "SELECT ?a WHERE { ?a :title ?t . FILTER NOT EXISTS { ?a :hasGenre ?x } }")
    out.append(f"\nInformational: {len(no_genre):,} of {len(q(g, 'SELECT ?a WHERE { ?a :title ?t }')):,} anime have no genre → no genre path for a GNN if never rated.")
    out.append(f"Caught while writing it: the same check phrased as `?a a :Anime` returns {len(vacuous)}, because anime are typed only *by entailment* "
               f"(domain of `hasGenre`/`title`…), never asserted. A query over un-materialised data silently misses what reasoning would add → run constraints after closure, or phrase them on asserted predicates.\n")

    text = "\n".join(out)
    (RESULTS / "logic.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
