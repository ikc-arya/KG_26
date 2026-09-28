"""Apply/evolution — write KG changes to disk: logic completion, ML completion, update.

Three steps, each an actual change to the KG, kept in SEPARATE files so certain and
uncertain knowledge never mix (data/generated/evolution/):

  1. inferred.ttl   LOGIC, certain. SPARQL rules over the FULL abox.ttl:
       - RDFS domain/range typing   ?p rdfs:domain ?C . ?s ?p ?o  =>  ?s a ?C   (+ range)
       - subclass                   ?s a ?C . ?C rdfs:subClassOf ?D  =>  ?s a ?D
       - likes                      rating >= 8  =>  ?u :likes ?a
     Evaluated partition by partition (content, then 500 users at a time). That is exact,
     not an approximation: each rule reads either one data triple (+ TBox) or the three
     triples of one rating node, which always sit in the same user partition -- so the
     union of the per-partition results equals the result on the whole graph.

  2. predicted.ttl  ML, scored. LightGCN (k=3) top-10 unseen anime per user, each as a
     reified :Prediction (watcher, anime, score, rank, model, status) -- n-ary, like a Rating.

  3. update         the held-out test ratings (never seen by the model) are replayed as new
     facts arriving; predictions they match change status :open -> :confirmed.

Run:  python src/evolve_kg.py      -> data/generated/evolution/ + results/evolution.md
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from rdflib import Graph

from recsys_eval import RESULTS, Evaluator, load_split
from train_gnn import load_final_embeddings

SRC = Path(__file__).resolve().parent
GEN = SRC.parent / "data" / "generated"
OUT = GEN / "evolution"
PREFIX = ("PREFIX : <http://kg26.example/anime#>\n"
          "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n")
TTL_PREFIX = "@prefix : <http://kg26.example/anime#> .\n@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
MODEL, K, CHUNK = "lightgcn_k3", 10, 500

RULES = {
    "domain typing": "CONSTRUCT { ?s a ?C } WHERE { ?p rdfs:domain ?C . ?s ?p ?o }",
    "range typing": "CONSTRUCT { ?o a ?C } WHERE { ?p rdfs:range ?C . ?s ?p ?o . FILTER(isIRI(?o)) }",
    "likes": "CONSTRUCT { ?u :likes ?a } WHERE { ?u :rated ?r . ?r :ratingOf ?a ; :ratingValue ?v . FILTER(?v >= 8) }",
}
SUBCLASS = "CONSTRUCT { ?s a ?D } WHERE { ?s a ?C . ?C rdfs:subClassOf ?D }"


def tbox() -> str:
    """ontology.ttl without its small sample ABox (only schema triples are needed)."""
    text = (SRC / "ontology.ttl").read_text()
    return text[:text.index("# ABox")] if "# ABox" in text else text


def apply_rules(data: str, tb: str) -> tuple[set, dict]:
    g = Graph()
    g.parse(data=tb + "\n" + data, format="turtle")
    new, counts = set(), {}
    for name, rule in RULES.items():
        got = {t for t in g.query(PREFIX + rule) if t not in g}
        counts[name] = len(got)
        new |= got
    for t in new:
        g.add(t)
    sub = {t for t in g.query(PREFIX + SUBCLASS) if t not in g}
    counts["subclass"] = len(sub)
    return new | sub, counts


def logic_completion() -> dict:
    lines = (GEN / "abox.ttl").read_text().splitlines()
    head = [l for l in lines[:10] if l.startswith("@prefix")]
    start = next(i for i, l in enumerate(lines) if l.startswith("#### Reified ratings"))
    tb = tbox()

    parts = ["\n".join(lines[:start])]  # content + vocabulary nodes
    by_user: dict[str, list[str]] = {}
    for l in lines[start:]:
        if m := re.match(r":(user_\d+) ", l):
            by_user.setdefault(m.group(1), []).append(l)
    users = sorted(by_user)
    for i in range(0, len(users), CHUNK):
        parts.append("\n".join(head + [l for u in users[i:i + CHUNK] for l in by_user[u]]))

    inferred, totals = set(), {}
    for part in parts:
        got, counts = apply_rules(part, tb)
        inferred |= got
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
    with open(OUT / "inferred.ttl", "w") as f:
        f.write("# Logic-derived, certain. Produced by evolve_kg.py from abox.ttl + ontology.ttl.\n")
        for s, p, o in sorted(inferred):
            f.write(f"<{s}> <{p}> <{o}> .\n")
    return {"partitions": len(parts), "new_triples": len(inferred), "by_rule": totals}


@torch.no_grad()
def ml_completion(split, ev) -> tuple[dict, dict]:
    E, n2i = load_final_embeddings(MODEL, split)
    users = sorted(u for u in set(split["train"].h) if u.startswith("user_") and u in n2i)
    cand = [a for a in ev.candidates if a in n2i]
    C = E[[n2i[a] for a in cand]]
    preds = {}
    for i in range(0, len(users), 512):
        ub = users[i:i + 512]
        S = E[[n2i[u] for u in ub]] @ C.T
        for r, u in enumerate(ub):
            known = list(ev.seen.get(u, ())) + list(ev.valid.get(u, ()))
            S[r, known] = -float("inf")  # predict only what is not already a fact
        top = S.topk(K, dim=1)
        for r, u in enumerate(ub):
            preds[u] = [(cand[j], float(s)) for j, s in zip(top.indices[r].tolist(), top.values[r].tolist())]
    return preds, {"users": len(users), "predictions": sum(map(len, preds.values()))}


def write_predictions(preds: dict, status: dict) -> None:
    with open(OUT / "predicted.ttl", "w") as f:
        f.write(f"# ML-derived, uncertain. {MODEL} top-{K} unseen anime per watcher (evolve_kg.py).\n")
        f.write(TTL_PREFIX)
        for u, items in preds.items():
            uid = u.split("_")[1]
            for rank, (a, s) in enumerate(items, 1):
                p = f":pred_{uid}_{a.split('_')[1]}"
                f.write(f"{p} a :Prediction ; :predictedFor :{u} ; :predictedAnime :{a} ; "
                        f":predictionScore \"{s:.4f}\"^^xsd:decimal ; :predictionRank {rank} ; "
                        f":predictedBy \"{MODEL}\" ; :hasStatus :{status.get((u, a), 'open')} .\n")


def write_summary(log: dict) -> None:
    lg, ml, up = log["logic"], log["ml"], log["update"]
    fired = sum(lg["by_rule"].values())
    md = f"""# KG evolution — what changed on disk

| step | file | effect |
|---|---|---|
| base KG | `abox.ttl` | {log['base_kg_triples']:,} triples |
| 1. logic completion (certain) | `evolution/inferred.ttl` | **+{lg['new_triples']:,}** unique new triples from {fired:,} rule firings (domain typing {lg['by_rule']['domain typing']:,}, range typing {lg['by_rule']['range typing']:,}, subclass {lg['by_rule']['subclass']:,}, likes {lg['by_rule']['likes']:,}); the same type is often derived by several rules. {lg['partitions']} partitions, {lg['seconds']} s |
| 2. ML completion (scored) | `evolution/predicted.ttl` | **+{ml['predictions']:,}** reified predictions for {ml['users']:,} watchers ({ml['predicted_ttl_triples']:,} triples), status `open` |
| 3. update (new facts arrive) | `evolution/predicted.ttl` | {up['arriving_ratings']:,} held-out ratings replayed → **{up['confirmed']:,}** predictions changed `open → confirmed` ({up['precision_of_predictions']:.1%} of all predictions; {up['users_with_a_confirmed_prediction']:,} watchers got at least one); {up['still_open']:,} stay open |
"""
    (RESULTS / "evolution.md").write_text(md)
    print(md)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = {"base_kg_triples": 2_492_955}

    t0 = time.time()
    log["logic"] = logic_completion()
    log["logic"]["seconds"] = round(time.time() - t0)
    print("logic:", log["logic"])

    split = load_split()
    ev = Evaluator(split)
    preds, log["ml"] = ml_completion(split, ev)
    write_predictions(preds, {})
    log["ml"]["predicted_ttl_triples"] = len(Graph().parse(OUT / "predicted.ttl", format="turtle"))
    print("ml:", log["ml"])

    # update: held-out ratings arrive as new facts -> matching predictions become confirmed
    arriving = set(zip(split["test"].h, split["test"].t))
    status = {(u, a): "confirmed" for u, items in preds.items() for a, _ in items if (u, a) in arriving}
    write_predictions(preds, status)
    users_hit = len({u for u, _ in status})
    log["update"] = {"arriving_ratings": len(arriving), "confirmed": len(status),
                     "still_open": log["ml"]["predictions"] - len(status),
                     "precision_of_predictions": round(len(status) / log["ml"]["predictions"], 4),
                     "users_with_a_confirmed_prediction": users_hit}
    print("update:", log["update"])
    (OUT / "evolution_log.json").write_text(json.dumps(log, indent=2))

    write_summary(log)


if __name__ == "__main__":
    main()
