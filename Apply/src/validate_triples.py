"""Apply/triples — validate the generated ABox.

Three levels, because they cost wildly different amounts:

  1. SYNTAX + SCALE   parse abox.ttl with rdflib (~130s at 2.5M triples).
     A clean parse is the real proof the emitter escapes literals correctly.
  2. STRUCTURE        SPARQL/graph checks on the FULL graph, no reasoning:
     every :Rating is well-formed, no dangling :ratingOf, the shared-hub
     property that motivated D1 actually holds at scale, etc.
  3. ENTAILMENT       OWL-RL closure on a SLICE (TBox + a few anime and their
     ratings). Full-graph closure is not attempted: OWL-RL is roughly
     quadratic in places and would materialize tens of millions of triples.
     The TBox axioms are global, so a slice is sufficient to show they fire
     on *generated* data (stage 1 already proved they fire on hand-written data).

Run:  python Apply/src/validate_triples.py
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import owlrl
from rdflib import Graph, Namespace, RDF, URIRef
from rdflib.namespace import RDFS

SRC = Path(__file__).resolve().parent
GEN = SRC.parent / "data" / "generated"
EX = Namespace("http://kg26.example/anime#")

results: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append((label, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))


def main() -> None:
    # ---------------- 1. syntax + scale ----------------------------------
    print("1. SYNTAX + SCALE")
    t = time.time()
    g = Graph()
    g.parse(GEN / "abox.ttl", format="turtle")
    print(f"  parsed {len(g):,} triples in {time.time() - t:.1f}s")
    check("abox.ttl parses as valid Turtle", len(g) > 0)

    # ---------------- 2. structure (no reasoning) ------------------------
    # Deliberately NOT SPARQL: `FILTER NOT EXISTS` over millions of triples is
    # pathologically slow in rdflib. These set differences hit the same indexes
    # directly and run in seconds.
    print("\n2. STRUCTURE (full graph, no reasoning)")

    titled = set(g.subjects(EX.title, None))
    rated_of = set(g.subjects(EX.ratingOf, None))
    has_value = set(g.subjects(EX.ratingValue, None))
    reached = set(g.objects(None, EX.rated))
    targets = set(g.objects(None, EX.ratingOf))
    watchers = set(g.subjects(EX.rated, None))
    print(f"  {len(titled):,} anime | {len(watchers):,} watchers | "
          f"{len(rated_of):,} rating nodes")

    # Every reified Rating needs a target AND a value, else the 2-hop path
    # cannot be collapsed into a flat edge for the embedding stage.
    missing_val = rated_of - has_value
    check("every :Rating has a :ratingValue", not missing_val,
          f"{len(missing_val):,} missing")

    dangling = targets - titled
    check("no :ratingOf points at a contentless Anime", not dangling,
          f"{len(dangling):,} dangling")

    unreachable = rated_of - reached
    check("every :Rating is reachable from some :Watcher", not unreachable,
          f"{len(unreachable):,} unreachable")

    # Value domain. -1 must SURVIVE here: the flat TSV is where it gets
    # promoted to a relation; the .ttl keeps the raw recorded fact.
    vals = Counter(int(o) for o in g.objects(None, EX.ratingValue))
    check("rating = -1 preserved in reified graph", vals.get(-1, 0) > 0,
          f"{vals.get(-1, 0):,} watched-unrated")
    lo, hi = min(vals), max(vals)
    check("ratingValue in {-1} ∪ [1,10]",
          lo == -1 and hi == 10 and 0 not in vals, f"[{lo}, {hi}], no 0")

    # D1's whole justification: companies recur across anime AND across roles.
    org_deg: Counter = Counter()
    for pred in (EX.animatedBy, EX.producedBy, EX.licensedBy):
        for a, o in g.subject_objects(pred):
            org_deg[o] += 1
    shared = sum(1 for c in org_deg.values() if c > 1)
    check("D1 holds at scale: organizations are shared hubs", shared > 100,
          f"{shared:,} of {len(org_deg):,} orgs on >1 anime")

    studios = set(g.objects(None, EX.animatedBy))
    producers = set(g.objects(None, EX.producedBy))
    multirole = studios & producers
    check("D1: same node reached by different role predicates", bool(multirole),
          f"{len(multirole):,} orgs are both studio and producer, e.g. " +
          ", ".join(sorted(str(o).split("#")[-1] for o in multirole)[:3]))

    # Vocabulary nodes must be labelled, so a slug stays human-decodable.
    typed = set(g.subjects(RDF.type, None))
    labelled = set(g.subjects(RDFS.label, None))
    unlabelled = typed - labelled
    check("every vocabulary node has an rdfs:label", not unlabelled,
          f"{len(unlabelled):,} unlabelled")

    # Junk-hub regression guard: the comma-split bug minted a ":Ltd" node that
    # falsely joined unrelated studios. It must not come back.
    junk = [s for s in ("Ltd", "Inc", "LLC", "Co", "Corp", "Limited")
            if (URIRef(EX + s), None, None) in g]
    check("no corporate-suffix junk nodes (comma-split regression)",
          not junk, f"found {junk}" if junk else "clean")

    # ---------------- 3. entailment on a slice ---------------------------
    print("\n3. ENTAILMENT (OWL-RL closure on a slice)")
    sl = Graph()
    sl.parse(SRC / "ontology.ttl", format="turtle")   # the TBox

    # Keep the slice SMALL and cap ratings per anime. OWL-RL materializes
    # class/property axioms pairwise, so it degrades badly with size — and the
    # popular anime (low MAL ids) carry hundreds of ratings each. A handful of
    # instances is enough: TBox axioms are global, so if they fire here they
    # fire everywhere.
    picked = sorted(titled, key=str)[:10]
    rating_nodes: set = set()
    users: set = set()
    for a in picked:
        for r in list(g.subjects(EX.ratingOf, a))[:3]:
            rating_nodes.add(r)
            users.update(g.subjects(EX.rated, r))

    # Copy the anime and rating nodes wholesale...
    for s in set(picked) | rating_nodes:
        for p, o in g.predicate_objects(s):
            sl.add((s, p, o))
    # ...but for a user, copy ONLY the :rated edges into this slice. A user has
    # ~140 ratings; pulling all of them in would drag the whole graph along.
    for u in users:
        for r in rating_nodes:
            if (u, EX.rated, r) in g:
                sl.add((u, EX.rated, r))
    # Vocabulary nodes the sliced anime point at, so range axioms have targets.
    for _, _, o in list(sl):
        if isinstance(o, URIRef):
            for p2, o2 in g.predicate_objects(o):
                if p2 == RDF.type or p2 == RDFS.label:
                    sl.add((o, p2, o2))

    before = len(sl)
    print(f"  slice: {len(picked)} anime, {len(rating_nodes)} ratings, "
          f"{len(users)} users -> {before:,} triples")

    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(sl)
    print(f"  slice: {before:,} asserted -> {len(sl):,} after closure")

    an = picked[0]
    check("generated Anime entailed a :Work (Anime ⊑ Work)",
          (an, RDF.type, EX.Work) in sl, str(an).split("#")[-1])

    # NOTE: rdflib's subjects()/objects() yield one item PER TRIPLE, not per
    # distinct node — always wrap in set() before counting.
    sl_watchers = set(sl.subjects(EX.rated, None))
    check("generated user entailed a :Watcher (domain of :rated)",
          bool(sl_watchers)
          and all((u, RDF.type, EX.Watcher) in sl for u in sl_watchers),
          f"{len(sl_watchers)} distinct watchers")

    rnodes = set(sl.subjects(EX.ratingOf, None))
    check("generated rating node entailed a :Rating (range of :rated)",
          bool(rnodes) and all((r, RDF.type, EX.Rating) in sl for r in rnodes),
          f"{len(rnodes)} distinct rating nodes")

    orgs = set(sl.objects(None, EX.animatedBy))
    check("studio entailed an :Organization (range of :animatedBy)",
          bool(orgs) and all((o, RDF.type, EX.Organization) in sl for o in orgs),
          f"{len(orgs)} distinct studios")

    check("NEG: deferred :StaffAssignment still unpopulated",
          not list(sl.triples((None, RDF.type, EX.StaffAssignment))))

    # ---------------- summary --------------------------------------------
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks pass")
    if n_fail:
        raise SystemExit(f"{n_fail} check(s) FAILED.")
    print("All checks pass.")


if __name__ == "__main__":
    main()
