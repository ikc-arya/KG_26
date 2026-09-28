"""Apply/ontology — load the ontology, run the OWL-RL closure, and assert that
the intended entailments fire on the real-data sample. Parallels Learn/01.

Run:  python Apply/src/build_ontology.py
"""
from pathlib import Path
from rdflib import Graph, Namespace, RDF
from rdflib.namespace import RDFS
import owlrl

TTL = Path(__file__).with_name("ontology.ttl")
EX = Namespace("http://kg26.example/anime#")


def main() -> None:
    g = Graph()
    g.parse(TTL, format="turtle")
    asserted = len(g)

    # rdflib has no built-in reasoner; materialize the RDFS/OWL-RL closure in place.
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    closed = len(g)
    print(f"triples: {asserted} asserted -> {closed} after closure")

    def has(s, p, o) -> bool:
        return (s, p, o) in g

    checks = {
        # domain/range typing
        "anime_1 typed :Anime (range of :ratingOf / it has :animeType etc.)":
            has(EX.anime_1, RDF.type, EX.Anime),
        "anime_1 typed :Work (Anime ⊑ Work)":
            has(EX.anime_1, RDF.type, EX.Work),
        "Sunrise typed :Organization (range of :animatedBy)":
            has(EX.Sunrise, RDF.type, EX.Organization),
        "Funimation typed :Organization (range of :licensedBy)":
            has(EX.Funimation, RDF.type, EX.Organization),
        "TV typed :AnimeType (range of :animeType)":
            has(EX.TV, RDF.type, EX.AnimeType),
        "Action typed :Genre (range of :hasGenre)":
            has(EX.Action, RDF.type, EX.Genre),
        "user_19 typed :Watcher (domain of :rated)":
            has(EX.user_19, RDF.type, EX.Watcher),
        "rt_19_1 typed :Rating (range of :rated / domain of :ratingOf)":
            has(EX.rt_19_1, RDF.type, EX.Rating),
        # negative: deferred classes must NOT exist
        "NEG: no :StaffAssignment class in graph":
            not any(g.triples((None, RDF.type, EX.StaffAssignment))),
    }
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    # Shared-node proof (D1): which anime share a producer? Expect Victor
    # Entertainment -> anime_1 AND anime_6 as one node.
    print("\nShared-organization query (D1) — anime sharing a producer:")
    q = """
    PREFIX : <http://kg26.example/anime#>
    SELECT ?org (COUNT(DISTINCT ?a) AS ?n) WHERE {
        ?a :producedBy ?org .
    } GROUP BY ?org HAVING (COUNT(DISTINCT ?a) > 1)
    """
    for row in g.query(q):
        print(f"  {row.org.split('#')[-1]} produced {row.n} of the sample anime")

    # Reconstruct a user rating via the reified path (link-prediction target shape).
    print("\nReified rating join (Watcher -> Rating -> Anime):")
    q2 = """
    PREFIX : <http://kg26.example/anime#>
    SELECT ?u ?title ?v WHERE {
        ?u :rated ?r . ?r :ratingOf ?a ; :ratingValue ?v . ?a :title ?title .
    } ORDER BY ?u
    """
    for row in g.query(q2):
        u = row.u.split("#")[-1]
        print(f"  {u} rated \"{row.title}\" = {row.v}")

    if not all(checks.values()):
        raise SystemExit("Some entailment checks FAILED.")
    print("\nAll checks pass.")


if __name__ == "__main__":
    main()
