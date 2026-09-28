"""Apply/triples — generate the full ABox from anime.csv + rating.csv.

Two artifacts from one pass, because the reified graph and the embedding input
want different shapes:

  abox.ttl     reified, semantically faithful:  Watcher -rated-> Rating -ratingOf-> Anime
               with :ratingValue on the Rating node (-1 kept as-is). Loads with
               ontology.ttl for OWL-RL validation. This is the KG (LO7/LO5).

  triples.tsv  flat (head, relation, tail) for KGE libraries (LO1). The reified
               2-hop path is COLLAPSED to a direct edge, and the -1 sentinel is
               promoted to its own relation:
                   rating 1..10  ->  user_<u>  rated    anime_<a>
                   rating == -1  ->  user_<u>  watched  anime_<a>
               Datatype properties are omitted here on purpose (see notes.md).

Turtle is streamed to disk rather than assembled in an rdflib Graph: at
--n-users 20000 the reified graph is ~8.6M triples, which does not fit
comfortably in memory.

Run:  python Apply/src/build_triples.py [--n-users 5000] [--min-ratings 20] [--seed 26]
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

APPLY = Path(__file__).resolve().parent.parent
DATA = APPLY / "data"
NS = "http://kg26.example/anime#"

# Columns the ontology actually touches. `year` is ~78% NaN -> use
# aired_prop_from_year for :releaseYear (EDA finding, see data/readme.md).
ACOLS = [
    "mal_id", "title", "type", "source", "episodes", "status", "score", "members",
    "aired_prop_from_year", "producers", "licensors", "studios",
    "genres", "themes", "demographics",
]

# multi-valued column -> predicate. One shared node per distinct value (D1/D2).
MULTI_PRED = {
    "genres": "hasGenre",
    "themes": "hasTheme",
    "demographics": "hasDemographic",
    "studios": "animatedBy",     # D1: role on the predicate, one :Organization class
    "producers": "producedBy",
    "licensors": "licensedBy",
}
SINGLE_PRED = {"type": "animeType", "source": "hasSource"}


# --------------------------------------------------------------------------
# IRI minting
# --------------------------------------------------------------------------

def slug(name: str) -> str:
    """'Sci-Fi' -> 'SciFi', 'Audio Planning U' -> 'AudioPlanningU'.

    Matches the hand-written convention in ontology.ttl. Strips accents so the
    IRI stays ASCII, then drops every non-alphanumeric character.
    """
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    parts = re.split(r"[^0-9A-Za-z]+", ascii_only)
    out = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return out or "_unnamed"


def esc(s: str) -> str:
    """Escape a Python str for a Turtle double-quoted literal."""
    return (
        str(s)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


# Company names contain commas: "Horgos Coloroom Pictures Co., Ltd." is ONE
# organization. A naive comma split mints a junk ":Ltd" node that becomes a
# false shared hub joining unrelated studios — poison for embeddings. Any token
# that is *only* a corporate suffix gets re-attached to the previous value.
CORP_SUFFIX = re.compile(
    r"^(?:Ltd|Inc|LLC|L\.L\.C|Co|Corp|Corporation|Company|Limited|GmbH|K\.K"
    r"|S\.A|S\.A\.S|B\.V|N\.V|Pty|Pte|PLC|AB|AG|SARL)\.?$",
    re.IGNORECASE,
)


def split_multi(cell) -> list[str]:
    """Comma-separated multi-value cell -> list of trimmed values.

    Re-joins corporate-suffix fragments produced by splitting inside a
    company name (see CORP_SUFFIX).
    """
    if pd.isna(cell):
        return []
    out: list[str] = []
    for tok in (t.strip() for t in str(cell).split(",")):
        if not tok:
            continue
        if out and CORP_SUFFIX.match(tok):
            out[-1] = f"{out[-1]}, {tok}"
        else:
            out.append(tok)
    return out


# --------------------------------------------------------------------------
# Load + sample
# --------------------------------------------------------------------------

def load_anime() -> pd.DataFrame:
    anime = pd.read_csv(DATA / "anime.csv", usecols=ACOLS)
    before = len(anime)
    anime = anime.drop_duplicates(subset="mal_id", keep="first").reset_index(drop=True)
    print(f"anime.csv: {before:,} rows -> {len(anime):,} unique mal_id "
          f"({before - len(anime)} duplicate rows dropped)")
    return anime


def sample_ratings(min_ratings: int, n_users: int, seed: int, valid_anime: set[int]):
    """Sample whole user profiles. Keeping a profile intact matters for a
    recommender: a user's taste vector is the unit of signal, so we sample the
    user axis rather than slicing individual edges."""
    rating = pd.read_csv(DATA / "rating.csv")
    total = len(rating)

    # Drop edges pointing at anime with no content row (73 anime / ~6.9k edges).
    orphan_mask = ~rating.anime_id.isin(valid_anime)
    n_orphan = int(orphan_mask.sum())
    rating = rating[~orphan_mask]

    udeg = rating.groupby("user_id").size()
    eligible = udeg[udeg >= min_ratings].index.values
    rng = np.random.default_rng(seed)
    take = min(n_users, len(eligible))
    keep = rng.choice(eligible, size=take, replace=False)
    sub = rating[rating.user_id.isin(set(keep))].reset_index(drop=True)

    print(f"rating.csv: {total:,} edges -> dropped {n_orphan:,} orphan "
          f"({n_orphan / total * 100:.2f}%)")
    print(f"  eligible users (>= {min_ratings} ratings): {len(eligible):,}; "
          f"sampled {take:,} (seed={seed})")
    print(f"  kept {len(sub):,} rating edges over {sub.anime_id.nunique():,} anime")
    return sub, n_orphan, len(eligible)


# --------------------------------------------------------------------------
# Emit
# --------------------------------------------------------------------------

def write_graph(anime: pd.DataFrame, ratings: pd.DataFrame, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    ttl_path, tsv_path = outdir / "abox.ttl", outdir / "triples.tsv"

    # Registry of vocabulary nodes: slug -> raw label, so we can (a) emit
    # rdfs:label and (b) detect two distinct names colliding on one IRI.
    vocab: dict[str, dict[str, str]] = {}   # slug -> {"label":…, "class":…}
    collisions: dict[str, set[str]] = {}
    n_ttl = n_tsv = 0

    def register(raw: str, cls: str) -> str:
        s = slug(raw)
        prev = vocab.get(s)
        if prev is None:
            vocab[s] = {"label": raw, "class": cls}
        elif prev["label"] != raw:
            collisions.setdefault(s, {prev["label"]}).add(raw)
        return s

    with ttl_path.open("w", encoding="utf-8") as ttl, \
         tsv_path.open("w", encoding="utf-8") as tsv:

        ttl.write(
            "# GENERATED by Apply/src/build_triples.py — do not edit by hand.\n"
            "# Full ABox. Load together with ../src/ontology.ttl (the TBox).\n"
            f"@prefix :     <{NS}> .\n"
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
            "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .\n\n"
        )

        # ---- Anime content -------------------------------------------------
        ttl.write("#### Anime content (from anime.csv)\n\n")
        for row in anime.itertuples(index=False):
            subj = f":anime_{row.mal_id}"
            lines: list[str] = []       # Turtle predicate-object lines
            n_row = 0                   # triples on this row (a `,` list is N, not 1)

            # Datatype properties. Emit only when present — missingness is the
            # norm here (score ~35% NaN, etc.), and a null node would be a lie.
            lines.append(f'    :title "{esc(row.title)}"')
            if pd.notna(row.episodes):
                lines.append(f"    :episodeCount {int(row.episodes)}")
            yr = row.aired_prop_from_year
            if pd.notna(yr) and int(yr) > 0:
                lines.append(f'    :releaseYear "{int(yr)}"^^xsd:gYear')
            if pd.notna(row.status):
                lines.append(f'    :status "{esc(row.status)}"')
            if pd.notna(row.score):
                lines.append(f'    :communityScore "{row.score}"^^xsd:decimal')
            if pd.notna(row.members):
                lines.append(f"    :memberCount {int(row.members)}")
            n_row += len(lines)     # one triple per datatype line

            # Object properties -> shared vocabulary nodes. These are the edges
            # that give the graph its connectivity, so they also go to the TSV.
            for col, pred in SINGLE_PRED.items():
                val = getattr(row, col)
                if pd.isna(val):
                    continue
                cls = "AnimeType" if col == "type" else "Source"
                s = register(str(val).strip(), cls)
                lines.append(f"    :{pred} :{s}")
                n_row += 1
                tsv.write(f"anime_{row.mal_id}\t{pred}\t{s}\n")
                n_tsv += 1

            for col, pred in MULTI_PRED.items():
                vals = split_multi(getattr(row, col))
                if not vals:
                    continue
                cls = {"genres": "Genre", "themes": "Theme",
                       "demographics": "Demographic"}.get(col, "Organization")
                # dict.fromkeys dedupes while keeping order: a value listed
                # twice is ONE triple (RDF is a set), so counting the raw list
                # would over-report.
                slugs = list(dict.fromkeys(register(v, cls) for v in vals))
                lines.append(f"    :{pred} " + " , ".join(f":{s}" for s in slugs))
                n_row += len(slugs)
                for s in slugs:
                    tsv.write(f"anime_{row.mal_id}\t{pred}\t{s}\n")
                    n_tsv += 1

            ttl.write(f"{subj}\n" + " ;\n".join(lines) + " .\n")
            n_ttl += n_row

        # ---- Vocabulary nodes: explicit type + human-readable label --------
        # The TBox range axioms would *derive* these types, but asserting them
        # keeps the ABox self-contained (and readable without a reasoner).
        ttl.write("\n#### Vocabulary nodes (shared hubs)\n\n")
        for s, meta in sorted(vocab.items()):
            ttl.write(f':{s} a :{meta["class"]} ; rdfs:label "{esc(meta["label"])}" .\n')
            n_ttl += 2

        # ---- Reified ratings ----------------------------------------------
        ttl.write("\n#### Reified ratings (from rating.csv)\n"
                  "# -1 = watched but unrated (implicit feedback, NOT missing).\n\n")
        n_rated = n_watched = 0
        for u, a, v in zip(ratings.user_id.values,
                           ratings.anime_id.values,
                           ratings.rating.values):
            rt = f":rt_{u}_{a}"
            ttl.write(f":user_{u} :rated {rt} . {rt} :ratingOf :anime_{a} ; "
                      f":ratingValue {v} .\n")
            n_ttl += 3
            # Flat view: collapse the 2-hop path, promote -1 to a relation.
            if v == -1:
                tsv.write(f"user_{u}\twatched\tanime_{a}\n")
                n_watched += 1
            else:
                tsv.write(f"user_{u}\trated\tanime_{a}\n")
                n_rated += 1
            n_tsv += 1

    if collisions:
        print(f"\n  !! {len(collisions)} IRI slug collision(s) — distinct names "
              f"sharing one node:")
        for s, names in list(collisions.items())[:10]:
            print(f"     :{s} <- {sorted(names)}")

    vocab_counts = Counter(m["class"] for m in vocab.values())
    return {
        "ttl_triples": n_ttl, "tsv_triples": n_tsv,
        "rated_edges": n_rated, "watched_edges": n_watched,
        "vocab_nodes": dict(vocab_counts), "slug_collisions": len(collisions),
        "ttl_path": str(ttl_path), "tsv_path": str(tsv_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-users", type=int, default=5000,
                   help="how many user profiles to sample (default 5000)")
    p.add_argument("--min-ratings", type=int, default=20,
                   help="only sample users with at least this many ratings")
    p.add_argument("--seed", type=int, default=26, help="sampling seed")
    p.add_argument("--out-dir", type=Path, default=DATA / "generated")
    args = p.parse_args()

    anime = load_anime()
    valid = set(anime.mal_id.unique())
    ratings, n_orphan, n_eligible = sample_ratings(
        args.min_ratings, args.n_users, args.seed, valid)

    print("\nwriting graph...")
    stats = write_graph(anime, ratings, args.out_dir)

    prov = {
        "params": {"n_users": args.n_users, "min_ratings": args.min_ratings,
                   "seed": args.seed},
        "sampled": {
            "users": int(ratings.user_id.nunique()),
            "eligible_users": n_eligible,
            "anime_with_content": int(len(anime)),
            "anime_reached_by_ratings": int(ratings.anime_id.nunique()),
            "rating_edges": int(len(ratings)),
            "orphan_edges_dropped": n_orphan,
        },
        "output": stats,
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))

    print(f"\n  abox.ttl     {stats['ttl_triples']:>10,} triples (reified)")
    print(f"  triples.tsv  {stats['tsv_triples']:>10,} triples (flat, for KGE)")
    print(f"               rated={stats['rated_edges']:,}  "
          f"watched={stats['watched_edges']:,}")
    print(f"  vocab nodes  {stats['vocab_nodes']}")
    print(f"\nwrote {args.out_dir}/ (abox.ttl, triples.tsv, provenance.json)")


if __name__ == "__main__":
    main()
