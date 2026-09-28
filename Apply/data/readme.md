# Apply data — sources & decisions

**Sources.** `anime.csv`: MyAnimeList-derived *Anime Dataset* — https://hyper.ai/en/datasets/42346 (torrent mirror of the Kaggle release; the exact file used ships with the project). `rating.csv`: https://github.com/rai-shivangi/Anime_-recommendation_-system (`rating.csv.zip`, SHA-1 verified identical to the file used; originally the Kaggle *Anime Recommendations Database*).

Data files live directly in `Apply/data/` (gitignored — only `.gitkeep` + this readme
are tracked; a fresh clone must drop the CSVs here before running the pipeline).

## Files there
| file | rows | key | role |
|------|------|-----|------|
| `anime.csv` | 28,858 rows → **28,627 unique** `mal_id` (231 dup rows to drop; `wc -l` over-counts to 70,850 — synopses have embedded newlines; **parse with pandas**) | `mal_id` | **content source** (chosen; was `anime2.csv`, renamed) |
| `rating.csv` | 7,813,737 edges | `(user_id, anime_id)` | **Watcher–Anime edges** (chosen) |

Note: an earlier thin Kaggle subset (also called `anime.csv`, keyed on `anime_id`,
12,294 rows) was evaluated and **dropped** as redundant — see the join-coverage numbers
below, kept for the record. The file now named `anime.csv` is the former `anime2.csv`
content source.

## Committed decisions

**Downstream task = recommender / link prediction.** Predict `Watcher–Anime` edges.
This ratifies the `Watcher` + reified `Rating` design from Learn/01. (MAIN LOs 1 & 3.)

**Source architecture = `anime.csv` (content, ex-`anime2.csv`) + `rating.csv` (edges).**
The dropped Kaggle subset's columns were a strict subset of anime2 on the same MAL
key — redundant.

**ID alignment — RESOLVED, no entity resolution needed.** Files key on the
MyAnimeList id (Kaggle `anime_id` == MAL `mal_id`). Measured overlap:
- rated anime ∩ anime (content) mal_id: 11,127 / 11,200 (**99.3%**); only 73 orphan
  rating edges.
- rated anime ∩ dropped Kaggle subset: 11,197 / 11,200 (99.97%).
- dropped Kaggle subset ∩ anime (content): 12,181 / 12,294 (99.1%).

## Ontology modeling decisions (see `../src/ontology.ttl`)

- **D1 — Organizations merged.** `studios` / `producers` / `licensors` all → one
  `:Organization` class; the role is the *predicate* (`animatedBy` / `producedBy` /
  `licensedBy`). Overlapping companies (Funimation, Victor Entertainment, Sunrise…)
  stay a **single shared node** → better graph connectivity for embeddings/GNN.
- **D2 — type & source as NODES.** `type` (TV/Movie/…) → `:animeType → :AnimeType`;
  `source` (Manga/Original/…) → `:hasSource → :Source`. Shared hub nodes, not literals.
- **D3 — `score` vs `Rating` disambiguated.** anime.csv `score` = community aggregate →
  literal `:communityScore` on the Anime. A user's rating → reified `:Rating` node.
  Same word, different things.
- **`rating = -1` = "watched but unrated"** (implicit feedback, *not* missing). Kept
  as-is; the link-prediction target is the *edge existing*, independent of the value.
- **Deferred (no source yet):** cast/staff (`StaffAssignment`, `Role`) and
  `sequelOf`/`prequelOf`. Left OUT of the active TBox. Populate later when a
  cast/relations source is added (manami relations JSON, or looked-up facts) — on ask.

## Multi-value field format (anime.csv)
`studios, producers, licensors, genres, themes, demographics` are **comma-separated**
strings (e.g. `Bandai Visual, Victor Entertainment, Audio Planning U`; pandas strips the
outer quotes). Split on `","` and `.strip()` → one node per value.

**Caveat — company names contain commas.** `Horgos Coloroom Pictures Co., Ltd.` is ONE
organization; a naive comma split yields `Horgos Coloroom Pictures Co.` + `Ltd.`, and the
bare suffix becomes a **junk hub node** falsely joining unrelated studios (poison for
embeddings: a high-degree node with no meaning). 101 cells affected. `build_triples.py`
re-attaches any token matching a corporate suffix (`Ltd|Inc|LLC|Co|Corp|GmbH|K.K|…`) to
the previous value; `validate_triples.py` has a regression guard for the junk nodes.

## EDA findings (see [`../eda.ipynb`](../eda.ipynb))
- **`year` is unusable** (~78% NaN) → use **`aired_prop_from_year`** (only ~3% NaN/0) for
  `:releaseYear`.
- **Missingness is normal**: score ~35% NaN, studios ~40%, themes ~41%, demographics ~63%,
  licensors ~82%. Emit an edge only when the value is present — no null nodes.
- **Vocab sizes**: type=9, source=17 (tiny hubs); genres/themes are mid-size shared vocabs.
- **Rating graph is long-tailed**: 73,515 users × 11,200 anime, 7.81M edges (18.9% are `-1`).
  Top 5% of anime hold 55% of edges; top 25% hold 93%. Median user has 57 ratings.
  → subsample by **active users (`>=k` ratings) and/or a popular-anime core**, not blind random.
- **Orphan edges**: only 73 rated anime (6,916 edges, 0.09%) absent from anime.csv → drop at build.
