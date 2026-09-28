# Apply — pipeline notes

Real pipeline on anime data. One short entry per stage as it's built, tagged with the
LO(s) it advances. Newest at the bottom.

Stages: ontology → triples → embeddings → evolution → GNN.

Data sources & all dataset/modeling decisions live in [`data/readme.md`](data/readme.md).

---

## Stage: ontology  `[LO7, LO5]`

Artifact: [`src/ontology.ttl`](src/ontology.ttl) — TBox + a small **real-data** ABox
(anime 1/6/5114 from anime.csv + a few real rating.csv edges). Validator (next):
`src/build_ontology.py`.

**What the Apply ontology is.** Content graph from `anime.csv` + user-rating edges from
`rating.csv`, keyed by MAL id. Classes: `Work ⊒ Anime`, `Organization`, `Genre`, `Theme`,
`Demographic`, `AnimeType`, `Source`, `Watcher`, `Rating`.

**How it diverges from the Learn/01 dummy ontology:**

- Staff reification (`StaffAssignment`) and `sequelOf`/`prequelOf` are **dropped** —
  anime.csv has no cast or relation columns to populate them (deferred, see data/readme).
- `studios`/`producers`/`licensors` collapse into one `:Organization` (role on the edge),
  where the dummy had a single `producedBy → :Studio`.
- Rich real vocabularies now present: `Theme`, `Demographic`, `AnimeType`, `Source` —
  none existed in the dummy.

**Key modeling facts (rationale in data/readme.md):**

- One `:Organization` node per company, reused across roles → shared nodes (Funimation
  licenses anime 1 & 6; Victor Entertainment produces both). This connectivity is the
  point for LO1/LO3.
- `type`/`source` are nodes, not literals — controlled-vocab hub nodes.
- `communityScore` (aggregate literal) ≠ `Rating` (reified per-user node). The reified
  `Watcher →rated→ Rating →ratingOf→ Anime` path is the **link-prediction target**;
  recommendation *is* link prediction on those edges.
- `ratingValue = -1` = watched-but-unrated (implicit feedback, not missing).

**Scope note.** The `.ttl` carries only a 3-anime sample ABox to make reasoning visible.
Generating the full ABox (~38k anime + 7.8M edges) from the CSVs is the **triples** stage.

---

## Stage: triples  `[LO7, LO1, LO4]`

Artifacts: [`src/build_triples.py`](src/build_triples.py) (generator) +
[`src/validate_triples.py`](src/validate_triples.py) (15 checks, all pass).
Output → `data/generated/` (gitignored): `abox.ttl`, `triples.tsv`, `provenance.json`.

**Two artifacts, because the KG and the embedding input want different shapes.**
This is the central lesson of the stage:

|          | `abox.ttl`                                                                          | `triples.tsv`                         |
| -------- | ------------------------------------------------------------------------------------- | --------------------------------------- |
| rating   | reified:`Watcher →rated→ Rating →ratingOf→ Anime`, `:ratingValue` on the node | flat:`user_<u> rated anime_<a>`       |
| `-1`   | kept as the literal value                                                             | promoted to its own relation`watched` |
| literals | present (`:title`, `:communityScore`, …)                                         | omitted                                 |
| size     | 2,492,955 triples                                                                     | 902,321 triples                         |

- **Reification costs 3×.** One rating = 3 triples. That alone makes the full 7.8M-edge
  graph (23.4M triples) infeasible in rdflib, so subsampling is *forced*, not a taste.
- **KGE models can't score a reified path.** They consume `(head, relation, tail)`.
  Reification turns the one edge we want to predict into a 2-hop path with the signal
  parked on a *literal* — unscoreable. So the flat file **collapses** the path back.
  → Reification buys semantic fidelity and costs machine-learnability. Faithful RDF is
  not automatically good ML input; the flat file is a *lossy projection chosen on purpose*.
- **`-1` → its own relation** in the flat view: a sentinel inside a numeric field is a lie
  to any model that reads it as a magnitude. Making it *structure* (`watched` vs `rated`)
  gives two clean signals. The `.ttl` still records the raw fact.
- **Literals dropped from the TSV.** `:title` is unique per anime → a relation with
  fan-out 1 teaches an embedding nothing. Only object properties carry relational signal.

**Sampling: sample the user axis, keep profiles intact.** The long tail is in *anime*, not
users — restricting to the top 2,000 anime only cuts 7.8M → 6.2M edges, because popular
titles hold nearly everything. Cutting users is what actually shrinks the graph. And a
user's *whole* rating profile is the unit of signal for a recommender, so we sample whole
users rather than random edges. Parameterized: `--n-users` (default 5000), `--min-ratings`
(20), `--seed` (26), all recorded in `provenance.json`.

**Two bugs the pipeline caught — both are data-modeling lessons, not typos:**

1. **Comma-split shredded company names.** `Horgos Coloroom Pictures Co., Ltd.` → a junk
   `:Ltd` node that became a **false shared hub** linking unrelated studios. Exactly the
   failure D1 (shared org nodes) is *supposed* to exploit, inverted into garbage: a
   high-degree meaningless node. Caught by an IRI-collision detector (`Ltd` vs `Ltd.`
   mapping to one slug), not by eyeballing. Slug collisions are a cheap canary.
   - The *other* collision, `J.C.F.` vs `JCF`, is the same company written two ways —
     there merging onto one node is correct. Same signal, opposite verdicts.
2. **Counting lines ≠ counting triples.** `:hasGenre :A , :B , :C` is one Turtle line but
   three triples; a value listed twice is one triple (RDF is a **set**). My count was
   46,668 short until reconciled against rdflib's independent parse.

**Validation is tiered, because cost differs by orders of magnitude.**
Full-graph OWL-RL closure is impossible (2.5M asserted; OWL-RL materializes pairwise).
So: (1) full parse = syntax/escaping proof, ~200s; (2) structure via **set differences on
rdflib's indexes**, seconds — `FILTER NOT EXISTS` in SPARQL is pathologically slow at this
size; (3) entailment via closure on a ~300-triple slice. TBox axioms are *global*, so if
they fire on a slice of generated data they fire everywhere.

**rdflib gotcha.** `g.subjects(p, None)` yields one item **per matching triple**, not per
distinct node (`unique=False` by default). Always `set()` before counting — this silently
reported "13,805 watchers" for a 10-anime slice.

**D1 confirmed at scale:** 1,945 of 2,551 organizations attach to >1 anime, and 487 are
both studio *and* producer — the same node reached by different role predicates. The
connectivity D1 was designed for is real, not hypothetical.

## Stage: embeddings  `[LO1]`

- [`src/make_split.py`](src/make_split.py) → per-user holdout of `rated` edges (10% test / 10% valid, min 1 each, cold-start edges dropped); content + `watched` stay in train.
- [`src/recsys_eval.py`](src/recsys_eval.py) → the one evaluator every model goes through (+ popularity baseline).
- [`src/train_kge.py`](src/train_kge.py) → hand-written DistMult and TransE sharing one loss, one negative sampler, one evaluator: **only the scoring function differs**. Default settings, one run each, no tuning — the aim is to explain differences, not maximise them.
- [`src/probe_embeddings.py`](src/probe_embeddings.py) → what each model learned, by title (`data/generated/results/probe.md`).
- Trials behind every choice: [`embeddings_trials.ipynb`](embeddings_trials.ipynb) (1k-user subset) + toy versions in `Learn/03_embeddings/embeddings_trials.ipynb`.

**Split reproducibility.** The script that made `data/generated/split/` was lost before commit. The rule is recovered exactly from the files (per-user counts = max(1, ⌊10%·n⌋)) and re-implemented in `make_split.py`, but the random draw can't be recovered (every tried procedure overlaps the original at chance level, ~10%). So the original split is shipped as data. A regenerated split has identical train size (791,640) and users (4,732). Results hold: DistMult test recall@10 **0.208 on both** (NDCG 0.183 vs 0.182, HR 0.675 vs 0.668), popularity 0.124 vs 0.121 → the reported numbers don't hinge on the particular draw.

**Evaluate as a recommender, not as a KG benchmark.** For each user rank *all* 8,001 candidate anime, mask what they already touched (filtered), score Recall/NDCG/HR@10 **averaged per user** — per-edge averaging would let heavy users dominate. Popularity is the bar: on long-tailed data "recommend what everyone watches" is strong. Unfiltered ranking would put known train edges in 65% of DistMult's top-10 slots (recall 0.074 vs 0.208).

| model (test, K=10) | Recall | NDCG | HR |
|---|---|---|---|
| popularity | 0.121 | 0.115 | 0.476 |
| DistMult, init std 1e-3 (+ L2) | 0.000 | 0.000 | 0.000 |
| **DistMult, init std 0.1** | **0.208** | **0.182** | **0.668** |
| TransE (γ = 6) | 0.193 | 0.176 | 0.617 |

**The init × L2 freeze (why the Sep 23 run never trained).** Loss sat at exactly **2·ln 2** = softplus of 0 for the positive and the negative term = every score is zero. Isolated on the 1k subset: tiny init + L2 → frozen; tiny init without L2 → trains (0.165); init 0.1 + L2 → trains (0.179). Neither factor alone does it. Likely: DistMult's data gradient on an embedding ∝ init² (product of the other two vectors), L2 pull ∝ init → at 1e-3 the pull wins. A small full-batch toy doesn't reproduce the freeze at all, so the mechanism stays a hypothesis; the cause is established empirically.
→ **Know your loss's value at zero scores; a loss stuck there is a dead model, not a weak one.**

**Negatives come from the relation's range.** Corrupting `(user, rated, anime)` with any entity mostly yields genres/users/studios (only 19% of entities are anime) → loss 0.05 but recall 0.152 vs 0.179. Low loss ≠ good model.

**Content edges help the KGE** → no genre/studio/source edges: 0.179 → 0.166 on the subset.

**DistMult vs TransE — a direction vs a point.** Our graph is almost all 1-N / N-N between disjoint types → favours the bilinear family, hits TransE's weak spot. Confirmed modestly on the metric (0.193 vs 0.208) and clearly in the **training loss: TransE plateaus at 0.61, DistMult reaches 0.27.**
- TransE demands `user + r_rated ≈ anime` for *each* of a user's ~97 anime → the user's taste becomes one point at their centre. DistMult: a user is a *direction*, many anime can score high along it without being in the same place.
- Toy check (`Learn/03`): TransE still *ranks* two-taste users perfectly but can't place their anime *close* (loss 1.10 vs 0.95) → the limit costs fit more than order, same as the real picture.
- Visible in the geometry: popular anime must sit near *many* users' `u + r` → pulled to the middle, **corr(log popularity, vector norm) = −0.87 for TransE** (−0.45 DistMult) → TransE recommends from a narrower pool: **1,847** distinct anime across all users vs **2,778**.
- DistMult's symmetry (f(h,r,t) = f(t,r,h), toy: identical scores to 1e-6) costs nothing here — user and anime never swap roles. It would bite on `sequelOf` (deferred).
- On the 1k subset the order **flips** (TransE 0.200 vs DistMult 0.179): DistMult overfits after epoch 2, TransE keeps climbing → rigid geometry = regulariser on thin data, ceiling on plenty. Subset trials decide mechanisms, not close model picks.

**Both learned real structure with no labels** (`probe.md`): Ghibli films cluster, K-On! sits with Lucky☆Star / Toradora! / Haruhi, Death Note with Code Geass / FMA:B / Elfen Lied. DistMult also groups franchises (K-On!! and the movie).

**One example is not a pattern.** The probe's random user (`user_66835`, who rated one adult title 10) got an all-Hentai TransE top-10 → looked like "TransE gets captured by tight niche clusters". Over all 4,732 users: false (0.5% of users get ≥5 such recs). The anecdote stays in the probe only next to the aggregate.

---

## Stage: GNN  `[LO3]`

[`src/train_gnn.py`](src/train_gnn.py) → hand-written LightGCN (BPR loss): E_{k+1} = D^-1/2 A D^-1/2 · E_k, final = mean over layers, score = dot product. No weights, no nonlinearity → `--layers 0` **is** MF, so each ablation isolates one idea. `--content` adds anime ↔ genre/theme/studio/… edges (types ignored). Trials: [`gnn_trials.ipynb`](gnn_trials.ipynb).

| model (test, K=10) | Recall | NDCG | HR |
|---|---|---|---|
| MF (= LightGCN k=0) | 0.196 | 0.182 | 0.635 |
| **LightGCN k=3** | **0.262** | **0.254** | **0.731** |
| LightGCN k=3 + content | 0.261 | 0.252 | 0.730 |

- **Message passing = the biggest single gain** → same table, same loss, + neighbour averaging: 0.196 → 0.262 (+34%). k=3 still improving at the 60-epoch cap.
- **Layers** (subset): k=1 already gets most of it (0.176 → 0.214), k=2/3 indistinguishable (0.217 / 0.209). Kept 3 (paper default); honest claim = "≥1 hop matters".
- **Propagation regularises** → MF peaks early and overfits, propagated models keep improving.
- **KG content: helps the KGE (+0.013), not LightGCN on warm anime** (full: 0.261 vs 0.262, best epoch 55 vs 60 → confirmed; subset: recall 0.209 → 0.213, NDCG 0.204 → 0.200, at 4.5× nodes / 2.3× time). LightGCN already links anime via shared users (anime–user–anime = 2 hops).
- **Content's real value = coverage** → 16,659 anime nobody in the subset rated still get embeddings from genre/studio neighbours; their nearest rated anime share genres 3.2× more than random (Jaccard 0.361 vs 0.112).

---

## Stage: evolution  `[LO8]`

No Apply script — studied on a toy + the real DistMult in `Learn/04_evolution/evolution_trials.ipynb`.
- **Completion is evaluated filtered** (raw ranking wastes 65% of top-10 on known facts).
- **New node** (genre-only anime, no retraining): MF has no row and can't learn one → scores it halfway for everyone. LightGCN + content places it from one genre edge (fans 0.72 vs non-fans 0.27 on a disliked→liked scale; cosine 1.00 to its genre's centroid).
- **Deleted fact**: MF keeps it exactly (rank 8.5 → 8.5); LightGCN drops it at once (9.4 → 16.0) but its layer-0 rows still remember (9.2).
- → strategy: KG content + propagation for additions, scheduled retrain for deletions/drift.

---

## Stage: logic  `[LO2, LO6, LO8]`

[`src/logic_queries.py`](src/logic_queries.py) → five rules/queries on a real slice (444,562 triples = all content + 201 users' ratings), output `data/generated/results/logic.md`.
- **OWL-RL typing** creates `rdf:type` edges nothing in the data asserts (22 watchers, 22 ratings, 12 orgs on a Bebop slice; 185 → 714 triples).
- **`likes` rule** (CONSTRUCT, rating ≥ 8) → 12,611 edges from 29,708 ratings = the flat-file collapse, written as an inspectable rule.
- **Recursive co-work path** from Sunrise: 188 direct collaborators, 2,225 / 2,550 orgs (87%) reachable → one connected industry component = the paths embeddings/GNNs use.
- **Explanation query** for DistMult's top pick (One Piece Film: Strong World, a TP): shared Action / Manga / Fantasy / Adventure / Shounen / Toei with the user's liked anime → score from ML, reasons from the graph.
- **Integrity constraints** → 0 violations; but writing them caught a trap: `?a a :Anime` matches **nothing** on raw data (anime are typed only by entailment via domain axioms) → "anime without genre" read 0 instead of **5,925**. Queries over un-materialised data silently miss what reasoning adds → run constraints after closure or phrase them on asserted predicates.

---

## Stage: evolution on disk + the service  `[LO8, LO6, LO9]`

[`src/evolve_kg.py`](src/evolve_kg.py) → the KG actually changes; certain and uncertain knowledge in separate files (`data/generated/evolution/`, summary `results/evolution.md`).
- **Logic completion (certain), `inferred.ttl`:** RDFS domain/range typing + subclass + `likes` (rating ≥ 8) as SPARQL rules over the FULL abox → **+1,117,044** unique triples (1.93M rule firings; the same type is often derived twice). Run in 11 partitions (content, then 500 users each) → exact, because each rule reads one triple or one rating node, never across partitions. 4 min on rdflib, where a full OWL-RL closure was infeasible → partitioning local rules = the scalable-reasoning trick.
- **ML completion (scored), `predicted.ttl`:** LightGCN top-10 unseen anime per watcher as **reified `:Prediction`** nodes (watcher, anime, score, rank, model, status) → 50,000 predictions / 350,000 triples. n-ary again → reified again.
- **Update:** the 55,305 held-out ratings replayed as arriving facts → **7,965** predictions `open → confirmed` (15.9%); 3,460 / 4,732 test watchers got ≥ 1 = HR@10 0.731, same number from a different angle.

[`src/recommend.py`](src/recommend.py) → the proposal's "around my taste, with tuned preferences": ML score (history) z-scored, + `--boost` per preferred tag found by SPARQL, `--avoid` tags removed; `--liked "A,B,C"` = a brand-new user as the mean of those anime's LightGCN vectors (no retraining). Examples in `results/recommend_examples.md`:
- user_66835 `--prefer Romance,Comedy --avoid Hentai` → preferred tags 30% → 100% of top-10, Hentai 30% → 0%; picks come from ML ranks 91–212 → boost 1.0 × 2 tags overrides history, it doesn't nudge it.
- new user liking Cowboy Bebop, Samurai Champloo, Trigun `--prefer SciFi --boost 0.5` → Death Note, Code Geass, Gurren Lagann, Evangelion …; Sci-Fi 40% → 60% → a gentle nudge. `--boost` = how much stated taste may override history.

[`src/web/`](src/web/) → **Aninext**, the same service as a web page (GitHub Pages, `docs/`). Pages serves static files only, so the heavy parts are exported once by `export_web.py`: LightGCN's propagated vectors (8,001 anime + 5,000 users, float32 .bin), each user's known anime, and each anime's genre/theme/demographic tags from SPARQL (25 MB for all six models; the page fetches one model, 3–6 MB, when it is picked). The browser runs only the cheap part, the re-rank from `recommend.rerank` (score, z-score, + boost per preferred tag, avoid → −∞). All six scorers can be switched in the page (popularity, MF, LightGCN, LightGCN + KG, DistMult, TransE), because each reduces to one of three kinds: dot product (LightGCN; DistMult with q = user ⊙ r_rated), −distance (TransE, q = user + r_rated), or a fixed count (popularity). A new user's q is the mean of the liked anime's vectors: LightGCN's own propagation rule, and for DistMult/TransE the point that scores those anime highest. That part is a heuristic, not something those models were trained for. Why offer the weaker models: their Recall@10 is lower, but the lists can still be judged by eye (e.g. MF on user_66835 returns a run of Naruto specials). The export self-checks a NumPy mirror of the JS against the CLI, and a headless browser run of both README examples gives the same top-10 and ML ranks as `recommend.py`. `serve.py` = local preview with live reload.
