# Knowledge-Graph Construction: Best Practices & Improvement Roadmap

Research synthesis (literature 2024–2026) mapping current best practices for
**LLM-based knowledge-graph construction** onto this repo, with concrete,
prioritized recommendations. Assumes local LLM inference is available (Ollama)
for both extraction and adjudication, and that a human can verify a small
fraction of outputs.

> Sourcing note: most numeric figures below come from the cited papers/docs.
> A few came from search abstracts where the publisher blocked full-text
> fetching (ScienceDirect, IEEE) — those are marked *(abstract-only)* and the
> direction is trustworthy even if the exact magnitude should be re-verified
> before quoting.

---

## 1. The core reframing: shape ≠ quality

The current `evaluate.py` scores a graph purely on **structural** signals —
entity count, relationship density, connected components, type/relation
coverage. The literature is consistent that these measure the graph's *shape*,
not its *correctness*, and that optimizing them directly is a Goodhart trap:
rewarding density rewards adding low-value or spurious triples.

Recognized KG quality decomposes into content-level dimensions:

| Dimension | Question it answers | Cheap proxy we can compute |
|-----------|--------------------|----------------------------|
| **Accuracy** | Are the triples correct? | triple precision vs a small gold set; LLM-judge screen |
| **Faithfulness/Grounding** | Is each triple supported by the source text? | verbatim-quote substring check + NLI entailment |
| **Consistency / Conformance** | Do entities/relations obey the schema? | fraction using allowed types; relation-hallucination rate |
| **Conciseness** | Are duplicate entities merged? | duplication rate from embedding clustering |
| **Completeness** | Did we capture what's there? | recall vs gold set |
| **Timeliness** | How fresh is the source? | source-date tag |

*Sources:* Zaveri et al., "Quality Assessment for Linked Data: A Survey"
(18 dimensions / 69 metrics) —
<https://www.semantic-web-journal.net/content/quality-assessment-linked-data-survey>;
Text2KGBench (Precision/Recall/F1, Ontology Conformance, S/R/O hallucination) —
<https://arxiv.org/abs/2308.02357>; GraphEval (per-triple NLI grounding) —
<https://arxiv.org/pdf/2407.10793>; Goodhart's law —
<https://www.practical-devsecops.com/glossary/goodharts-law/>.

**Implication for this repo:** demote the structural numbers to a descriptive
"shape" panel and add a content-level scorecard. (This touches the autoresearch
"fixed harness" contract — see Roadmap §A.)

---

## 2. Extraction: constrain to a schema, ground every triple

**2.1 Schema-guided extraction is the single biggest precision lever.**
Constraining extraction to an allow-list of entity types *and* relation types —
ideally typed `(source_type, RELATION, target_type)` patterns — plus post-hoc
validation against that list, is reported to lift raw-LLM precision from ~91% to
~98.8% and cut hallucinated extractions ~35% (ODKE+, enterprise scale —
generalize cautiously to a 7–8B local model).
*Sources:* <https://arxiv.org/html/2509.04696v1>;
LangChain LLMGraphTransformer `(s_type, REL, o_type)` tuples & `strict_mode` —
<https://medium.com/data-science/building-knowledge-graphs-with-llm-graph-transformer-a91045c49b59>;
Neo4j GraphPruning — <https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html>.

- **Repo today:** `extract.py` filters entity *names* but does **not** validate
  relation predicates against `RELATIONSHIP_TYPES`, and there is no typed
  `(s_type, rel, o_type)` pattern check.
- **Change:** add allow-lists + a post-extraction validation pass that drops or
  flags out-of-vocabulary predicates and type-violating edges.

**2.2 Grounding is the highest-leverage, cheapest defense.**
Require the LLM to emit a **verbatim supporting quote** per triple, then
mechanically verify the quote is a substring of the source chunk — deterministic,
no second model needed. ~77% of valid quotes match exactly; ~20% are near-
paraphrases (tier the check: exact → normalized → embedding-similarity flag →
reject). This matters *more* for small local models, which hallucinate more.
*Sources:* "Show Your Work: Verbatim Evidence Requirements" —
<https://www.medrxiv.org/content/10.64898/2026.03.03.26346690v1.full.pdf>;
AEVS anchor-constrained grounded extraction w/ provenance —
<https://www.mdpi.com/2073-431X/15/3/178>;
MS VeriTrail (validate the *citation*, not just the claim) —
<https://www.microsoft.com/en-us/research/blog/veritrail-detecting-hallucination-and-tracing-provenance-in-multi-step-ai-workflows/>.

- **Repo today:** edges carry `source_file` but no chunk id, offsets, or
  supporting quote, and nothing verifies support.
- **Change:** extend the extraction schema with a `quote` per relationship;
  add a grounding gate in `extract.py`; persist `{chunk_id, quote, grounded}`
  on the edge.

**2.3 Don't trust the model's self-reported confidence.**
Verbalized confidence is systematically overconfident (ECE ~0.41 under noisy
retrieval). Prefer an **empirical** confidence from self-consistency: sample the
extraction N times (temp > 0), keep triples appearing in ≥k samples, and use the
vote frequency as the confidence.
*Sources:* <https://www.emergentmind.com/topics/verbalized-confidence-scores>;
self-consistency — <https://arxiv.org/html/2510.24476v1> (and its caveat,
<https://arxiv.org/pdf/2509.06870>).

- **Repo today:** thresholds gate on the LLM's stated `confidence` (`exp.MIN_*`).
- **Change (optional/heavier):** add an N-sample voting mode; derive confidence
  from agreement rather than the model's number.

**2.4 Other extraction settings (low-cost, well-supported).**
Joint entity+relation extraction in one guided pass; `temperature=0` and
JSON/structured output for reproducibility; one optional "gleaning" round
("what did you miss?", cap at 1); ~600–1200-token chunks with ~100 overlap
(smaller = higher recall, more calls). *Sources:* MS GraphRAG dataflow —
<https://microsoft.github.io/graphrag/index/default_dataflow/>; auto-tuning —
<https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/>.
The repo already uses `temperature=0.1` and section-aware chunking; gleaning and
structured-output enforcement are the open gaps.

---

## 3. Entity resolution: string normalization is too weak

`graph.py`'s dedup is `lower/strip/replace`. It cannot merge
"Power BI" ↔ "PowerBI" ↔ "Microsoft Power BI", cannot relate a code to its title
("BADM 557" ↔ "Business Intelligence"), and (because it keys only on the string)
can wrongly merge same-string/different-type entities.

Best practice is a **cost-tiered cascade**:

1. **Exact / alias match** (free) — check a canonical node's `aliases` first.
2. **Embedding "blocking"** — embed each entity once with a local model, take
   top-k nearest neighbors as the *only* candidate pairs (turns O(n²) into
   ~O(kn)). *Sources:* <https://arxiv.org/pdf/2404.14831>;
   <https://towardsdatascience.com/the-rise-of-semantic-entity-resolution/>.
3. **Hybrid score**, same-type gated — e.g. `0.7·embedding + 0.3·fuzzy`; the
   same-`type` gate stops "Apple"(org) merging with "Apple"(concept).
   *Source:* <https://neo4j.com/labs/agent-memory/explanation/resolution-deduplication/>.
4. **LLM adjudication on the medium band only** — Ollama decides "same entity?"
   for borderline pairs; auto-merge above a high threshold, new-entity below a
   low one. Batch a cluster's candidates into one prompt.
5. **Cluster → canonical** — connected components over confirmed-match edges
   (Louvain fallback if a few weak bridges create "monster" clusters); keep all
   surface forms as `aliases`, store per-merge provenance
   `{method, score, reviewed_by}`.

*Thresholds (Neo4j Agent Memory):* domain-tune the auto-merge cutoff
(finance ~0.98, retail ~0.95, content ~0.92); start conservative, lower as
quality is observed; route the in-between band to human review (§5).

---

## 4. Provenance throughout (GraphRAG-shaped)

Every triple should reference the text unit(s) it came from
(`doc_id`, `chunk_id`, char offsets, supporting quote), exactly as GraphRAG
attaches a cited snippet to each assertion so a human can audit it. This makes
faithfulness auditable and powers citation in downstream RAG. The pipeline shape:
chunk → joint extract (+optional gleaning) → merge by `(name, type)` →
LLM-summarize merged descriptions → community detect → community reports.
*Sources:* <https://microsoft.github.io/graphrag/index/default_dataflow/>;
<https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/>.

The repo's OKF export is already a good provenance surface (`sources:` in
frontmatter); extending it with per-triple quotes would complete the chain.

---

## 5. Human-in-the-loop: verify a targeted minority

Don't review everything. Two signals decide what a human sees:

- **Uncertainty** — queue triples whose confidence is nearest the decision
  boundary (active learning / uncertainty sampling: max value per label).
  *Sources:* Prodigy active learning — <https://prodi.gy/docs/recipes>;
  uncertainty = distance from 0.5 — <https://arxiv.org/pdf/1612.03871>.
- **Structural impact** — entity merges, schema-violating triples, high-degree
  ("hub") nodes, and low-grounding triples; errors here cascade.
  *Sources:* CORE-KG — <https://arxiv.org/html/2510.26512>;
  KG validation w/ HITL (validation lifted precision >20% *(abstract-only)*) —
  <https://www.sciencedirect.com/science/article/pii/S030645732500086X>.

**UX:** export an accept/reject/edit queue to CSV/markdown (one row per item,
source chunk + quote + confidence + suggested decision), re-import to apply
decisions, and **persist a `verified` flag + content-hash** so re-runs don't
re-ask. *Source (file-based triage round-trip):* ExtracTable —
<https://arxiv.org/html/2506.03221>.

---

## 6. Evaluation: a content-level scorecard

Replace the single structural `composite_score` with a per-build scorecard, and
keep the structural numbers as a clearly-labeled descriptive panel:

| Metric | How to compute locally | Source |
|--------|------------------------|--------|
| **grounding_rate** | fraction of triples whose quote substring-matches the source (then optional NLI) | <https://arxiv.org/pdf/2407.10793> |
| **ontology_conformance** | fraction of entities/relations using allowed types | <https://arxiv.org/abs/2308.02357> |
| **relation_hallucination_rate** | predicates not in the ontology / total | <https://arxiv.org/abs/2308.02357> |
| **duplication_rate** | `1 − (#clusters / #raw entities)` via embedding clustering → conciseness | <https://towardsdatascience.com/entity-resolved-knowledge-graphs-6b22c09a1442/> |
| **triple P/R/F1** | match vs a small gold set (exact + cosine ≥0.8 fallback) | <https://arxiv.org/abs/2308.02357> |
| **LLM-judge screen** | per-triple "true? Y/N" — *cheap screen only*, gate facts on grounding | <https://arxiv.org/html/2411.17388v2>; limits: <https://arxiv.org/pdf/2412.05579> |

A small **gold set** is cheap: sample ~100–200 sentences, let Ollama propose
triples, a human accepts/rejects in a CSV; the accepted set is your reference
(measure inter-rater on a 20-item subset to estimate label noise).
*Source:* <https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1505877/full>.

> **Caveat — LLM-as-judge:** high recall on *consistent* cases (>95%) but poor
> recall on *inconsistent* ones (30–60%) and only low–moderate human correlation
> (ρ 0.27–0.46); it carries position/verbosity/self-enhancement biases. Use it to
> *flag*, never to silently approve; the deterministic grounding check is the
> robust gate. *Sources:* <https://eugeneyan.com/writing/llm-evaluators>;
> <https://arxiv.org/pdf/2603.29403>.

---

## 7. Prioritized roadmap (highest leverage first)

| # | Change | Effort | Why it ranks here |
|---|--------|--------|-------------------|
| **P1** | **Grounding gate** — `quote` per triple + deterministic substring check; persist `{chunk_id, quote, grounded}` on edges | S–M | Cheapest, most robust defense; deterministic; matters most for small local models |
| **P2** | **Schema conformance** — validate relation predicates + typed `(s_type,rel,o_type)` patterns; drop/flag violations | S | Big precision lever; small, self-contained |
| **P3** | **Content-level eval scorecard** — grounding_rate, conformance, relation-hallucination, duplication; structural → "shape" panel | M | Fixes the Goodhart-gameable score — **but changes the autoresearch "fixed harness" contract (decision needed)** |
| **P4** | **Entity-resolution cascade** — embedding blocking → hybrid score → LLM adjudication → cluster; aliases + provenance | L | Largest quality gain on messy corpora; needs a local embedding model |
| **P5** | **HITL review queue** — uncertainty + impact triage exported to CSV/md; `verified` flag persistence | M | Compounds P1–P4; turns the pipeline into a curation loop |
| **P6** | **Self-consistency confidence** — N-sample voting; empirical confidence | M | Replaces untrustworthy verbalized confidence; multiplies LLM cost |

**Architectural flag:** P3 modifies `evaluate.py`, which the repo currently
declares *fixed* (the autoresearch agent optimizes against it). Improving it is
the point — the current score is gameable — but it changes the experiment
contract, so it's a deliberate decision rather than a silent edit.

> **Decision (deferred implementation):** when P3 is built, the preferred
> approach is to keep `evaluate.py` as the single fixed harness and add the new
> content-level metrics *additively*, folding them into `composite_score` so the
> autoresearch target rewards correctness (grounding/conformance) rather than
> density alone. Implementation is deferred — this file is the roadmap.
