# local-kg: Local Knowledge Graph Builder

Build a knowledge graph from unstructured text in hours, not weeks — entirely
on your own hardware. No cloud APIs, no data leaves your machine.

Inspired by [Vin Vashishta's Local AI series](https://vinvashishta.substack.com/p/local-ai-the-5-hour-knowledge-graph)
and [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## The Approach

This project follows Vashishta's information-first framework:

1. **Define the workflow first** (see `WORKFLOW.md`) — "if you give the prompt
   to an intern, and they cannot deliver the expected outcome, AI will not either"
2. **Start with information structure**, not technology
3. **Use local models** via Ollama — smaller models work when augmented with
   structured knowledge
4. **Build agentic memory** across 5 layers (sensory, working, episodic,
   semantic, procedural + external)

## Quick Start

```bash
# 1. Install Ollama (https://ollama.com) and pull a model
ollama pull llama3.2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build a knowledge graph from your content
python build_kg.py /path/to/your/documents -m llama3.2

# 4. Explore the graph
python explore.py knowledge_graph.json
```

## How It Maps to Vashishta's Memory Framework

| Memory Layer | Component | Purpose |
|-------------|-----------|---------|
| Sensory | `ingest.py` | Raw files → parsed text chunks |
| Working | `extract.py` | LLM processes chunks in context window |
| Semantic | `graph.py` | Deduplicated entity-relationship graph |
| Episodic | `extraction_log.json` | What was processed, when, with what results |
| Procedural | `WORKFLOW.md` + prompts | Defines what to extract and how |
| External | `knowledge_graph.json` | Persistent graph queryable on demand |

## Pipeline

```
Documents → ingest.py → Chunks → extract.py → Entities/Rels → graph.py → Knowledge Graph
  (files)    (sensory)  (working)   (LLM)      (raw triples)   (semantic)   (external)
```

## Usage

### Build
```bash
# Basic — processes all .md, .html, .txt files in directory
python build_kg.py ./my-content

# Custom model and output
python build_kg.py ./my-content -m mistral -o my_graph.json
```

### Grounding & schema conformance (provenance)

Every extracted relationship is checked for two things, with the results stored
on the edge (and shown in the build summary, the explorer, and the OKF export):

- **Grounding** — the LLM must return a verbatim `quote` supporting each triple,
  which is then deterministically verified to appear in the source chunk. Edges
  carry `grounded` + the supporting `quote`. The explorer marks grounded edges
  with `✓`; OKF renders the quote as a blockquote.
- **Schema conformance** — the relation predicate must be in the allow-list, and
  the `(source_type, relation, target_type)` triple must match an allowed
  pattern. Non-conforming edges are flagged (`conforms`).

By default these **flag** rather than drop (safer for small local models). To
turn them into hard gates, flip the switches in `experiment.py`:

```python
REQUIRE_GROUNDING = True   # drop relationships whose quote isn't in the source
STRICT_SCHEMA     = True   # drop relationships that violate the schema
```

The rationale and citations are in [`RESEARCH.md`](RESEARCH.md) (P1, P2).

### Explore
```bash
# Interactive mode
python explore.py knowledge_graph.json

# Quick commands
python explore.py knowledge_graph.json --stats
python explore.py knowledge_graph.json --search "knowledge graph"
python explore.py knowledge_graph.json --show "Claude Code"
```

### Explorer Commands
- `search <query>` — Find entities by name
- `show <entity>` — See entity details and all connections
- `path <A> -> <B>` — Find shortest path between two entities
- `stats` — Graph overview and most connected entities

### Share (Open Knowledge Format)

The internal `knowledge_graph.json` is NetworkX-specific. To share the graph —
with a teammate, with another AI agent, or with your future self — export it to
an [**Open Knowledge Format (OKF)**](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing)
bundle: a directory of plain markdown files with YAML frontmatter, one file per
entity, with relationships expressed as ordinary markdown links.

```bash
# Export while building
python build_kg.py ./my-content --okf okf_bundle

# Or export an existing graph
python explore.py knowledge_graph.json --export-okf okf_bundle
python okf_export.py knowledge_graph.json -o okf_bundle
```

OKF is a *format, not a platform* — vendor-neutral, readable on GitHub, and
re-ingestable by any consumer (`okf_export.load_okf_bundle()` round-trips a
bundle back into a graph). It keeps the "no data leaves your machine" principle
while making the graph portable. Bundle layout:

```
okf_bundle/
├── index.md                     # stats + links to each entity type
└── entities/
    ├── course/
    │   ├── index.md             # progressive disclosure
    │   └── badm-557.md          # type, title, description + relationship links
    └── person/
        └── xing.md
```

## Development

```bash
# Install dev dependencies (runtime deps + pytest)
pip install -r requirements-dev.txt

# Run the test suite (no Ollama required — the LLM is mocked)
pytest
```

Tests cover ingestion, graph construction/dedup, querying, the fixed evaluation
harness, and OKF export/round-trip. CI runs them on every push and PR.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- Any model that fits your hardware (llama3.2 recommended for Apple Silicon)

## Autoresearch Mode (Karpathy's Loop)

The autoresearch integration applies Karpathy's autonomous experiment loop
to knowledge graph quality optimization. An AI agent iterates on `experiment.py`
— the only file it can modify — while `evaluate.py` provides fixed scoring.

**The mapping:**

| Karpathy's autoresearch | local-kg autoresearch |
|------------------------|----------------------|
| `train.py` (agent edits) | `experiment.py` (agent edits) |
| `prepare.py` (read-only) | `evaluate.py` (read-only) |
| `program.md` (human writes) | `program.md` (human writes) |
| val_bpb (lower = better) | composite_score (lower = better) |
| 5-min training runs | extraction runs on sample data |
| `results.tsv` | `results.tsv` |

**Usage:**

```bash
# Run a single experiment (agent calls this in each iteration)
python run_autoresearch.py ./articles --experiment --desc "tighter prompt"

# Evaluate an existing graph
python run_autoresearch.py --evaluate knowledge_graph.json

# View experiment history
python run_autoresearch.py --history
```

**The agent loop** (from `program.md`):
1. Read `experiment.py` and recent results
2. Hypothesize a change (prompt wording, chunk size, thresholds, etc.)
3. Modify `experiment.py` and commit
4. Run `python run_autoresearch.py <dir> --experiment`
5. If score improved → keep. If worse → revert.
6. NEVER STOP.

## Why Local?

Vashishta's argument: leadership will not risk uploading proprietary data to
frontier LLMs. Local AI keeps your competitive advantage — your knowledge —
entirely under your control. The knowledge graph is what makes smaller models
viable. They don't need to know everything; they query the right knowledge at
the right time.
