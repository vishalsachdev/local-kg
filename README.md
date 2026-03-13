# local-kg: Local Knowledge Graph Builder

Build a knowledge graph from unstructured text in hours, not weeks — entirely
on your own hardware. No cloud APIs, no data leaves your machine.

Inspired by [Vin Vashishta's Local AI series](https://vinvashishta.substack.com/p/local-ai-the-5-hour-knowledge-graph).

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

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- Any model that fits your hardware (llama3.2 recommended for Apple Silicon)

## Why Local?

Vashishta's argument: leadership will not risk uploading proprietary data to
frontier LLMs. Local AI keeps your competitive advantage — your knowledge —
entirely under your control. The knowledge graph is what makes smaller models
viable. They don't need to know everything; they query the right knowledge at
the right time.
