# Knowledge Graph Workflow Design

> "If you give the prompt to an intern, and they cannot deliver the expected outcome, AI will not either."
> — Vin Vashishta

## Step 0: Define the Work Before Building the Agent

Following Vashishta's framework, this document defines exactly what the
knowledge graph builder must do — before writing a line of code.

### What Is the Input?
- Markdown files (.md)
- HTML articles (.html)
- Plain text files (.txt)

### What Is the Output?
A structured knowledge graph stored as JSON with:
- **Entities**: Named concepts, people, tools, techniques, organizations
- **Relationships**: How entities connect (uses, enables, part_of, created_by, etc.)
- **Properties**: Metadata on each entity (type, description, source_file, confidence)

### What Does "Good" Look Like?
1. Every entity has a type (person, concept, tool, technique, organization)
2. Every relationship has a labeled edge with direction
3. Entities are deduplicated (e.g., "Claude Code" and "claude-code" merge)
4. Source provenance is tracked (which document, which section)
5. Confidence scores reflect extraction certainty

### What Should Be Ignored?
- Boilerplate (headers, footers, nav elements in HTML)
- Generic stop-entities ("the system", "the user", "it")
- Formatting artifacts

### How to Handle Ambiguity?
- When entity type is unclear → mark as "concept" with low confidence
- When relationship is implied but not stated → use "related_to" edge
- When same entity appears with different names → create alias list

## Agentic Memory Mapping (Vashishta's 5-Layer Model)

| Memory Type | Implementation |
|-------------|---------------|
| Sensory | Raw file bytes → parsed text chunks |
| Working | Current document context during extraction |
| Semantic | The knowledge graph itself (entities + relationships) |
| Episodic | Extraction log (what was processed, when, results) |
| Procedural | This workflow document + extraction prompts |
| External | JSON graph file on disk, queryable at any time |
