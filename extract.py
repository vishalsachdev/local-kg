"""
Entity and relationship extraction using a local LLM via Ollama.

This is the working memory → semantic memory transition.
The LLM processes each chunk and extracts structured entities and relationships.

Following Vashishta's principle: the LLM is called with precise instructions
about what to extract, what quality looks like, and how to handle ambiguity.
"""

import json
import re
from dataclasses import dataclass, field

try:
    import ollama
except ImportError:
    ollama = None

from ingest import Chunk


@dataclass
class Entity:
    name: str
    entity_type: str  # person, concept, tool, technique, organization
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.8
    source_file: str = ""
    source_section: str = ""


@dataclass
class Relationship:
    source: str
    target: str
    relation: str  # uses, enables, part_of, created_by, related_to, etc.
    description: str = ""
    confidence: float = 0.8
    source_file: str = ""


# This prompt follows Vashishta's "intern test": specific enough that
# a junior analyst could produce the expected output.
EXTRACTION_PROMPT = """You are an entity and relationship extractor. Given the text below,
extract ALL meaningful entities and their relationships.

ENTITY TYPES (pick one per entity):
- person: Named individuals
- concept: Ideas, methodologies, patterns, principles
- tool: Software, libraries, frameworks, hardware
- technique: Specific methods, algorithms, approaches
- organization: Companies, teams, communities

RELATIONSHIP TYPES (pick one per relationship):
- uses: X uses Y
- enables: X enables/supports Y
- part_of: X is part of Y
- created_by: X was created by Y
- related_to: X is related to Y (use when relationship is unclear)
- implements: X implements Y
- alternative_to: X is an alternative to Y
- depends_on: X depends on Y

RULES:
1. Extract SPECIFIC named entities, not generic terms like "the system" or "users"
2. Each entity needs a short description (1 sentence)
3. Each relationship needs source, target, and type
4. If entity type is unclear, use "concept" with confidence 0.5
5. If relationship is implied but not stated, use "related_to" with confidence 0.5
6. Merge obvious duplicates (e.g., "JS" and "JavaScript")

Return ONLY valid JSON in this exact format:
{
  "entities": [
    {"name": "Entity Name", "type": "concept", "description": "What it is", "confidence": 0.9}
  ],
  "relationships": [
    {"source": "Entity A", "target": "Entity B", "relation": "uses", "description": "How they relate", "confidence": 0.8}
  ]
}

TEXT TO ANALYZE:
---
{text}
---

Return ONLY the JSON. No markdown fences, no explanation."""


def extract_from_chunk(
    chunk: Chunk,
    model: str = "llama3.2",
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from a single chunk using local LLM."""
    if ollama is None:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")

    prompt = EXTRACTION_PROMPT.format(text=chunk.text)

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_ctx": 4096},
    )

    raw = response["message"]["content"]

    # Parse JSON from response, handling common LLM quirks
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return [], []
        else:
            return [], []

    entities = []
    for e in data.get("entities", []):
        entities.append(Entity(
            name=e.get("name", "").strip(),
            entity_type=e.get("type", "concept"),
            description=e.get("description", ""),
            confidence=float(e.get("confidence", 0.8)),
            source_file=chunk.source_file,
            source_section=chunk.section,
        ))

    relationships = []
    for r in data.get("relationships", []):
        relationships.append(Relationship(
            source=r.get("source", "").strip(),
            target=r.get("target", "").strip(),
            relation=r.get("relation", "related_to"),
            description=r.get("description", ""),
            confidence=float(r.get("confidence", 0.8)),
            source_file=chunk.source_file,
        ))

    return entities, relationships


def extract_batch(
    chunks: list[Chunk],
    model: str = "llama3.2",
    on_progress=None,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract from all chunks with progress tracking."""
    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        if on_progress:
            on_progress(i + 1, len(chunks), chunk.source_file)

        entities, relationships = extract_from_chunk(chunk, model=model)
        all_entities.extend(entities)
        all_relationships.extend(relationships)

    return all_entities, all_relationships
