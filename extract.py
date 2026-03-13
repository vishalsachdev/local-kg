"""
Entity and relationship extraction using a local LLM via Ollama.

This is the working memory → semantic memory transition.
The LLM processes each chunk and extracts structured entities and relationships.

All configuration (prompt, model, thresholds, stop entities) lives in
experiment.py — the single file the autoresearch agent iterates on.
Both build_kg.py and run_autoresearch.py share this config.
"""

import json
import re
from dataclasses import dataclass, field

try:
    import ollama
except ImportError:
    ollama = None

from ingest import Chunk
import experiment as exp


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


def _parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling common quirks."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        return {}


def _filter_entity(name: str, confidence: float) -> bool:
    """Return True if the entity should be kept."""
    if len(name) < exp.MIN_ENTITY_NAME_LENGTH:
        return False
    if confidence < exp.MIN_ENTITY_CONFIDENCE:
        return False
    if name.lower() in exp.STOP_ENTITIES:
        return False
    return True


def extract_from_chunk(
    chunk: Chunk,
    model: str | None = None,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from a single chunk using local LLM."""
    if ollama is None:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")

    model = model or exp.MODEL
    prompt = exp.EXTRACTION_PROMPT.format(text=chunk.text)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": exp.TEMPERATURE, "num_ctx": exp.NUM_CTX},
        )
        raw = response["message"]["content"]
    except Exception as e:
        # Connection refused, model not found, etc.
        print(f"  [warning] LLM error: {e}")
        return [], []

    data = _parse_llm_json(raw)

    entities = []
    for e in data.get("entities", []):
        name = e.get("name", "").strip()
        confidence = float(e.get("confidence", 0.8))
        if not _filter_entity(name, confidence):
            continue
        entities.append(Entity(
            name=name,
            entity_type=e.get("type", "concept"),
            description=e.get("description", ""),
            confidence=confidence,
            source_file=chunk.source_file,
            source_section=chunk.section,
        ))

    relationships = []
    for r in data.get("relationships", []):
        confidence = float(r.get("confidence", 0.8))
        if confidence < exp.MIN_RELATIONSHIP_CONFIDENCE:
            continue
        relationships.append(Relationship(
            source=r.get("source", "").strip(),
            target=r.get("target", "").strip(),
            relation=r.get("relation", "related_to"),
            description=r.get("description", ""),
            confidence=confidence,
            source_file=chunk.source_file,
        ))

    return entities, relationships


def extract_batch(
    chunks: list[Chunk],
    model: str | None = None,
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
