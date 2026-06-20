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
    # Provenance + quality flags (P1 grounding, P2 schema conformance)
    quote: str = ""          # verbatim supporting span copied from the source
    chunk_id: str = ""       # which chunk this triple came from
    grounded: bool = False   # quote verified to appear in the source chunk
    conforms: bool = True    # predicate + endpoint types obey the schema


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


def _normalize_for_grounding(s: str) -> str:
    """Collapse whitespace and lowercase, for tolerant substring matching."""
    return re.sub(r"\s+", " ", s).strip().lower()


def is_grounded(quote: str, source_text: str) -> bool:
    """
    Deterministic grounding check (P1): is ``quote`` a verbatim span of
    ``source_text``? Tiered: exact substring, then whitespace/case-normalized
    substring. No LLM call — cheap and robust, which matters most for the small
    local models that hallucinate the most.
    """
    quote = (quote or "").strip()
    if len(quote) < exp.MIN_QUOTE_LENGTH:
        return False
    if quote in source_text:
        return True
    return _normalize_for_grounding(quote) in _normalize_for_grounding(source_text)


def conforms_to_schema(relation: str, source_type: str, target_type: str) -> bool:
    """
    Schema conformance check (P2): the predicate must be in the allow-list, and
    if both endpoint types are known, the (source_type, relation, target_type)
    triple must match an allowed pattern ("*" is a wildcard). Endpoints with an
    unknown type are not penalized on the pattern check.
    """
    if relation not in exp.ALLOWED_RELATION_TYPES:
        return False
    if not source_type or not target_type:
        return True  # can't pattern-check without both endpoint types
    for s_pat, r_pat, t_pat in exp.ALLOWED_RELATION_PATTERNS:
        if r_pat != relation:
            continue
        if s_pat in ("*", source_type) and t_pat in ("*", target_type):
            return True
    return False


def extract_from_chunk(
    chunk: Chunk,
    model: str | None = None,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from a single chunk using local LLM."""
    if ollama is None:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")

    model = model or exp.MODEL
    # Use plain substitution rather than str.format(): the prompt necessarily
    # contains literal JSON braces ({...}) as a worked example, which would make
    # str.format() raise KeyError. The agent can edit the prompt freely without
    # worrying about escaping braces.
    prompt = exp.EXTRACTION_PROMPT.replace("{text}", chunk.text)

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

    # Map normalized entity name → type (for the schema pattern check below).
    def _norm(name: str) -> str:
        return name.lower().strip().replace("-", " ").replace("_", " ")

    type_by_name = {_norm(e.name): e.entity_type for e in entities}

    relationships = []
    for r in data.get("relationships", []):
        confidence = float(r.get("confidence", 0.8))
        if confidence < exp.MIN_RELATIONSHIP_CONFIDENCE:
            continue

        source = r.get("source", "").strip()
        target = r.get("target", "").strip()
        relation = r.get("relation", "related_to")
        quote = (r.get("quote", "") or "").strip()

        grounded = is_grounded(quote, chunk.text)
        conforms = conforms_to_schema(
            relation,
            type_by_name.get(_norm(source), ""),
            type_by_name.get(_norm(target), ""),
        )

        # Optional hard gates (default off — flag rather than silently drop).
        if exp.REQUIRE_GROUNDING and not grounded:
            continue
        if exp.STRICT_SCHEMA and not conforms:
            continue

        relationships.append(Relationship(
            source=source,
            target=target,
            relation=relation,
            description=r.get("description", ""),
            confidence=confidence,
            source_file=chunk.source_file,
            quote=quote,
            chunk_id=chunk.id,
            grounded=grounded,
            conforms=conforms,
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
