"""
Experiment configuration — THE SINGLE FILE THE AGENT ITERATES ON.

This is the equivalent of Karpathy's train.py. The autoresearch agent
modifies ONLY this file to improve knowledge graph quality.

Everything is fair game:
- Extraction prompt wording and structure
- Chunk size and overlap
- Entity type definitions
- Relationship type definitions
- Post-processing rules
- Deduplication strategies
- Confidence thresholds
"""

# ── Chunking Parameters ─────────────────────────────────────────────
CHUNK_MAX_CHARS = 2000
CHUNK_OVERLAP = 200

# ── Model Configuration ─────────────────────────────────────────────
MODEL = "llama3.2"
TEMPERATURE = 0.1
NUM_CTX = 4096

# ── Entity Types ─────────────────────────────────────────────────────
ENTITY_TYPES = """ENTITY TYPES (pick one per entity):
- person: Named individuals
- concept: Ideas, methodologies, patterns, principles
- tool: Software, libraries, frameworks, hardware
- technique: Specific methods, algorithms, approaches
- organization: Companies, teams, communities"""

# ── Relationship Types ───────────────────────────────────────────────
RELATIONSHIP_TYPES = """RELATIONSHIP TYPES (pick one per relationship):
- uses: X uses Y
- enables: X enables/supports Y
- part_of: X is part of Y
- created_by: X was created by Y
- related_to: X is related to Y (use when relationship is unclear)
- implements: X implements Y
- alternative_to: X is an alternative to Y
- depends_on: X depends on Y"""

# ── Extraction Rules ─────────────────────────────────────────────────
EXTRACTION_RULES = """RULES:
1. Extract SPECIFIC named entities, not generic terms like "the system" or "users"
2. Each entity needs a short description (1 sentence)
3. Each relationship needs source, target, and type
4. If entity type is unclear, use "concept" with confidence 0.5
5. If relationship is implied but not stated, use "related_to" with confidence 0.5
6. Merge obvious duplicates (e.g., "JS" and "JavaScript")"""

# ── The Extraction Prompt ────────────────────────────────────────────
# This is the main lever. Rewrite this to improve extraction quality.
EXTRACTION_PROMPT = f"""You are an entity and relationship extractor. Given the text below,
extract ALL meaningful entities and their relationships.

{ENTITY_TYPES}

{RELATIONSHIP_TYPES}

{EXTRACTION_RULES}

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "Entity Name", "type": "concept", "description": "What it is", "confidence": 0.9}}
  ],
  "relationships": [
    {{"source": "Entity A", "target": "Entity B", "relation": "uses", "description": "How they relate", "confidence": 0.8}}
  ]
}}

TEXT TO ANALYZE:
---
{{text}}
---

Return ONLY the JSON. No markdown fences, no explanation."""

# ── Post-Processing ──────────────────────────────────────────────────
# Minimum confidence threshold to keep an entity
MIN_ENTITY_CONFIDENCE = 0.3

# Minimum confidence threshold to keep a relationship
MIN_RELATIONSHIP_CONFIDENCE = 0.3

# Minimum entity name length
MIN_ENTITY_NAME_LENGTH = 2

# Entities to always filter out (lowercase)
STOP_ENTITIES = {
    "the system", "the user", "it", "they", "this", "that",
    "the author", "the article", "the document", "the text",
}
