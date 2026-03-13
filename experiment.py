"""
Experiment configuration — THE SINGLE FILE THE AGENT ITERATES ON.

This is the equivalent of Karpathy's train.py. The autoresearch agent
modifies ONLY this file to improve knowledge graph quality.

Tailored for: msba-online (MSBA program curriculum design repo)
Content: course syllabi, program design docs, strategy docs, decision logs,
email threads, meeting transcripts, stakeholder discussions.
"""

# ── Chunking Parameters ─────────────────────────────────────────────
# Curriculum docs are structured with clear sections; keep chunks focused
CHUNK_MAX_CHARS = 1500
CHUNK_OVERLAP = 150

# ── Model Configuration ─────────────────────────────────────────────
MODEL = "llama3.2"
TEMPERATURE = 0.1
NUM_CTX = 4096

# ── Entity Types ─────────────────────────────────────────────────────
ENTITY_TYPES = """ENTITY TYPES (pick one per entity):
- person: Named faculty, staff, stakeholders (e.g., "Xing", "Mathias", "Amanda", "Amber")
- course: Academic courses with codes (e.g., "BADM 557", "FIN 550", "Agentic AI for Analytics")
- tool: Software, platforms, IDEs (e.g., "VS Code", "Copilot Pro", "Power BI", "Canvas LMS", "Colab")
- concept: Pedagogical methods, frameworks, competencies (e.g., "oral defense", "coupled integration", "L-C-E framework")
- organization: Universities, departments, accreditation bodies (e.g., "Gies College of Business", "AACSB", "Coursera")"""

# ── Relationship Types ───────────────────────────────────────────────
RELATIONSHIP_TYPES = """RELATIONSHIP TYPES (pick one per relationship):
- prerequisite_of: Course X is a prerequisite for course Y
- taught_by: Course X is taught by person Y
- uses: X uses tool/method Y
- part_of: X is part of program/semester Y
- covers: Course X covers competency/topic Y
- decided_by: Decision X was made by person Y
- depends_on: X depends on Y
- related_to: X is related to Y (use when relationship is unclear)"""

# ── Extraction Rules ─────────────────────────────────────────────────
EXTRACTION_RULES = """RULES:
1. Extract SPECIFIC named entities — course codes (BADM 557), people (first or full names), tools (Power BI), organizations (Gies)
2. Each entity needs a short description (1 sentence)
3. Preserve course codes exactly as written (e.g., "BADM 557" not "Business Intelligence")
4. For decisions, extract WHO decided WHAT and connect them
5. For email threads, extract the people involved and topics discussed
6. If entity type is unclear, use "concept" with confidence 0.5
7. If relationship is implied but not stated, use "related_to" with confidence 0.5"""

# ── The Extraction Prompt ────────────────────────────────────────────
# This is the main lever. Rewrite this to improve extraction quality.
EXTRACTION_PROMPT = f"""You are extracting entities and relationships from a university program
design document. This is for an MSBA (Master of Science in Business Analytics)
online program. Extract ALL meaningful entities and their relationships.

{ENTITY_TYPES}

{RELATIONSHIP_TYPES}

{EXTRACTION_RULES}

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "BADM 557", "type": "course", "description": "Business Intelligence with AI, 4 credits", "confidence": 0.95}}
  ],
  "relationships": [
    {{"source": "FIN 550", "target": "BADM 576", "relation": "prerequisite_of", "description": "ML I before ML II", "confidence": 0.9}}
  ]
}}

TEXT TO ANALYZE:
---
{{text}}
---

Return ONLY the JSON. No markdown fences, no explanation."""

# ── Post-Processing ──────────────────────────────────────────────────
MIN_ENTITY_CONFIDENCE = 0.3
MIN_RELATIONSHIP_CONFIDENCE = 0.3
MIN_ENTITY_NAME_LENGTH = 2

# Entities to always filter out (lowercase)
STOP_ENTITIES = {
    "the system", "the user", "it", "they", "this", "that",
    "the program", "the course", "the student", "the faculty",
    "the document", "the text", "the article",
    "n/a", "tbd", "none",
}
