"""Tests for extraction parsing/filtering (LLM mocked — no Ollama needed)."""

import extract
from extract import _filter_entity, _parse_llm_json, extract_from_chunk
from ingest import Chunk


def test_parse_llm_json_plain():
    assert _parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_llm_json_strips_code_fence():
    raw = '```json\n{"entities": []}\n```'
    assert _parse_llm_json(raw) == {"entities": []}


def test_parse_llm_json_extracts_embedded_object():
    raw = 'Sure! Here is the result: {"x": 2} hope that helps'
    assert _parse_llm_json(raw) == {"x": 2}


def test_parse_llm_json_garbage_returns_empty():
    assert _parse_llm_json("not json at all") == {}


def test_filter_entity_rules():
    assert _filter_entity("BADM 557", 0.9) is True
    assert _filter_entity("x", 0.9) is False          # too short
    assert _filter_entity("Thing", 0.1) is False       # below confidence floor
    assert _filter_entity("the system", 0.9) is False  # stop entity


def test_extract_from_chunk_with_mocked_llm(monkeypatch):
    """extract_from_chunk should turn an LLM JSON reply into typed objects."""
    fake_reply = {
        "message": {
            "content": (
                '{"entities": ['
                '{"name": "BADM 557", "type": "course", "description": "BI", "confidence": 0.95},'
                '{"name": "it", "type": "concept", "confidence": 0.95}'  # stop word, dropped
                '], "relationships": ['
                '{"source": "BADM 557", "target": "Power BI", "relation": "uses", "confidence": 0.9}'
                ']}'
            )
        }
    }

    class FakeOllama:
        @staticmethod
        def chat(*args, **kwargs):
            return fake_reply

    monkeypatch.setattr(extract, "ollama", FakeOllama)

    chunk = Chunk(text="BADM 557 uses Power BI.", source_file="s.md", section="intro")
    entities, relationships = extract_from_chunk(chunk, model="test-model")

    names = [e.name for e in entities]
    assert "BADM 557" in names
    assert "it" not in names  # stop entity filtered
    assert len(relationships) == 1
    assert relationships[0].relation == "uses"
    assert entities[0].source_file == "s.md"


def test_extract_from_chunk_handles_llm_error(monkeypatch):
    class BoomOllama:
        @staticmethod
        def chat(*args, **kwargs):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(extract, "ollama", BoomOllama)
    chunk = Chunk(text="x", source_file="s.md", section="intro")
    assert extract_from_chunk(chunk) == ([], [])
