"""Tests for P1 (grounding gate) and P2 (schema conformance)."""

import extract
from extract import (
    Relationship,
    conforms_to_schema,
    extract_from_chunk,
    is_grounded,
)
from ingest import Chunk


# ── P1: grounding ────────────────────────────────────────────────────

def test_is_grounded_exact_substring():
    text = "BADM 557 is a prerequisite for BADM 576 in the spring."
    assert is_grounded("is a prerequisite for", text) is True


def test_is_grounded_normalizes_whitespace_and_case():
    text = "FIN 550   is\ntaught  by Professor Xing."
    assert is_grounded("is taught by professor xing", text) is True


def test_is_grounded_rejects_absent_quote():
    text = "BADM 557 covers business intelligence."
    assert is_grounded("is taught by Mathias", text) is False


def test_is_grounded_rejects_too_short():
    assert is_grounded("uses", "the course uses Power BI heavily") is False


# ── P2: schema conformance ───────────────────────────────────────────

def test_conforms_valid_predicate_and_pattern():
    assert conforms_to_schema("taught_by", "course", "person") is True


def test_conforms_rejects_unknown_predicate():
    assert conforms_to_schema("invented_relation", "course", "person") is False


def test_conforms_rejects_type_pattern_violation():
    # taught_by expects (course, person); (tool, person) has no matching pattern.
    assert conforms_to_schema("taught_by", "tool", "person") is False


def test_conforms_unknown_types_not_penalized():
    # Predicate is allowed and at least one endpoint type is unknown.
    assert conforms_to_schema("uses", "", "") is True
    assert conforms_to_schema("related_to", "concept", "") is True


# ── Integration: extract_from_chunk wires both in ────────────────────

def _mock_ollama(reply_json):
    class FakeOllama:
        @staticmethod
        def chat(*args, **kwargs):
            return {"message": {"content": reply_json}}

    return FakeOllama


def test_extract_marks_grounded_and_conforming(monkeypatch):
    chunk = Chunk(
        text="BADM 557 is taught by Xing and uses Power BI in every lab.",
        source_file="s.md", section="intro",
    )
    reply = (
        '{"entities": ['
        '{"name": "BADM 557", "type": "course", "confidence": 0.95},'
        '{"name": "Xing", "type": "person", "confidence": 0.9},'
        '{"name": "Power BI", "type": "tool", "confidence": 0.9}'
        '], "relationships": ['
        '{"source": "BADM 557", "target": "Xing", "relation": "taught_by",'
        ' "quote": "BADM 557 is taught by Xing", "confidence": 0.9},'
        '{"source": "BADM 557", "target": "Power BI", "relation": "uses",'
        ' "quote": "this quote is not in the text", "confidence": 0.9}'
        ']}'
    )
    monkeypatch.setattr(extract, "ollama", _mock_ollama(reply))

    _, rels = extract_from_chunk(chunk, model="test")
    by_target = {r.target: r for r in rels}

    # taught_by: quote present in source -> grounded; (course,person) -> conforms
    assert by_target["Xing"].grounded is True
    assert by_target["Xing"].conforms is True
    assert by_target["Xing"].chunk_id == chunk.id

    # uses: quote NOT in source -> not grounded, but kept (default flag-only)
    assert by_target["Power BI"].grounded is False
    assert by_target["Power BI"].conforms is True  # (course, uses, tool) allowed


def test_require_grounding_drops_ungrounded(monkeypatch):
    monkeypatch.setattr(extract.exp, "REQUIRE_GROUNDING", True)
    chunk = Chunk(text="BADM 557 covers analytics.", source_file="s.md", section="i")
    reply = (
        '{"entities": [{"name": "BADM 557", "type": "course", "confidence": 0.9}],'
        ' "relationships": ['
        '{"source": "BADM 557", "target": "Mathias", "relation": "taught_by",'
        ' "quote": "fabricated unsupported span", "confidence": 0.9}'
        ']}'
    )
    monkeypatch.setattr(extract, "ollama", _mock_ollama(reply))
    _, rels = extract_from_chunk(chunk, model="test")
    assert rels == []  # ungrounded triple dropped under REQUIRE_GROUNDING


def test_strict_schema_drops_nonconforming(monkeypatch):
    monkeypatch.setattr(extract.exp, "STRICT_SCHEMA", True)
    chunk = Chunk(text="BADM 557 frobnicates the widget.", source_file="s.md", section="i")
    reply = (
        '{"entities": [{"name": "BADM 557", "type": "course", "confidence": 0.9}],'
        ' "relationships": ['
        '{"source": "BADM 557", "target": "widget", "relation": "frobnicates",'
        ' "quote": "BADM 557 frobnicates the widget", "confidence": 0.9}'
        ']}'
    )
    monkeypatch.setattr(extract, "ollama", _mock_ollama(reply))
    _, rels = extract_from_chunk(chunk, model="test")
    assert rels == []  # unknown predicate dropped under STRICT_SCHEMA


# ── Propagation through the graph ────────────────────────────────────

def test_graph_carries_grounding_and_stats(sample_graph):
    from graph import graph_stats

    assert sample_graph.edges["BADM 557", "Xing"]["grounded"] is True
    assert sample_graph.edges["BADM 557", "Xing"]["quote"]
    assert sample_graph.edges["BADM 557", "Power BI"]["grounded"] is False

    stats = graph_stats(sample_graph)
    # 2 of 3 sample relationships are grounded.
    assert stats["grounded_relationships"] == 2
    assert 0.0 < stats["grounding_rate"] < 1.0
    assert stats["conforming_relationships"] == sample_graph.number_of_edges()
    assert stats["conformance_rate"] == 1.0
