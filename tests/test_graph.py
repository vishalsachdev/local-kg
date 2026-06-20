"""Tests for graph construction, deduplication, and persistence."""

from graph import build_graph, graph_stats, load_graph, normalize_name, save_graph


def test_normalize_name():
    assert normalize_name("BADM-557") == "badm 557"
    assert normalize_name("Power_BI") == "power bi"


def test_build_graph_dedupes_entities(sample_graph):
    # "BADM 557" and "badm 557" normalize to the same node.
    assert "BADM 557" in sample_graph
    assert "badm 557" not in sample_graph
    node = sample_graph.nodes["BADM 557"]
    assert node["mention_count"] == 2  # two mentions merged
    assert node["confidence"] == 0.95  # highest-confidence wins as canonical


def test_build_graph_keeps_relationships(sample_graph):
    assert sample_graph.has_edge("BADM 557", "BADM 576")
    assert sample_graph.edges["BADM 557", "BADM 576"]["relation"] == "prerequisite_of"


def test_build_graph_creates_missing_endpoint_nodes():
    from extract import Relationship

    G = build_graph([], [Relationship("Ghost A", "Ghost B", "related_to")])
    assert "Ghost A" in G and "Ghost B" in G
    assert G.nodes["Ghost A"]["confidence"] == 0.4  # low-confidence placeholder


def test_save_load_round_trip(sample_graph, tmp_path):
    path = tmp_path / "kg.json"
    save_graph(sample_graph, str(path))
    loaded = load_graph(str(path))
    assert loaded.number_of_nodes() == sample_graph.number_of_nodes()
    assert loaded.number_of_edges() == sample_graph.number_of_edges()
    assert loaded.has_edge("BADM 557", "Xing")


def test_load_legacy_links_key(tmp_path):
    """Graphs written by older NetworkX used a 'links' key — still loadable."""
    import json

    legacy = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": "A", "type": "concept"}, {"id": "B", "type": "concept"}],
        "links": [{"source": "A", "target": "B", "relation": "uses"}],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    G = load_graph(str(path))
    assert G.has_edge("A", "B")


def test_graph_stats(sample_graph):
    stats = graph_stats(sample_graph)
    assert stats["total_entities"] == sample_graph.number_of_nodes()
    assert stats["total_relationships"] == sample_graph.number_of_edges()
    assert "course" in stats["entity_types"]
    assert stats["top_connected"][0][0] == "BADM 557"  # most connected
