"""Tests for the fixed evaluation harness used by the autoresearch loop."""

from evaluate import evaluate_graph, format_results
from graph import save_graph


def test_evaluate_empty_graph(tmp_path):
    import networkx as nx

    path = tmp_path / "empty.json"
    save_graph(nx.DiGraph(), str(path))
    metrics = evaluate_graph(str(path))
    assert metrics["entity_count"] == 0
    assert metrics["composite_score"] == 999.0  # worst case


def test_evaluate_populated_graph(sample_graph, tmp_path):
    path = tmp_path / "kg.json"
    save_graph(sample_graph, str(path))
    metrics = evaluate_graph(str(path))

    assert metrics["entity_count"] == sample_graph.number_of_nodes()
    assert metrics["relationship_count"] == sample_graph.number_of_edges()
    assert 0.0 <= metrics["composite_score"] <= 1.0
    assert metrics["density"] > 0
    assert 0.0 <= metrics["type_coverage"] <= 1.0
    assert 0.0 <= metrics["relation_coverage"] <= 1.0


def test_lower_score_is_better(sample_graph, tmp_path):
    """A richer graph should score no worse (lower) than a sparse one."""
    rich_path = tmp_path / "rich.json"
    save_graph(sample_graph, str(rich_path))
    rich_score = evaluate_graph(str(rich_path))["composite_score"]

    import networkx as nx

    sparse = nx.DiGraph()
    sparse.add_node("Lonely", type="concept", confidence=0.5)
    sparse_path = tmp_path / "sparse.json"
    save_graph(sparse, str(sparse_path))
    sparse_score = evaluate_graph(str(sparse_path))["composite_score"]

    assert rich_score < sparse_score


def test_format_results_is_tsv_friendly(sample_graph, tmp_path):
    path = tmp_path / "kg.json"
    save_graph(sample_graph, str(path))
    line = format_results(evaluate_graph(str(path)))
    assert "\t" in line
    assert "composite_score=" in line
