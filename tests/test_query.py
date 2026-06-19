"""Tests for the query / exploration layer."""

from query import find_entity, find_path, get_cluster, get_neighbors


def test_find_entity_partial_match(sample_graph):
    results = find_entity(sample_graph, "badm")
    names = [n for n, _ in results]
    assert "BADM 557" in names
    assert "BADM 576" in names


def test_find_entity_sorted_by_mentions(sample_graph):
    results = find_entity(sample_graph, "badm")
    # BADM 557 has 2 mentions, BADM 576 has 1 — 557 ranks first.
    assert results[0][0] == "BADM 557"


def test_get_neighbors(sample_graph):
    result = get_neighbors(sample_graph, "BADM 557")
    out_targets = {r["entity"] for r in result["outgoing"]}
    assert {"BADM 576", "Xing", "Power BI"} <= out_targets


def test_get_neighbors_fuzzy_resolves(sample_graph):
    # Lowercase query resolves to the canonical node.
    result = get_neighbors(sample_graph, "power bi")
    assert result["entity"] == "Power BI"


def test_get_neighbors_unknown(sample_graph):
    assert "error" in get_neighbors(sample_graph, "Nonexistent Thing")


def test_find_path(sample_graph):
    steps = find_path(sample_graph, "BADM 576", "Xing")
    assert steps and "error" not in steps[0]
    # Path runs BADM 576 — BADM 557 — Xing.
    entities_in_path = {s["from"] for s in steps} | {s["to"] for s in steps}
    assert "BADM 557" in entities_in_path


def test_find_path_no_path(sample_graph):
    sample_graph.add_node("Island", type="concept")
    steps = find_path(sample_graph, "BADM 557", "Island")
    assert "error" in steps[0]


def test_get_cluster(sample_graph):
    sub = get_cluster(sample_graph, "BADM 557", depth=1)
    assert "BADM 557" in sub
    assert sub.number_of_nodes() >= 3
