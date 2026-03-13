"""
Knowledge graph evaluation harness (FIXED — do not modify).

This is the equivalent of Karpathy's prepare.py — the read-only scoring
function that the agent cannot touch. It provides objective metrics for
comparing one extraction run against another.

Metrics:
  - entity_count: Total unique entities extracted
  - relationship_count: Total unique relationships
  - density: relationships / entities (higher = more connected graph)
  - avg_confidence: Mean confidence across entities
  - component_ratio: 1 - (components / entities) (higher = more connected)
  - type_coverage: How many entity types are represented (out of 5)
  - relation_coverage: How many relationship types used (out of 8)

The composite score combines these into a single number (lower is better,
like val_bpb) so the autoresearch loop can compare experiments.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx


ENTITY_TYPES = {"person", "concept", "tool", "technique", "organization"}
RELATION_TYPES = {
    "uses", "enables", "part_of", "created_by",
    "related_to", "implements", "alternative_to", "depends_on",
}


def evaluate_graph(graph_path: str) -> dict:
    """Score a knowledge graph. Returns metrics dict with composite score."""
    data = json.loads(Path(graph_path).read_text())
    G = nx.node_link_graph(data)

    n_entities = G.number_of_nodes()
    n_relationships = G.number_of_edges()

    if n_entities == 0:
        return {
            "entity_count": 0,
            "relationship_count": 0,
            "density": 0.0,
            "avg_confidence": 0.0,
            "component_ratio": 0.0,
            "type_coverage": 0.0,
            "relation_coverage": 0.0,
            "composite_score": 999.0,  # worst possible
        }

    # Density: relationships per entity
    density = n_relationships / n_entities if n_entities > 0 else 0

    # Average confidence
    confidences = [
        attrs.get("confidence", 0.5)
        for _, attrs in G.nodes(data=True)
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Connectedness: fewer components = better
    n_components = nx.number_weakly_connected_components(G)
    component_ratio = 1.0 - (n_components / n_entities) if n_entities > 1 else 0

    # Type coverage: how many of the 5 entity types are used
    used_types = {attrs.get("type", "") for _, attrs in G.nodes(data=True)}
    type_coverage = len(used_types & ENTITY_TYPES) / len(ENTITY_TYPES)

    # Relation coverage: how many of the 8 relationship types are used
    used_relations = {attrs.get("relation", "") for _, _, attrs in G.edges(data=True)}
    relation_coverage = len(used_relations & RELATION_TYPES) / len(RELATION_TYPES)

    # Composite score (lower is better, like val_bpb)
    # Penalize: low density, low confidence, many components, poor coverage
    composite = (
        1.0
        - 0.25 * min(density / 3.0, 1.0)       # density up to 3.0 is good
        - 0.20 * avg_confidence                   # higher confidence = better
        - 0.20 * component_ratio                  # more connected = better
        - 0.15 * type_coverage                    # more types = better
        - 0.10 * relation_coverage                # more relation types = better
        - 0.10 * min(n_entities / 100.0, 1.0)    # more entities (up to 100) = better
    )

    return {
        "entity_count": n_entities,
        "relationship_count": n_relationships,
        "density": round(density, 4),
        "avg_confidence": round(avg_confidence, 4),
        "component_ratio": round(component_ratio, 4),
        "type_coverage": round(type_coverage, 4),
        "relation_coverage": round(relation_coverage, 4),
        "composite_score": round(max(composite, 0.0), 6),
    }


def format_results(metrics: dict) -> str:
    """Format metrics as a single-line TSV-friendly string."""
    return "\t".join(f"{k}={v}" for k, v in metrics.items())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <knowledge_graph.json>")
        sys.exit(1)

    metrics = evaluate_graph(sys.argv[1])
    print(format_results(metrics))
    print(f"\ncomposite_score: {metrics['composite_score']}")
