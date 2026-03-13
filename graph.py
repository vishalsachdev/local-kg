"""
Knowledge graph construction and storage.

This is the semantic memory layer. Entities and relationships are deduplicated,
merged, and stored as a NetworkX graph that can be serialized to JSON.

Following Vashishta: the knowledge graph IS the competitive advantage.
It's why smaller models work — they don't need to know everything,
just query the right knowledge at the right time.
"""

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx

from extract import Entity, Relationship


def normalize_name(name: str) -> str:
    """Normalize entity names for deduplication."""
    return name.lower().strip().replace("-", " ").replace("_", " ")


def build_graph(
    entities: list[Entity],
    relationships: list[Relationship],
) -> nx.DiGraph:
    """
    Build a deduplicated knowledge graph from extracted entities and relationships.
    """
    G = nx.DiGraph()

    # Group entities by normalized name for deduplication
    entity_groups: dict[str, list[Entity]] = defaultdict(list)
    for e in entities:
        if not e.name or len(e.name) < 2:
            continue
        key = normalize_name(e.name)
        entity_groups[key].append(e)

    # Merge duplicates: pick highest confidence, collect all sources
    canonical_names: dict[str, str] = {}  # normalized → display name

    for norm_name, group in entity_groups.items():
        # Use the version with highest confidence as canonical
        best = max(group, key=lambda e: e.confidence)
        canonical_names[norm_name] = best.name

        sources = list({e.source_file for e in group if e.source_file})
        descriptions = list({e.description for e in group if e.description})
        aliases = list({e.name for e in group if normalize_name(e.name) == norm_name} - {best.name})

        G.add_node(best.name, **{
            "type": best.entity_type,
            "description": descriptions[0] if descriptions else "",
            "all_descriptions": descriptions,
            "confidence": max(e.confidence for e in group),
            "mention_count": len(group),
            "sources": sources,
            "aliases": aliases,
        })

    # Add relationships, mapping to canonical names
    for rel in relationships:
        src_norm = normalize_name(rel.source)
        tgt_norm = normalize_name(rel.target)

        src_name = canonical_names.get(src_norm)
        tgt_name = canonical_names.get(tgt_norm)

        # If source or target entity wasn't extracted, create it as low-confidence
        if not src_name:
            src_name = rel.source
            G.add_node(src_name, type="concept", confidence=0.4, mention_count=0, sources=[])
            canonical_names[src_norm] = src_name

        if not tgt_name:
            tgt_name = rel.target
            G.add_node(tgt_name, type="concept", confidence=0.4, mention_count=0, sources=[])
            canonical_names[tgt_norm] = tgt_name

        if G.has_edge(src_name, tgt_name):
            # Merge: keep highest confidence, collect descriptions
            existing = G.edges[src_name, tgt_name]
            existing["confidence"] = max(existing["confidence"], rel.confidence)
            existing["mention_count"] = existing.get("mention_count", 1) + 1
        else:
            G.add_edge(src_name, tgt_name, **{
                "relation": rel.relation,
                "description": rel.description,
                "confidence": rel.confidence,
                "mention_count": 1,
                "source_file": rel.source_file,
            })

    return G


def save_graph(G: nx.DiGraph, path: str):
    """Save the knowledge graph as JSON."""
    data = nx.node_link_data(G)
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def load_graph(path: str) -> nx.DiGraph:
    """Load a knowledge graph from JSON."""
    data = json.loads(Path(path).read_text())
    return nx.node_link_graph(data)


def graph_stats(G: nx.DiGraph) -> dict:
    """Return summary statistics about the knowledge graph."""
    type_counts = defaultdict(int)
    for _, attrs in G.nodes(data=True):
        type_counts[attrs.get("type", "unknown")] += 1

    relation_counts = defaultdict(int)
    for _, _, attrs in G.edges(data=True):
        relation_counts[attrs.get("relation", "unknown")] += 1

    # Find most connected entities
    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top_entities = degree_sorted[:10]

    return {
        "total_entities": G.number_of_nodes(),
        "total_relationships": G.number_of_edges(),
        "entity_types": dict(type_counts),
        "relationship_types": dict(relation_counts),
        "top_connected": [(name, deg) for name, deg in top_entities],
        "connected_components": nx.number_weakly_connected_components(G),
    }
