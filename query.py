"""
Knowledge graph query and exploration interface.

This is how agents (or humans) access the semantic memory layer.
Vashishta's point: the graph is external memory accessed on-demand,
enabling smaller models to punch above their weight.
"""

import networkx as nx
from graph import load_graph, normalize_name


def find_entity(G: nx.DiGraph, query: str) -> list[tuple[str, dict]]:
    """Find entities matching a query string."""
    query_norm = normalize_name(query)
    results = []
    for node, attrs in G.nodes(data=True):
        node_norm = normalize_name(node)
        aliases = [normalize_name(a) for a in attrs.get("aliases", [])]
        if query_norm in node_norm or any(query_norm in a for a in aliases):
            results.append((node, attrs))
    return sorted(results, key=lambda x: x[1].get("mention_count", 0), reverse=True)


def get_neighbors(G: nx.DiGraph, entity: str) -> dict:
    """Get all entities connected to the given entity."""
    if entity not in G:
        matches = find_entity(G, entity)
        if not matches:
            return {"error": f"Entity '{entity}' not found"}
        entity = matches[0][0]

    outgoing = []
    for _, target, attrs in G.out_edges(entity, data=True):
        outgoing.append({
            "entity": target,
            "relation": attrs.get("relation", "related_to"),
            "description": attrs.get("description", ""),
        })

    incoming = []
    for source, _, attrs in G.in_edges(entity, data=True):
        incoming.append({
            "entity": source,
            "relation": attrs.get("relation", "related_to"),
            "description": attrs.get("description", ""),
        })

    node_data = G.nodes[entity]
    return {
        "entity": entity,
        "type": node_data.get("type"),
        "description": node_data.get("description"),
        "sources": node_data.get("sources", []),
        "outgoing": outgoing,
        "incoming": incoming,
    }


def find_path(G: nx.DiGraph, source: str, target: str) -> list[dict]:
    """Find the shortest path between two entities."""
    # Resolve names
    for query, var_name in [(source, "source"), (target, "target")]:
        if query not in G:
            matches = find_entity(G, query)
            if not matches:
                return [{"error": f"Entity '{query}' not found"}]
            if var_name == "source":
                source = matches[0][0]
            else:
                target = matches[0][0]

    try:
        path = nx.shortest_path(G.to_undirected(), source, target)
    except nx.NetworkXNoPath:
        return [{"error": f"No path between '{source}' and '{target}'"}]

    steps = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if G.has_edge(a, b):
            edge = G.edges[a, b]
            steps.append({"from": a, "relation": edge.get("relation", "?"), "to": b})
        elif G.has_edge(b, a):
            edge = G.edges[b, a]
            steps.append({"from": b, "relation": edge.get("relation", "?"), "to": a})
    return steps


def get_cluster(G: nx.DiGraph, entity: str, depth: int = 2) -> nx.DiGraph:
    """Get a subgraph around an entity up to N hops deep."""
    if entity not in G:
        matches = find_entity(G, entity)
        if not matches:
            return nx.DiGraph()
        entity = matches[0][0]

    undirected = G.to_undirected()
    neighbors = set()
    current_layer = {entity}

    for _ in range(depth):
        next_layer = set()
        for node in current_layer:
            next_layer.update(undirected.neighbors(node))
        neighbors.update(current_layer)
        current_layer = next_layer - neighbors

    neighbors.update(current_layer)
    return G.subgraph(neighbors).copy()
