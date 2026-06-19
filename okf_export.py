#!/usr/bin/env python3
"""
Open Knowledge Format (OKF) export and import.

Turns the knowledge graph (NetworkX) into an OKF v0.1 bundle: a directory of
markdown files with YAML frontmatter, one file per entity ("concept"), with
relationships expressed as standard markdown links between files.

Why OKF? The internal `knowledge_graph.json` is NetworkX-specific. OKF is a
vendor-neutral, agent- and human-friendly format you can drop in a git repo,
read on GitHub, or feed to another LLM with no proprietary tooling. It makes
the graph *shareable* without giving up the "no data leaves your machine"
principle — it's a format, not a platform.

Spec: https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing

Bundle layout (directory structure implies the type taxonomy):

    okf_bundle/
    ├── index.md                 # root: stats + links to each type
    └── entities/
        ├── course/
        │   ├── index.md         # lists every course concept
        │   └── badm-557.md
        └── person/
            ├── index.md
            └── xing.md

Each entity file:

    ---
    type: course
    title: BADM 557
    description: Business Intelligence with AI, 4 credits
    tags: [course]
    timestamp: 2026-06-19T14:30:00Z
    confidence: 0.95
    mention_count: 3
    sources: [syllabus.md]
    aliases: []
    ---

    # BADM 557

    Business Intelligence with AI, 4 credits

    ## Relationships

    ### prerequisite_of
    - → [BADM 576](../course/badm-576.md)

    ## Referenced by

    - [FIN 550](../course/fin-550.md) — prerequisite_of
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import yaml

# Frontmatter keys that carry first-class OKF meaning. Everything else from the
# node is written as a custom field (OKF is "minimally opinionated": only `type`
# is required, producers may add their own fields).
_STANDARD_KEYS = ("type", "title", "description", "resource", "tags", "timestamp")


def slugify(name: str) -> str:
    """Turn an entity name into a filesystem- and URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "concept"


def _entity_type(attrs: dict) -> str:
    return (attrs.get("type") or "concept").strip() or "concept"


def _build_path_map(G: nx.DiGraph) -> dict[str, str]:
    """
    Map each node name → its relative path within the bundle
    (e.g. "entities/course/badm-557.md"), resolving slug collisions.
    """
    path_map: dict[str, str] = {}
    used: set[str] = set()
    # Sort for deterministic output regardless of insertion order.
    for name in sorted(G.nodes()):
        etype = slugify(_entity_type(G.nodes[name]))
        base = slugify(name)
        slug = base
        i = 2
        while f"{etype}/{slug}" in used:
            slug = f"{base}-{i}"
            i += 1
        used.add(f"{etype}/{slug}")
        path_map[name] = f"entities/{etype}/{slug}.md"
    return path_map


def _frontmatter(attrs: dict, name: str) -> dict:
    """Build an ordered OKF frontmatter dict for an entity node."""
    fm: dict = {"type": _entity_type(attrs)}
    fm["title"] = name
    desc = attrs.get("description") or ""
    if desc:
        fm["description"] = desc
    fm["tags"] = [_entity_type(attrs)]
    fm["timestamp"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Custom (producer-defined) fields carried over from the graph.
    for key in ("confidence", "mention_count", "sources", "aliases"):
        if key in attrs and attrs[key] not in (None, "", [], {}):
            fm[key] = attrs[key]
    return fm


def _render_entity(G: nx.DiGraph, name: str, path_map: dict[str, str]) -> str:
    """Render a single entity as an OKF markdown document."""
    attrs = G.nodes[name]
    fm = _frontmatter(attrs, name)

    lines: list[str] = ["---"]
    lines.append(yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).strip())
    lines.append("---\n")
    lines.append(f"# {name}\n")
    if fm.get("description"):
        lines.append(fm["description"] + "\n")

    self_path = path_map[name]

    def link(target: str) -> str:
        rel = _relative_link(self_path, path_map[target])
        return f"[{target}]({rel})"

    # Outgoing relationships, grouped by relation type.
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for _, tgt, eattrs in G.out_edges(name, data=True):
        if tgt in path_map:
            outgoing[eattrs.get("relation", "related_to")].append(
                (tgt, eattrs.get("description", ""))
            )

    if outgoing:
        lines.append("## Relationships\n")
        for relation in sorted(outgoing):
            lines.append(f"### {relation}\n")
            for tgt, edesc in sorted(outgoing[relation]):
                suffix = f" — {edesc}" if edesc else ""
                lines.append(f"- → {link(tgt)}{suffix}")
            lines.append("")

    # Incoming relationships (navigational; not re-parsed on import).
    incoming = [
        (src, eattrs.get("relation", "related_to"))
        for src, _, eattrs in G.in_edges(name, data=True)
        if src in path_map
    ]
    if incoming:
        lines.append("## Referenced by\n")
        for src, relation in sorted(incoming):
            lines.append(f"- {link(src)} — {relation}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _relative_link(from_path: str, to_path: str) -> str:
    """Relative markdown link from one bundle file to another."""
    import posixpath

    from_dir = posixpath.dirname(from_path)
    return posixpath.relpath(to_path, from_dir)


def _render_type_index(etype: str, names: list[str]) -> str:
    lines = [
        "---",
        f"type: index\ntitle: {etype}",
        "---\n",
        f"# {etype}\n",
        f"{len(names)} concept(s).\n",
    ]
    for name in sorted(names):
        lines.append(f"- [{name}]({slugify(name)}.md)")
    return "\n".join(lines).rstrip() + "\n"


def _render_root_index(G: nx.DiGraph, by_type: dict[str, list[str]]) -> str:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        "---",
        "type: index",
        "title: Knowledge Graph",
        f"timestamp: {ts}",
        "---\n",
        "# Knowledge Graph\n",
        "Open Knowledge Format (OKF) bundle exported from local-kg.\n",
        f"- **Entities:** {G.number_of_nodes()}",
        f"- **Relationships:** {G.number_of_edges()}",
        f"- **Types:** {len(by_type)}\n",
        "## Entity types\n",
    ]
    for etype in sorted(by_type):
        slug = slugify(etype)
        lines.append(
            f"- [{etype}](entities/{slug}/index.md) — {len(by_type[etype])} concept(s)"
        )
    return "\n".join(lines).rstrip() + "\n"


def export_okf(G: nx.DiGraph, out_dir: str) -> dict:
    """
    Write an OKF bundle for graph ``G`` into ``out_dir``.

    Returns a small summary dict (entity/file counts).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    path_map = _build_path_map(G)
    by_type: dict[str, list[str]] = defaultdict(list)
    for name in G.nodes():
        by_type[_entity_type(G.nodes[name])].append(name)

    files_written = 0
    for name in G.nodes():
        rel_path = path_map[name]
        dest = out / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_render_entity(G, name, path_map), encoding="utf-8")
        files_written += 1

    # Per-type index.md (progressive disclosure).
    for etype, names in by_type.items():
        type_dir = out / "entities" / slugify(etype)
        type_dir.mkdir(parents=True, exist_ok=True)
        (type_dir / "index.md").write_text(
            _render_type_index(etype, names), encoding="utf-8"
        )

    # Root index.md.
    (out / "index.md").write_text(_render_root_index(G, by_type), encoding="utf-8")

    return {
        "entities": G.number_of_nodes(),
        "relationships": G.number_of_edges(),
        "types": len(by_type),
        "files_written": files_written,
        "out_dir": str(out),
    }


# ── Consumer side: read an OKF bundle back into a graph ──────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file into (frontmatter_dict, body)."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm = yaml.safe_load(parts[1]) or {}
            return fm, parts[2]
    return {}, text


_REL_HEADER = re.compile(r"^###\s+(.+?)\s*$")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def load_okf_bundle(bundle_dir: str) -> nx.DiGraph:
    """
    Read an OKF bundle back into a NetworkX graph (producer/consumer
    independence — proves the bundle is self-describing). Re-parses only the
    "## Relationships" section so incoming "Referenced by" links don't duplicate
    edges.
    """
    root = Path(bundle_dir)
    G = nx.DiGraph()

    entity_files = [
        p for p in root.rglob("*.md")
        if p.name != "index.md" and p.name != "log.md"
    ]

    # First pass: nodes + path → name map for link resolution.
    path_to_name: dict[Path, str] = {}
    bodies: dict[str, tuple[Path, str]] = {}
    for path in entity_files:
        fm, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        name = fm.get("title") or path.stem
        attrs = {k: v for k, v in fm.items() if k not in ("title",)}
        G.add_node(name, **attrs)
        path_to_name[path.resolve()] = name
        bodies[name] = (path, body)

    # Second pass: edges from the Relationships section only.
    for name, (path, body) in bodies.items():
        in_rels = False
        current_relation = "related_to"
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                in_rels = stripped[3:].strip().lower() == "relationships"
                continue
            if not in_rels:
                continue
            header = _REL_HEADER.match(stripped)
            if header:
                current_relation = header.group(1).strip()
                continue
            link = _LINK.search(stripped)
            if link and stripped.startswith("-"):
                target_path = (path.parent / link.group(2)).resolve()
                target = path_to_name.get(target_path, link.group(1))
                desc = ""
                tail = stripped.split(")", 1)[-1]
                if "—" in tail:
                    desc = tail.split("—", 1)[1].strip()
                if not G.has_node(target):
                    G.add_node(target, type="concept")
                G.add_edge(name, target, relation=current_relation, description=desc)
    return G


def main():
    parser = argparse.ArgumentParser(
        description="Export a knowledge graph to an Open Knowledge Format (OKF) bundle.",
    )
    parser.add_argument("graph_file", help="Path to knowledge_graph.json")
    parser.add_argument(
        "-o", "--output", default="okf_bundle",
        help="Output directory for the OKF bundle (default: okf_bundle)",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.graph_file).read_text())
    # Tolerate both the modern "edges" key and the legacy "links" key.
    edges_key = "edges" if "edges" in data else "links"
    G = nx.node_link_graph(data, edges=edges_key)

    summary = export_okf(G, args.output)
    print(
        f"Exported OKF bundle to {summary['out_dir']}/ "
        f"({summary['entities']} entities, {summary['relationships']} relationships, "
        f"{summary['files_written']} files)"
    )


if __name__ == "__main__":
    main()
