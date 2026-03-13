#!/usr/bin/env python3
"""
Interactive knowledge graph explorer.

This is the external memory access layer — how agents (or humans)
query the knowledge graph built by build_kg.py.
"""

import argparse
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

from graph import load_graph, graph_stats
from query import find_entity, get_neighbors, find_path

console = Console()


def show_stats(G):
    stats = graph_stats(G)
    table = Table(title="Knowledge Graph Overview")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Entities", str(stats["total_entities"]))
    table.add_row("Relationships", str(stats["total_relationships"]))
    table.add_row("Components", str(stats["connected_components"]))
    console.print(table)

    if stats["top_connected"]:
        console.print("\n[bold]Most connected:[/bold]")
        for name, deg in stats["top_connected"][:5]:
            console.print(f"  {name} ({deg} connections)")


def show_entity(G, name):
    result = get_neighbors(G, name)
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        return

    node = result
    console.print(Panel(
        f"[bold]{node['entity']}[/bold] ({node['type']})\n{node['description'] or 'No description'}",
        title="Entity",
    ))

    if node["outgoing"]:
        tree = Tree(f"[bold]{node['entity']}[/bold] →")
        for rel in node["outgoing"]:
            tree.add(f"[cyan]{rel['relation']}[/cyan] → {rel['entity']}")
        console.print(tree)

    if node["incoming"]:
        tree = Tree(f"→ [bold]{node['entity']}[/bold]")
        for rel in node["incoming"]:
            tree.add(f"{rel['entity']} → [cyan]{rel['relation']}[/cyan]")
        console.print(tree)


def show_path(G, source, target):
    steps = find_path(G, source, target)
    if steps and "error" in steps[0]:
        console.print(f"[red]{steps[0]['error']}[/red]")
        return

    path_str = ""
    for step in steps:
        if path_str:
            path_str += " → "
        path_str += f"[bold]{step['from']}[/bold] --[cyan]{step['relation']}[/cyan]--> [bold]{step['to']}[/bold]"
    console.print(Panel(path_str, title="Path"))


def show_search(G, query):
    results = find_entity(G, query)
    if not results:
        console.print(f"[yellow]No entities matching '{query}'[/yellow]")
        return

    table = Table(title=f"Search: {query}")
    table.add_column("Entity", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Mentions", style="green")
    table.add_column("Description")
    for name, attrs in results[:15]:
        table.add_row(
            name,
            attrs.get("type", "?"),
            str(attrs.get("mention_count", 0)),
            (attrs.get("description", "") or "")[:60],
        )
    console.print(table)


def interactive(G):
    console.print(Panel.fit(
        "[bold]Knowledge Graph Explorer[/bold]\n\n"
        "Commands:\n"
        "  [cyan]search <query>[/cyan]  — Find entities\n"
        "  [cyan]show <entity>[/cyan]   — Show entity details and connections\n"
        "  [cyan]path <a> -> <b>[/cyan] — Find path between entities\n"
        "  [cyan]stats[/cyan]           — Show graph statistics\n"
        "  [cyan]quit[/cyan]            — Exit",
        title="local-kg explorer",
    ))

    while True:
        try:
            cmd = console.input("\n[bold green]kg>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue
        elif cmd in ("quit", "exit", "q"):
            break
        elif cmd == "stats":
            show_stats(G)
        elif cmd.startswith("search "):
            show_search(G, cmd[7:])
        elif cmd.startswith("show "):
            show_entity(G, cmd[5:])
        elif "->" in cmd and cmd.startswith("path "):
            parts = cmd[5:].split("->")
            if len(parts) == 2:
                show_path(G, parts[0].strip(), parts[1].strip())
        else:
            console.print("[yellow]Unknown command. Try: search, show, path, stats, quit[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Explore a local knowledge graph")
    parser.add_argument("graph_file", help="Path to knowledge_graph.json")
    parser.add_argument("--stats", action="store_true", help="Show stats and exit")
    parser.add_argument("--search", help="Search for an entity")
    parser.add_argument("--show", help="Show entity details")
    args = parser.parse_args()

    G = load_graph(args.graph_file)
    console.print(f"Loaded graph: {G.number_of_nodes()} entities, {G.number_of_edges()} relationships")

    if args.stats:
        show_stats(G)
    elif args.search:
        show_search(G, args.search)
    elif args.show:
        show_entity(G, args.show)
    else:
        interactive(G)


if __name__ == "__main__":
    main()
