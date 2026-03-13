#!/usr/bin/env python3
"""
Local Knowledge Graph Builder — Main Entry Point

Replicates Vin Vashishta's 5-hour knowledge graph approach:
1. Start with information (workflow design in WORKFLOW.md)
2. Ingest unstructured text → chunks (sensory → working memory)
3. Extract entities + relationships via local LLM (working → semantic memory)
4. Build deduplicated knowledge graph (semantic memory)
5. Save and query (external memory)

Hardware: Runs on Apple Silicon Mac via Ollama. No cloud APIs, no data leaves your machine.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from ingest import ingest_directory
from extract import extract_batch
from graph import build_graph, save_graph, graph_stats

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from unstructured text using local AI",
        epilog="Following Vin Vashishta's information-first framework.",
    )
    parser.add_argument("input_dir", help="Directory containing documents to process")
    parser.add_argument(
        "-o", "--output",
        default="knowledge_graph.json",
        help="Output file for the knowledge graph (default: knowledge_graph.json)",
    )
    parser.add_argument(
        "-m", "--model",
        default="llama3.2",
        help="Ollama model to use (default: llama3.2)",
    )
    parser.add_argument(
        "--log",
        default="extraction_log.json",
        help="Extraction log file for episodic memory (default: extraction_log.json)",
    )
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]Local Knowledge Graph Builder[/bold]\n"
        "Information-first framework — no cloud APIs, no data leaves your machine.",
        title="local-kg",
    ))

    # ── Step 1: Ingest ──────────────────────────────────────────────
    console.print("\n[bold cyan]Step 1:[/bold cyan] Ingesting documents (sensory → working memory)")
    start = time.time()
    chunks = ingest_directory(args.input_dir)
    ingest_time = time.time() - start
    console.print(f"  Ingested [green]{len(chunks)}[/green] chunks in {ingest_time:.1f}s")

    if not chunks:
        console.print("[red]No content found. Check your input directory.[/red]")
        sys.exit(1)

    # Show file breakdown
    files = {}
    for c in chunks:
        files[c.source_file] = files.get(c.source_file, 0) + 1
    for f, count in sorted(files.items()):
        console.print(f"    {f}: {count} chunks")

    # ── Step 2: Extract ─────────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 2:[/bold cyan] Extracting entities via [yellow]{args.model}[/yellow] (working → semantic memory)")

    extraction_log = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(chunks))

        def on_progress(current, total, source_file):
            progress.update(task, completed=current, description=f"Processing {source_file}")
            extraction_log.append({
                "chunk": current,
                "total": total,
                "file": source_file,
                "timestamp": time.time(),
            })

        start = time.time()
        entities, relationships = extract_batch(chunks, model=args.model, on_progress=on_progress)
        extract_time = time.time() - start

    console.print(f"  Extracted [green]{len(entities)}[/green] entities and [green]{len(relationships)}[/green] relationships in {extract_time:.1f}s")

    # ── Step 3: Build Graph ─────────────────────────────────────────
    console.print("\n[bold cyan]Step 3:[/bold cyan] Building knowledge graph (semantic memory)")
    G = build_graph(entities, relationships)
    stats = graph_stats(G)

    # Display stats
    table = Table(title="Knowledge Graph Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Unique Entities", str(stats["total_entities"]))
    table.add_row("Relationships", str(stats["total_relationships"]))
    table.add_row("Connected Components", str(stats["connected_components"]))
    console.print(table)

    # Entity types
    type_table = Table(title="Entity Types")
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", style="green")
    for etype, count in sorted(stats["entity_types"].items(), key=lambda x: -x[1]):
        type_table.add_row(etype, str(count))
    console.print(type_table)

    # Top connected
    if stats["top_connected"]:
        top_table = Table(title="Most Connected Entities")
        top_table.add_column("Entity", style="cyan")
        top_table.add_column("Connections", style="green")
        for name, deg in stats["top_connected"][:10]:
            top_table.add_row(name, str(deg))
        console.print(top_table)

    # ── Step 4: Save ────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 4:[/bold cyan] Saving to {args.output}")
    save_graph(G, args.output)

    # Save episodic memory (extraction log)
    Path(args.log).write_text(json.dumps(extraction_log, indent=2))

    total_time = ingest_time + extract_time
    console.print(Panel.fit(
        f"[bold green]Done![/bold green] Knowledge graph saved to [cyan]{args.output}[/cyan]\n"
        f"Total time: {total_time:.1f}s | "
        f"{stats['total_entities']} entities | "
        f"{stats['total_relationships']} relationships\n\n"
        f"Next: Run [cyan]python explore.py {args.output}[/cyan] to query the graph.",
        title="Complete",
    ))


if __name__ == "__main__":
    main()
