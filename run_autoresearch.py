#!/usr/bin/env python3
"""
Autoresearch runner for knowledge graph optimization.

Integrates Karpathy's autoresearch loop with Vashishta's KG framework:
- Fixed evaluation harness (evaluate.py) — never modified
- Single experiment file (experiment.py) — the agent's canvas
- Autonomous loop: modify → run → score → keep/revert → repeat

Both this runner and build_kg.py share the same pipeline through
experiment.py → ingest.py → extract.py → graph.py. Improvements
discovered via autoresearch automatically apply to build_kg.py.

Usage:
  # Single run (used by the agent in each experiment iteration)
  python run_autoresearch.py ./articles --experiment

  # Evaluate an existing graph
  python run_autoresearch.py --evaluate knowledge_graph.json

  # Show experiment history
  python run_autoresearch.py --history
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

RESULTS_FILE = "results.tsv"
GRAPH_OUTPUT = "kg_experiment.json"


def run_experiment(input_dir: str) -> dict:
    """
    Run a single KG extraction experiment using current experiment.py config.
    Returns evaluation metrics.
    """
    # Force reimport to pick up agent's changes to experiment.py
    for mod in ("experiment", "extract", "ingest"):
        if mod in sys.modules:
            del sys.modules[mod]

    from ingest import ingest_directory
    from extract import extract_batch
    from graph import build_graph, save_graph
    from evaluate import evaluate_graph
    import experiment as exp

    console.print("[cyan]Running extraction experiment...[/cyan]")
    start = time.time()

    # Ingest uses experiment.py's chunk params directly (no double-chunking)
    chunks = ingest_directory(input_dir)
    console.print(f"  {len(chunks)} chunks (max_chars={exp.CHUNK_MAX_CHARS}, overlap={exp.CHUNK_OVERLAP})")

    # Extract using experiment.py's prompt, model, and filters (via extract.py)
    def on_progress(current, total, source_file):
        console.print(f"  Extracting chunk {current}/{total}: {source_file}")

    entities, relationships = extract_batch(chunks, on_progress=on_progress)

    elapsed = time.time() - start
    console.print(f"  Extracted {len(entities)} entities, {len(relationships)} relationships in {elapsed:.1f}s")

    # Build graph
    G = build_graph(entities, relationships)
    save_graph(G, GRAPH_OUTPUT)

    # Evaluate (using the FIXED harness)
    metrics = evaluate_graph(GRAPH_OUTPUT)
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["raw_entities"] = len(entities)
    metrics["raw_relationships"] = len(relationships)

    return metrics


def log_result(experiment_num: int, metrics: dict, description: str = ""):
    """Append experiment results to results.tsv."""
    results_path = Path(RESULTS_FILE)

    if not results_path.exists():
        header = "experiment\ttimestamp\tcomposite_score\tentity_count\trelationship_count\tdensity\tavg_confidence\tcomponent_ratio\ttype_coverage\trelation_coverage\telapsed_s\tdescription"
        results_path.write_text(header + "\n")

    row = (
        f"{experiment_num}\t"
        f"{datetime.now().isoformat()}\t"
        f"{metrics.get('composite_score', 999)}\t"
        f"{metrics.get('entity_count', 0)}\t"
        f"{metrics.get('relationship_count', 0)}\t"
        f"{metrics.get('density', 0)}\t"
        f"{metrics.get('avg_confidence', 0)}\t"
        f"{metrics.get('component_ratio', 0)}\t"
        f"{metrics.get('type_coverage', 0)}\t"
        f"{metrics.get('relation_coverage', 0)}\t"
        f"{metrics.get('elapsed_seconds', 0)}\t"
        f"{description}"
    )

    with open(results_path, "a") as f:
        f.write(row + "\n")


def show_history():
    """Display experiment history from results.tsv."""
    results_path = Path(RESULTS_FILE)
    if not results_path.exists():
        console.print("[yellow]No results.tsv found. Run an experiment first.[/yellow]")
        return

    lines = results_path.read_text().strip().split("\n")
    if len(lines) < 2:
        console.print("[yellow]No experiments logged yet.[/yellow]")
        return

    headers = lines[0].split("\t")
    table = Table(title="Experiment History")
    for h in headers:
        table.add_column(h, style="cyan" if h == "composite_score" else None)

    # Find best score for highlighting
    best_score = 999.0
    for line in lines[1:]:
        cols = line.split("\t")
        try:
            score = float(cols[2])
            best_score = min(best_score, score)
        except (IndexError, ValueError):
            pass

    for line in lines[1:]:
        cols = line.split("\t")
        try:
            score = float(cols[2])
            style = "bold green" if score == best_score else None
        except (IndexError, ValueError):
            style = None
        table.add_row(*cols, style=style)

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch runner for knowledge graph optimization",
    )
    parser.add_argument("input_dir", nargs="?", help="Directory with documents to process")
    parser.add_argument("--experiment", action="store_true", help="Run a single experiment")
    parser.add_argument("--evaluate", metavar="GRAPH", help="Evaluate an existing graph file")
    parser.add_argument("--history", action="store_true", help="Show experiment history")
    parser.add_argument("--desc", default="", help="Description for this experiment")
    args = parser.parse_args()

    if args.history:
        show_history()
        return

    if args.evaluate:
        from evaluate import evaluate_graph, format_results
        metrics = evaluate_graph(args.evaluate)
        console.print(format_results(metrics))
        console.print(f"\n[bold]composite_score: {metrics['composite_score']}[/bold]")
        return

    if args.experiment and args.input_dir:
        # Determine experiment number
        results_path = Path(RESULTS_FILE)
        if results_path.exists():
            n_lines = len(results_path.read_text().strip().split("\n")) - 1
            exp_num = n_lines + 1
        else:
            exp_num = 1

        console.print(Panel.fit(
            f"[bold]Experiment #{exp_num}[/bold]",
            title="autoresearch",
        ))

        metrics = run_experiment(args.input_dir)
        log_result(exp_num, metrics, description=args.desc)

        # Print result
        score = metrics.get("composite_score", 999)
        console.print(f"\n[bold]composite_score: {score}[/bold]")

        # Check against previous best
        if results_path.exists():
            lines = results_path.read_text().strip().split("\n")
            scores = []
            for line in lines[1:-1]:  # exclude header and current
                try:
                    scores.append(float(line.split("\t")[2]))
                except (IndexError, ValueError):
                    pass
            if scores:
                prev_best = min(scores)
                if score < prev_best:
                    console.print(f"[bold green]IMPROVED! {prev_best} → {score}[/bold green]")
                elif score > prev_best:
                    console.print(f"[bold red]WORSE. {prev_best} → {score} (consider reverting)[/bold red]")
                else:
                    console.print(f"[yellow]No change from previous best: {prev_best}[/yellow]")

        return

    parser.print_help()


if __name__ == "__main__":
    main()
