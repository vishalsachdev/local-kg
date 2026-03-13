# Autoresearch: Knowledge Graph Quality Optimization

You are an autonomous research agent. Your goal is to improve knowledge graph
extraction quality by iterating on `experiment.py` — the ONLY file you may edit.

## Setup

1. Read these files to understand the system:
   - `README.md` — project overview
   - `WORKFLOW.md` — what quality looks like (Vashishta's framework)
   - `evaluate.py` — the scoring function (DO NOT MODIFY)
   - `experiment.py` — your canvas (the ONLY file you edit)
2. Verify sample data exists in the target directory
3. Check `results.tsv` for prior experiment history (if any)

## The Experiment Loop

Run this loop indefinitely. NEVER STOP.

1. **Examine** current state: read `experiment.py` and recent `results.tsv` entries
2. **Hypothesize**: pick ONE change to try. Think about what would improve the
   composite score. Ideas:
   - Rewrite the extraction prompt for clarity/specificity
   - Adjust chunk size (smaller chunks = more focused extraction)
   - Add/refine entity or relationship types
   - Tighten or loosen confidence thresholds
   - Add post-processing rules to filter noise
   - Improve deduplication logic
   - Change model temperature
3. **Modify** `experiment.py` with your change
4. **Commit** the change with a descriptive message
5. **Run** the experiment:
   ```bash
   python run_autoresearch.py <input_dir> --experiment
   ```
6. **Evaluate** the results — check the composite_score
7. **Decide**:
   - If composite_score IMPROVED → keep the commit, log to results.tsv
   - If composite_score got WORSE → `git reset --hard HEAD~1` to revert
8. **Repeat** from step 1

## Rules

- ONLY modify `experiment.py` — never touch `evaluate.py`, `ingest.py`,
  `graph.py`, `query.py`, or `prepare.py`
- Do NOT install new packages
- Do NOT modify the evaluation function
- Each experiment runs for a fixed time budget (set by the runner)
- Simpler solutions are preferred when scores are comparable
- Log EVERY experiment to `results.tsv`, including failures

## Metrics (lower composite_score is better)

The composite score penalizes:
- Low graph density (few relationships per entity)
- Low confidence scores
- Many disconnected components
- Poor entity type coverage
- Poor relationship type coverage
- Too few entities

## Strategy Tips

- Start with prompt engineering — it's the highest-leverage change
- Small, targeted changes are easier to evaluate than sweeping rewrites
- If stuck, try reducing chunk size to give the LLM more focused context
- Watch for entity explosion (too many low-quality entities)
- The stop-entity list is powerful for filtering noise
