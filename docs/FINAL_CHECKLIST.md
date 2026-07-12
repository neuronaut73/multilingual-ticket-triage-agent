# Final Checklist — Submission Readiness

Use this checklist before submitting or presenting the project.

---

## Tests

- [ ] `python -m pytest tests -m "not slow" -q` passes with no failures (723+ fast tests at time of writing; exact count may grow)
- [ ] `python -m pytest tests -q` passes or known slow/integration tests are noted

---

## Application Smoke Check

- [ ] `python cli.py` starts without errors and shows the six-option menu
- [ ] Option 1 → Analyzer only: one ticket triages successfully (requires Ollama)
- [ ] Option 1 → Conditional reviewer: one ticket triages with the reviewer enabled (requires Ollama + reviewer model)
- [ ] Option 4 → Curated evaluation: shows seven configured runs grouped by methodology
- [ ] Option 4 → Full experiment history: full run list remains accessible
- [ ] Option 5: leakage audit completes without violations
- [ ] Option 6: runtime configuration is displayed

---

## No-Leakage

- [ ] `answer` column is not used anywhere in the prediction pipeline
- [ ] Eval tickets are never inserted into LanceDB
- [ ] Current-ticket `actual_*` and `proxy_*` labels are accessed only after prediction for evaluation
- [ ] Historical labels from reference-split neighbor tickets are used only as retrieval metadata and aggregated evidence
- [ ] Prediction-time input is `subject` and `body` only
- [ ] Reference/eval split is deterministic and reproducible

---

## Documentation

- [ ] `README.md` matches the current implementation: pipeline, CLI, artifacts, DuckDB tables, prediction boundary
- [ ] `docs/technical_solution.md` describes the full pipeline including the conditional reviewer
- [ ] `docs/MODEL_EVALUATION.md` reports curated leaderboard metrics accurately
- [ ] `docs/PROJECT_ARCHITECTURE.md` is a concise current-state reference (no sprint plans, no obsolete tables)
- [ ] `docs/RUNBOOK.md` contains only verified commands; reviewer role is correctly distinguished from human_review_rate
- [ ] Docs use "evidence and decision trace" (not "reasoning trace" or "chain-of-thought")

---

## Evaluation Artifacts

- [ ] `outputs/curated_leaderboard.md` exists and shows seven runs
- [ ] The committed `outputs/triage_results.csv` and `outputs/triage_trace.jsonl` are documented as representative public samples
- [ ] Controlled 200-ticket evaluation runs are documented in `outputs/curated_leaderboard.md` and `docs/MODEL_EVALUATION.md`
- [ ] `outputs/triage_trace.jsonl` exists with one JSON line per processed ticket
- [ ] `outputs/run_summary.json` exists with `run_id`, `evaluation_metrics`, and timing
- [ ] DuckDB `triage_runs` table contains at least one completed run
- [ ] DuckDB `triage_metrics` table contains KPI rows
- [ ] DuckDB `triage_confusion_matrix` table contains confusion counts

---

## Presentation

- [ ] Presentation PDF opens correctly
- [ ] GitHub link tested in private/incognito browser
- [ ] Drive link has viewer access (if applicable)

---

## Public Snapshot — Included

- [ ] Representative outputs are present: `triage_results.csv`, `triage_trace.jsonl`, `run_summary.json`, `curated_leaderboard.csv`, `curated_leaderboard.md`

---

## Public Snapshot — Excluded

- [ ] Full raw dataset CSV is not tracked (`data/dataset-*.csv` excluded by `.gitignore`)
- [ ] DuckDB database is not tracked (`data/tickets.duckdb` excluded by `.gitignore`)
- [ ] LanceDB vector index is not tracked (`data/lancedb/` excluded by `.gitignore`)
- [ ] Python virtual environment is not tracked (`.venv/` excluded by `.gitignore`)
- [ ] Python caches are not tracked (`__pycache__/`, `.pytest_cache/`, `*.pyc` excluded)
- [ ] HuggingFace model cache is not tracked (`.cache/` excluded)
- [ ] ZIP archive is not tracked (`ticket_triage_agent.zip` excluded)
- [ ] Full 52-run snapshot files (`leaderboard_database_snapshot.*`) are excluded
- [ ] No credentials, API keys, or machine-specific paths in tracked files
- [ ] No confidential HDI invitation material in the repository
- [ ] No private email addresses in tracked files
