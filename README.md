# Local Multilingual Ticket Triage Agent

A local Python prototype for insurance support ticket triage. The system classifies multilingual customer-support tickets into assignment topics, urgency levels, and next actions using retrieval-augmented historical evidence, a local LLM, deterministic validation, an optional conditional reviewer, and deterministic routing.

---

## Key Capabilities

- **Multilingual ingestion** — reads a multilingual support-ticket CSV dataset
- **DuckDB storage** — structured storage of tickets, labels, predictions, run tracking, and evaluation metrics
- **Ontology and proxy-label mapping** — bridges Kaggle labels to the assignment insurance triage schema
- **Multilingual embeddings** — `intfloat/multilingual-e5-large` via SentenceTransformers (1024-dim)
- **LanceDB vector search** — top-k cosine search over reference-split ticket embeddings
- **Weighted neighbor evidence** — weighted vote over retrieved neighbor labels for predicted queue and priority
- **Local Ollama LLM analyzer** — structured triage output (topic, urgency, confidence, missing info, note)
- **Pydantic validation and retry** — invalid LLM output triggers retry; persistent failure triggers fallback
- **Deterministic validator** — checks confidence, missing info, and LLM/kNN disagreement
- **Conditional reviewer** — optional second-pass LLM invoked only when the validator raises trigger flags
- **Deterministic router** — maps validator signals to a final `next_action` via business rules
- **Simulated action executor** — executes a simulated local action and records the result
- **Batch output and evaluation** — CSV, JSONL, and run metrics stored in DuckDB

---

## Pipeline

```text
preprocessing
→ reference-only retrieval (LanceDB vector index from reference split)
→ weighted neighbor evidence (queue, priority vote from top-k neighbors)
→ local structured triage analyzer (Ollama LLM → Pydantic LLMAnalysis)
→ deterministic validator (confidence, missing info, LLM/kNN agreement)
→ optional conditional reviewer (second-pass LLM when trigger flags fire)
→ deterministic router (business rules → next_action)
→ simulated action executor (records action result)
→ persistence and evaluation (CSV, JSONL, DuckDB metrics)
```

---

## Prediction-Time Boundary

**Prediction-time input is `subject` and `body` only.**

Current-ticket labels (`queue`, `priority`, `type`, tags, `actual_*`, and `proxy_*`) are never used during prediction. Historical labels belonging to reference-split neighbor tickets may be used as retrieval metadata and aggregated historical evidence. Labels belonging to the current eval ticket are accessed only after prediction for evaluation.

The `answer` column is never read, stored, or used at any point.

---

## No-Leakage Design

| Split | Role |
|---|---|
| **Reference** (~80%) | Indexed into LanceDB; used for neighbor retrieval and historical evidence |
| **Eval** (~20%) | Simulates new incoming tickets; **never** embedded into LanceDB |

The split is deterministic: based on a hash of `ticket_id`, reproducible without a random seed.

---

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start Ollama and pull models

```powershell
ollama serve
ollama pull llama3.2:3b
```

### 3. Download the dataset

```powershell
kaggle datasets download -d tobiasbueck/multilingual-customer-support-tickets -p data --unzip
```

---

## How to Run

### Interactive demo (interview)

```powershell
python cli.py
```

### Batch build / evaluation

```powershell
python main.py
```

On the first run, set `runtime.rebuild_duckdb: true` and `runtime.rebuild_lancedb: true` in `config.yaml`. After the index is built, set both back to `false`.

---

## CLI — Six Top-Level Options

```
0.  Exit
1.  Triage a ticket          (random eval ticket or specific ticket_id; reviewer OFF/ON per run)
2.  Run batch evaluation      (configurable limit; reviewer OFF/ON per run)
3.  Compare reviewer OFF vs ON
4.  Inspect evaluation results
      1. Curated evaluation   (seven configured runs grouped by methodology)
      2. Full experiment history
      3. Run details
      4. Confusion matrix
      5. Ticket prediction details
5.  Run leakage audit
6.  Show runtime configuration
```

---

## Output Artifacts

| File | Description |
|---|---|
| `outputs/triage_results.csv` | One row per processed eval ticket: topic, urgency, next action, confidence, flags |
| `outputs/triage_trace.jsonl` | Full per-ticket evidence and decision trace: neighbors, LLM output, validator signals, reviewer trace, action result |
| `outputs/run_summary.json` | Run-level summary: run_id, config, timing, evaluation metrics |
| `outputs/curated_leaderboard.csv` | Curated KPI table for the seven configured runs |
| `outputs/curated_leaderboard.md` | Human-readable curated leaderboard |

---

## DuckDB Tables

| Table | Contents |
|---|---|
| `historical_tickets` | All imported CSV rows with split assignment and label metadata |
| `triage_runs` | One row per batch run: run_id, model, config settings |
| `triage_predictions` | One row per ticket per run: topic, urgency, next action, confidence, flags |
| `triage_metrics` | One row per KPI per run |
| `triage_confusion_matrix` | Confusion counts per run |

---

## Evaluation Metrics

| Metric | Notes |
|---|---|
| `urgency_accuracy` / `urgency_macro_f1` | Strongest signal — direct 1:1 mapping from `priority` |
| `topic_proxy_accuracy` / `topic_macro_f1` | Weaker — queue-to-topic mapping is indirect |
| `next_action_proxy_agreement` | **Not true accuracy** — proxy action derived from proxy topic; no ground truth in dataset |
| `human_review_rate` | Operational safety metric |
| `avg_seconds_per_ticket` | Throughput |

---

## Curated Evaluation Summary

Seven runs are configured in `config.yaml` and displayed in the curated leaderboard.

**Controlled Granite reviewer A/B (llama3.2:3b, 200 tickets, seed 42 — reviewer OFF vs granite4.1:8b reviewer ON):**

Both runs used the same `llama3.2:3b` analyzer, the same balanced 200-ticket eval sample, seed 42, the same `intfloat/multilingual-e5-large` embedding model, the same top_k=5, and the same thresholds and evaluation logic. Only reviewer OFF versus `granite4.1:8b` reviewer ON differed.

| Configuration | Rev. Rate | Urgency Acc | Urgency F1 | Topic Acc | Topic F1 | Action Agr. | Human Rev. | Avg Conf. | sec/ticket |
|---|---|---|---|---|---|---|---|---|---|
| Reviewer OFF (`ab_off`) | 0.0% | 55.5% | 55.4% | 59.5% | 56.0% | 19.5% | 42.0% | 83.3% | 2.90 |
| granite4.1:8b ON (`ab_on`) | 21.0% | 57.0% | 56.0% | 60.0% | 56.0% | 23.0% | 38.0% | 84.9% | 3.60 |

The conditional Granite reviewer was invoked for 21% of tickets. It improved urgency accuracy, action proxy agreement, average confidence, and reduced the human-review rate. Topic macro F1 remained effectively unchanged. Average latency increased from 2.904 to 3.599 seconds per ticket, approximately 24%.

P95 latency increased from 2.917 to 6.099 seconds because reviewed tickets require an additional model call. The reviewer is therefore beneficial selectively, not free. Results support conditional invocation instead of reviewing every ticket.

**Earlier best observed Granite reviewer run (different sample):**
`run_20260711_200952` — urgency 57.0%/55.7%, topic 61.5%/58.5%, action 24.0%, human review 37.0%, 3.55 sec/ticket. This run used a different ticket sample and is retained as historical context, not as the controlled A/B result.

**Historical analyzer screening:** `qwen3-coder:30b`, `devstral-small-2:24b`, `deepseek-r1:14b`, and `llama3.2:3b` on 200 tickets without reviewer. See full leaderboard in the CLI or `outputs/curated_leaderboard.md`.

---

## Tests

```powershell
# Fast tests — no Ollama, no LanceDB, no DuckDB required
python -m pytest tests -m "not slow" -q

# All tests
python -m pytest tests -q
```

---

## Project Structure

```text
config.yaml                    — runtime configuration
main.py                        — batch entry point
cli.py                         — interactive demo entry point
ontology/ticket_ontology.yaml  — assignment triage ontology and proxy mapping

src/
  domain/         — enums, Pydantic models, ontology loader, proxy mapping
  application/    — agent, analyzer, reviewer, validator, router, executor, batch runner, metrics, CLI menu
  infrastructure/ — DuckDB repository, LanceDB store, embedding model, LLM client, CSV/JSONL I/O

outputs/
  triage_results.csv          — representative batch output (included)
  triage_trace.jsonl          — representative trace (included)
  run_summary.json            — representative run summary (included)
  curated_leaderboard.csv     — curated KPI table (included)
  curated_leaderboard.md      — curated leaderboard (included)

docs/
  technical_solution.md
  MODEL_EVALUATION.md
  PROJECT_ARCHITECTURE.md
  RUNBOOK.md
  FINAL_CHECKLIST.md

tests/                         — unit and integration tests
```

---

## Documentation

- [docs/technical_solution.md](docs/technical_solution.md)
- [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md)
- [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md)
- [docs/RUNBOOK.md](docs/RUNBOOK.md)

---

## Limitations

- The source dataset is a generic IT and customer-support dataset, not native insurance data. Topic and action proxy labels are approximate.
- `queue → topic` mapping is curated but imperfect; some Kaggle queues do not map cleanly to the five insurance topics.
- `next_action` has no ground truth in the dataset. Action proxy agreement is a lower-bound diagnostic only.
- No live external systems are called. The action executor is fully simulated.
- No human feedback loop. The model does not improve from labelled corrections.

---

## Dataset Attribution

The dataset is sourced from Kaggle:  
**Multilingual Customer Support Tickets** by Tobias Bueck  
[kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)

Redistribution terms are uncertain. The raw dataset CSV is **not included** in this public repository.

---

## What Is Not Included in This Repository

The following are excluded from the public snapshot:

- Full raw dataset CSV (`data/dataset-*.csv`)
- DuckDB database (`data/tickets.duckdb`)
- LanceDB vector index (`data/lancedb/`)
- Ollama models (pulled locally via `ollama pull`)
- Python virtual environment (`.venv/`)
- HuggingFace model cache (`.cache/`)

> **Fresh-clone note:** `outputs/curated_leaderboard.csv` and `outputs/curated_leaderboard.md` are static files included in the repository. They were generated from the submitted local experiment history. The run IDs configured in `config.yaml` under `leaderboard` refer to that history and will not exist in a freshly rebuilt database. Rebuilding via `python main.py` creates new run IDs. To regenerate the curated leaderboard files from a local database, run:
>
> ```powershell
> python scripts/export_curated_leaderboard.py
> ```

---

## Public Snapshot Artifacts

The committed `outputs/triage_results.csv` and `outputs/triage_trace.jsonl` are a representative 20-ticket sample from a local run. The controlled 200-ticket evaluations are documented in `outputs/curated_leaderboard.md` and `docs/MODEL_EVALUATION.md`.

The full raw dataset, DuckDB experiment database, LanceDB index, local Ollama models, and virtual environment are intentionally excluded from the public repository.
