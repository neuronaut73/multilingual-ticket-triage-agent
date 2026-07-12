# Runbook — Local Multilingual Ticket Triage Agent

---

## 1. Prerequisites

Before running the agent, ensure the following:

- Python 3.10 or later
- Virtual environment activated (`.venv`)
- All dependencies installed (`pip install -r requirements.txt`)
- Ollama running locally (`ollama serve`)
- Analyzer model pulled (`ollama pull llama3.2:3b`)
- Dataset CSV at `data/dataset-tickets-multi-lang3-4k.csv`

> **Note:** A 10-ticket application run still requires Ollama. Only the fast unit tests (`-m "not slow"`) avoid Ollama.

---

## Fresh-Clone Note

The public snapshot excludes the DuckDB database (`data/tickets.duckdb`), the LanceDB vector index (`data/lancedb/`), and the raw dataset CSV. Static curated results are included in `outputs/curated_leaderboard.csv` and `outputs/curated_leaderboard.md`.

The run IDs configured in `config.yaml` under `leaderboard` refer to the submitted local experiment history. These run IDs will not exist in a freshly rebuilt database — rebuilding via `python main.py` assigns new run IDs. To regenerate the curated leaderboard files from a local database after a rebuild:

```powershell
python scripts/export_curated_leaderboard.py
```

---

## 2. Environment Activation

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
pip list | Select-String "duckdb\|lancedb\|sentence"
```

---

## 3. Ollama Startup and Model Availability

```powershell
# Start Ollama (if not already running as a background service)
ollama serve

# Pull the default analyzer model
ollama pull llama3.2:3b

# (Optional) Pull the configured reviewer model
ollama pull granite4.1:8b

# Verify available models
ollama list
```

---

## 4. First Build (Run Once)

On the very first run, set rebuild flags in `config.yaml`:

```yaml
runtime:
  rebuild_duckdb: true
  rebuild_lancedb: true
```

Then run:

```powershell
python main.py
```

This will:

1. Read the CSV and populate `data/tickets.duckdb` (`historical_tickets` table, reference/eval split).
2. Embed all reference-split tickets and build the LanceDB vector index at `data/lancedb/`.
3. Run the batch triage pipeline over the eval split.
4. Write outputs to `outputs/`.

**Expected time:** several minutes for the first build (embedding ~3200 reference tickets). Subsequent runs skip this step.

After the first build, set both flags back to `false` in `config.yaml`:

```yaml
runtime:
  rebuild_duckdb: false
  rebuild_lancedb: false
```

---

## 5. Interactive Demo (Interview)

```powershell
python cli.py
```

The CLI loads `config.yaml` on startup. The embedding model is loaded lazily — only when a triage option is first selected.

### Recommended Demo Sequence

**Step 1 — Triage a ticket (option 1)**

Select "Random eval ticket", then choose "Analyzer only" for reviewer mode. The agent runs the full pipeline and prints the evidence and decision trace.

**Step 2 — Inspect curated evaluation (option 4 → 1)**

Shows the seven configured runs grouped by: controlled reviewer A/B, best observed reviewer configuration, and historical analyzer screening. No Ollama required.

**Step 3 — Show full experiment history if requested (option 4 → 2)**

Shows all stored runs in DuckDB, including pilot and smoke-test runs.

**Step 4 — Run leakage audit (option 5)**

Verifies that eval tickets are not present in LanceDB and that the `answer` column is absent from stored data.

**Step 5 — Show runtime configuration (option 6)**

Prints the active settings from `config.yaml`.

---

## 6. Reviewer OFF vs ON Comparison (option 3)

Option 3 runs a small shared ticket sample twice — once with the analyzer only, once with the conditional reviewer enabled — and prints a side-by-side comparison.

This requires Ollama with both the analyzer and reviewer models available.

To enable the reviewer for a single operation (without modifying `config.yaml`), select "Conditional reviewer" when prompted for reviewer mode in options 1 or 2.

---

## 7. Curated Evaluation Submenu (option 4)

```
4.  Inspect evaluation results
  1. Curated evaluation     — grouped view of seven configured runs
  2. Full experiment history — all stored runs
  3. Run details            — KPIs for a specific run_id
  4. Confusion matrix       — confusion counts for a specific run_id
  5. Ticket prediction details — original ticket + stored prediction for a ticket_id + run_id
```

---

## 8. Batch Evaluation via main.py

For a standard 200-ticket batch evaluation with the default analyzer:

```yaml
llm:
  model_name: llama3.2:3b

runtime:
  rebuild_duckdb: false
  rebuild_lancedb: false
  run_smoke_search: false
  run_end_to_end_smoke: false

batch:
  enabled: true
  split: eval
  limit: 200
```

```powershell
python main.py
```

---

## 9. Run Tests

### Fast unit tests (no Ollama, no LanceDB, no DuckDB required)

```powershell
python -m pytest tests -m "not slow" -q
```

### All tests (includes integration tests requiring local infrastructure)

```powershell
python -m pytest tests -q
```

---

## 10. Inspect Outputs

```powershell
# List output files
Get-ChildItem outputs

# Preview triage results CSV
Import-Csv outputs\triage_results.csv | Select-Object -First 5 | Format-Table -AutoSize

# Read run summary
Get-Content outputs\run_summary.json

# Read last 5 trace entries
Get-Content outputs\triage_trace.jsonl -Tail 5

# Read curated leaderboard
Get-Content outputs\curated_leaderboard.md
```

---

## 11. DuckDB Queries

Open `data/tickets.duckdb` in [DBeaver](https://dbeaver.io/) or the DuckDB Python shell.

### Latest batch runs

```sql
select run_id, created_at, model_name, limit_n
from triage_runs
order by created_at desc
limit 10;
```

### All KPIs per run

```sql
select
  r.run_id,
  r.created_at,
  r.model_name,
  r.limit_n,
  max(case when m.metric_name = 'urgency_accuracy' then m.metric_value end) as urgency_accuracy,
  max(case when m.metric_name = 'urgency_macro_f1' then m.metric_value end) as urgency_macro_f1,
  max(case when m.metric_name = 'topic_proxy_accuracy' then m.metric_value end) as topic_proxy_accuracy,
  max(case when m.metric_name = 'topic_macro_f1' then m.metric_value end) as topic_macro_f1,
  max(case when m.metric_name = 'next_action_proxy_agreement' then m.metric_value end) as next_action_proxy_agreement,
  max(case when m.metric_name = 'human_review_rate' then m.metric_value end) as human_review_rate,
  max(case when m.metric_name = 'missing_info_rate' then m.metric_value end) as missing_info_rate,
  max(case when m.metric_name = 'average_confidence' then m.metric_value end) as average_confidence,
  max(case when m.metric_name = 'avg_seconds_per_ticket' then m.metric_value end) as avg_seconds_per_ticket
from triage_runs r
join triage_metrics m using (run_id)
group by r.run_id, r.created_at, r.model_name, r.limit_n
order by r.created_at desc;
```

### Confusion matrix for a specific run

```sql
select label_type, actual_label, predicted_label, count
from triage_confusion_matrix
where run_id = 'run_20260713_014052_ab_off'
order by label_type, actual_label, predicted_label;
```

### Sample predictions

```sql
select ticket_id, predicted_topic, predicted_urgency, predicted_next_action,
       llm_confidence, requires_human_review
from triage_predictions
where run_id = 'run_20260713_014052_ab_off'
limit 20;
```

---

## 12. Troubleshooting

### DuckDB database not found / table missing

**Symptom:** Error mentioning `historical_tickets` table or missing database.

**Fix:** Set `runtime.rebuild_duckdb: true` and run `python main.py`. This recreates the database from the CSV.

---

### LanceDB table not found

**Symptom:** Error mentioning `ticket_embeddings` table or empty vector store.

**Fix:** Set `runtime.rebuild_lancedb: true` and run `python main.py`.

---

### LLM output validation fails repeatedly

**Symptom:** Many tickets flagged as fallback or human review; `missing_info_rate` close to 1.0; `average_confidence` near 0.

**Cause:** The active model may not reliably produce the required JSON schema. This can happen with any model if prompt adaptation is needed.

**Fix:** Switch to the default model:

```yaml
llm:
  model_name: llama3.2:3b
```

For the reviewer, check that the configured reviewer model is available. If using a large reviewer model with limited VRAM, try `granite4.1:8b` which has lower resource requirements.

---

### Ollama model missing

**Symptom:** Connection error or model-not-found error from the LLM client.

**Fix:**

```powershell
ollama pull llama3.2:3b
ollama list
```

---

### Ollama not running

**Symptom:** Connection refused on `http://localhost:11434`.

**Fix:**

```powershell
ollama serve
```

---

### Batch too slow

**Symptom:** Each ticket takes more than 10 seconds; total run time is excessive.

**Fix:** Reduce the batch limit for testing:

```yaml
batch:
  limit: 10
```

For evaluation, `llama3.2:3b` averages ~2.9 sec/ticket. Enabling the reviewer adds ~0.6–1.6 sec/ticket depending on the reviewer model.

---

### Embedding step slow on first run

**Explanation:** Embedding ~3200 reference tickets with `intfloat/multilingual-e5-large` on CPU takes several minutes. This is a one-time cost. Subsequent runs skip this step when `rebuild_lancedb: false`.

---

## 13. Output File Reference

| File | When written | Contents |
|---|---|---|
| `outputs/triage_results.csv` | After each batch run | One row per eval ticket: topic, urgency, next action, confidence, flags |
| `outputs/triage_trace.jsonl` | After each batch run | One JSON line per ticket: full evidence and decision trace |
| `outputs/run_summary.json` | After each batch run | run_id, config, ticket count, timing, evaluation metrics |
| `outputs/curated_leaderboard.csv` | On demand — `python scripts/export_curated_leaderboard.py` | Curated KPI table for the seven configured runs |
| `outputs/curated_leaderboard.md` | On demand — `python scripts/export_curated_leaderboard.py` | Human-readable curated leaderboard |
| `data/tickets.duckdb` | On first build or rebuild | Full structured database |
| `data/lancedb/` | On first build or rebuild | Vector store with reference-split embeddings |
