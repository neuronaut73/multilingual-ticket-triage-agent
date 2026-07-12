# Project Architecture — Local Multilingual Ticket Triage Agent

---

## 1. Scope and Constraints

The system is a local, auditable prototype for insurance support ticket triage. All components run locally — no cloud APIs, no paid services.

**Design priorities:**

1. Correctness
2. Data-leakage safety
3. Readability and explainability
4. Minimalism — no unnecessary abstractions, no frameworks unless needed
5. Reproducibility — deterministic splits, deterministic routing, versioned config

---

## 2. Final Architecture

```text
CSV ticket data
→ preprocessing (normalize, build representation_text)
→ DuckDB (historical_tickets, reference/eval split)
→ reference split: embeddings → LanceDB vector index
→ eval split (simulates new tickets):
    → embed query (subject + body only)
    → LanceDB top-k neighbor retrieval
    → weighted neighbor voting (queue, priority)
    → local LLM analyzer (Ollama → Pydantic LLMAnalysis)
    → deterministic validator (confidence, missing info, LLM/kNN agreement)
    → optional conditional reviewer (second-pass LLM when trigger flags fire)
    → deterministic router (business rules → next_action)
    → simulated action executor (records action result)
→ CSV / JSONL outputs
→ evaluation metrics + run tracking in DuckDB
```

Two entry points:

- `main.py` — batch build and evaluation
- `cli.py` — interactive demo

---

## 3. Module Responsibilities

```text
src/domain/
  enums.py              — Topic, Urgency, NextAction enums
  models.py             — Pydantic models: TicketInput, LLMAnalysis, NeighborPrediction,
                          ValidationResult, TriageResult, ActionExecutionResult
  ontology.py           — loads ontology/ticket_ontology.yaml
  mapping.py            — queue→topic, priority→urgency, topic→proxy_next_action

src/application/
  agent.py              — TicketTriageAgent: orchestrates the full per-ticket workflow
  analyzer.py           — LLM analyzer: builds prompt, calls Ollama, validates Pydantic output
  reviewer.py           — ConditionalLLMReviewer: second-pass LLM, invoked on trigger flags
  validator.py          — deterministic ValidationResult from LLMAnalysis + NeighborPrediction
  router.py             — maps ValidationResult → NextAction via business rules
  action_executor.py    — ActionExecutor: simulated local action tools
  batch_runner.py       — iterates eval tickets, calls agent, writes CSV/JSONL, stores metrics
  metrics.py            — accuracy, macro F1, confusion counts, evaluation metric computation
  neighbor_retriever.py — embed query, top-k LanceDB search, weighted vote → NeighborPrediction
  preprocessing.py      — normalize, clean, build representation_text
  weighted_vote.py      — weighted neighbor vote for queue and priority labels
  cli_menu.py           — interactive CLI menu actions (leakage audit, curated leaderboard, etc.)

src/infrastructure/
  duckdb_repository.py       — reads/writes historical_tickets
  evaluation_repository.py   — reads/writes triage_runs, triage_predictions,
                               triage_metrics, triage_confusion_matrix
  lancedb_ticket_store.py    — LanceDB table: recreate, insert, search
  embedding_model.py         — SentenceTransformer wrapper (encode_passages, encode_queries)
  llm_client.py              — HTTP client for Ollama generate_json
  csv_loader.py              — reads CSV, excludes answer column
  csv_writer.py              — writes triage_results.csv
  trace_writer.py            — writes triage_trace.jsonl

```

---

## 4. Runtime Data Flow

**Build phase (first run or rebuild flags set):**

1. `csv_loader` reads `subject`, `body`, and label columns — never reads `answer`.
2. `duckdb_repository` stores all rows in `historical_tickets` with deterministic reference/eval split.
3. `embedding_model` encodes all reference-split tickets.
4. `lancedb_ticket_store` stores reference vectors with label metadata.

**Triage phase (per eval ticket):**

1. Build `TicketInput` from `ticket_id`, `subject`, `body` only.
2. `neighbor_retriever.retrieve_and_predict` — embed query, top-k search, weighted vote → `NeighborPrediction`.
3. `analyzer.analyze` — build prompt, call Ollama, validate Pydantic → first `LLMAnalysis`.
4. `validator.validate` — check confidence, missing info, LLM/kNN agreement → `ValidationResult`.
5. If reviewer is enabled and trigger flags fire → `reviewer.review` → reviewed `LLMAnalysis`; re-validate; accept if `confidence > 0`.
6. `router.route` → `NextAction`.
7. `action_executor.execute` → `ActionExecutionResult`.
8. Assemble `TriageResult`; write row to CSV and JSONL; store prediction in DuckDB.

**Evaluation phase (after batch):**

9. `metrics.compute_evaluation_metrics` — compare predictions to proxy labels from DuckDB.
10. `evaluation_repository` — write run record, KPIs, confusion counts.
11. Write `run_summary.json`. Curated leaderboard files are not written automatically; run `python scripts/export_curated_leaderboard.py` to regenerate them.

---

## 5. Prediction/Evaluation Leakage Boundary

**Allowed at prediction time:**

- `subject` and `body` of the current ticket
- Neighbor evidence from **reference** tickets (historical labels of other tickets — not the current ticket's labels)
- Assignment ontology (allowed topics, urgency values, required fields)

**Never used at prediction time:**

- `answer` of any ticket
- `queue`, `priority`, `type`, `tag_1`–`tag_8` of the current ticket
- `proxy_topic`, `proxy_urgency`, `proxy_next_action` of the current ticket

**Used only after prediction:**

- All `actual_*` and `proxy_*` columns of the current ticket — retrieved from DuckDB for evaluation comparison only.

**LanceDB contains only reference-split tickets.** Eval tickets are never inserted.

---

## 6. Conditional Reviewer Workflow

```text
analyzer → first LLMAnalysis
         ↓
validator → ValidationResult (flags, notes, requires_human_review)
         ↓
      trigger flags? ──No──→ router
         ↓ Yes
reviewer.review(ticket, neighbor_prediction, first_analysis, validation)
         → reviewed LLMAnalysis
         ↓
      confidence > 0? ──No──→ keep first_analysis
         ↓ Yes
      re-validate reviewed LLMAnalysis → reviewed ValidationResult
         ↓
      router
```

**Trigger flags** (configurable in `config.yaml`):
- `low_llm_confidence` — always triggers (no confidence gate)
- `urgency_disagreement` — triggers only when first_analysis.confidence < `urgency_disagreement_confidence_ceiling`
- `topic_disagreement` — triggers only when first_analysis.confidence < `disagreement_confidence_ceiling`

**Reviewer versus `requires_human_review`:**
- Reviewer invocation is a pre-routing LLM step that may revise the analysis.
- `requires_human_review` is a post-routing flag set by deterministic business rules — it routes the ticket to a human agent. The two are independent.

The reviewer is **disabled by default** (`reviewer.enabled: false`). The CLI allows enabling or disabling it per operation without modifying `config.yaml`.

---

## 7. Deterministic Router and Simulated Executor

### Router logic (simplified)

```python
if validation.has_flag("missing_information"):
    return NextAction.ASK_FOR_MORE_INFORMATION
elif validation.requires_human_review or analysis.urgency == Urgency.HIGH:
    return NextAction.ESCALATE_TO_HUMAN_SUPERVISOR
else:
    return ontology.default_action(analysis.topic)
```

All six next actions are produced by deterministic rules, never by unconstrained LLM output.

### Six next actions

```text
send_standard_faq_or_self_service_link
create_or_update_claim
forward_to_billing_team
forward_to_technical_support
escalate_to_human_supervisor
ask_for_more_information
```

### ActionExecutor

The executor receives `next_action`, `ticket_id`, and `short_note`. It returns a simulated `ActionExecutionResult` with `status: "simulated"`. No real external systems are called.

---

## 8. Persistence Schema

### DuckDB — `data/tickets.duckdb`

**`historical_tickets`**
```text
ticket_id, split_name, subject, body, raw_text, cleaned_text,
representation_text, actual_queue, actual_priority, actual_type,
actual_tags_json, proxy_topic, proxy_urgency, proxy_next_action,
proxy_topic_source, language, source_row_json
```

**`triage_runs`**
```text
run_id, created_at, model_name, embedding_model_name,
limit_n, split_name, sample_strategy, random_seed,
reviewer_model_name, reviewer_enabled, top_k
```

**`triage_predictions`**
```text
run_id, ticket_id, predicted_topic, predicted_urgency, predicted_next_action,
llm_confidence, requires_human_review, missing_info, reviewer_used,
reviewer_model, reviewer_changed_topic, reviewer_changed_urgency,
first_topic, first_urgency, first_confidence, created_at
```

**`triage_metrics`**
```text
run_id, metric_name, metric_value
```

**`triage_confusion_matrix`**
```text
run_id, label_type, actual_label, predicted_label, count
```

### LanceDB — `data/lancedb/`

Table: `ticket_embeddings`

```text
ticket_id, split_name, actual_queue, actual_priority, actual_type,
actual_tags_json, proxy_topic, proxy_urgency, proxy_next_action,
proxy_topic_source, text_snippet, vector (1024-dim float32)
```

---

## 9. Output Artifacts

| File | Description |
|---|---|
| `outputs/triage_results.csv` | One row per processed eval ticket: topic, urgency, next action, confidence, flags |
| `outputs/triage_trace.jsonl` | One JSON line per ticket: full evidence and decision trace |
| `outputs/run_summary.json` | Run-level summary: run_id, config, timing, evaluation metrics |
| `outputs/curated_leaderboard.csv` | Curated KPI table for the seven configured runs |
| `outputs/curated_leaderboard.md` | Human-readable curated leaderboard |

---

## 10. Evaluation Design

### Metrics

| Metric | Signal quality | Notes |
|---|---|---|
| `urgency_accuracy` / `urgency_macro_f1` | Strong | Direct 1:1 mapping: `priority` → urgency |
| `topic_proxy_accuracy` / `topic_macro_f1` | Weaker | Proxy via `queue → topic` mapping |
| `next_action_proxy_agreement` | Proxy only | No ground truth; derived from proxy topic |
| `human_review_rate` | Operational | Reflects routing safety behaviour |
| `reviewer_invocation_rate` | Reviewer | Share of tickets where reviewer was called |
| `avg_seconds_per_ticket` / `p95_seconds_per_ticket` | Throughput | Wall-clock timing |

### Curated leaderboard

Seven runs are grouped in `config.yaml` under `leaderboard`:
- `controlled_reviewer_run_ids` — two runs for the controlled reviewer A/B
- `featured_run_id` — best observed reviewer configuration
- `analyzer_screening_run_ids` — historical analyzer screening runs

The CLI (option 4→1) displays these runs grouped by methodology. The full experiment history (all stored runs) is accessible via CLI option 4→2 or by querying DuckDB directly.

The static files `outputs/curated_leaderboard.csv` and `outputs/curated_leaderboard.md` are generated by:

```
python scripts/export_curated_leaderboard.py
```

The script opens DuckDB in read-only mode and reads the configured run IDs from `config.yaml`. The configured run IDs refer to the submitted local experiment history and will not be found in a freshly rebuilt database, which assigns new run IDs.

---

## 11. Explicit Exclusions and Future Extensions

**Not implemented (out of scope for this prototype):**

- LangGraph, LangFlow, or other orchestration frameworks
- Cloud APIs or paid services
- Real external ticket system integrations
- Human feedback loop or active learning
- FastAPI web app or dashboard
- Automatic prompt optimizer
- Knowledge graph or Neo4j
- Synthetic training data generation
- Full classifier training pipeline

**Potential future extensions:**

- LangGraph wrapper around the existing sequential node boundaries
- Specialist agents per department (Claims, Billing, Technical, Escalation)
- Phoenix or Langfuse for LLM observability and prompt experiment tracking
- Domain-specific insurance ticket data for reference set expansion
- Richer simulated action tools (claim draft generation, FAQ lookup)
- Cross-encoder reranking for top-k neighbor retrieval
