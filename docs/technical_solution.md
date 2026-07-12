# Technical Solution — Local Multilingual Ticket Triage Agent

---

## 1. Problem and Objective

The objective is to build a local, auditable prototype for an insurance support ticket triage agent. The system:

- Ingests multilingual customer-support tickets from a CSV dataset.
- Classifies each ticket into a **topic**, an **urgency level**, and a **next action**.
- Processes at least 200 tickets end-to-end in a batch run.
- Produces structured CSV and JSON outputs for audit and submission.
- Remains entirely local — no cloud APIs, no paid services, no external calls.
- Is explainable line by line in a technical interview.

The five assignment topics are:

- **Policy / Contract** — coverage questions, cancellations, contract documents
- **Claims / Damage** — accidents, damage reports, reimbursements
- **Billing / Payment** — invoices, premiums, refunds, failed payments
- **Technical / Online Access** — portal login, password, app access
- **Other** — unclear or insufficiently specific requests

The three urgency levels are: **Low**, **Medium**, **High**.

The six next actions are: `send_standard_faq_or_self_service_link`, `create_or_update_claim`, `forward_to_billing_team`, `forward_to_technical_support`, `escalate_to_human_supervisor`, `ask_for_more_information`.

---

## 2. Data and No-Leakage Setup

### Dataset

The dataset is a Kaggle multilingual customer-support ticket CSV:

```text
subject, body, answer, type, queue, priority, language, tag_1, ..., tag_8
```

### Prediction-time input

Only `subject` and `body` are used as model input at prediction time. All other columns are treated as historical metadata or ground-truth labels.

```text
input_text = "Subject: " + subject + "\n\nBody: " + first_1500_chars(body)
```

### No-leakage rules

| Column | Role |
|---|---|
| `subject`, `body` | Prediction-time input |
| `answer` | **Never used** — not available for new tickets |
| `queue`, `priority`, `type`, `tag_1`–`tag_8` | Historical labels — used for evaluation and retrieval metadata only |

### Dataset split

The dataset is split deterministically by `ticket_id` hash:

| Split | Size | Role |
|---|---|---|
| **Reference** | ~80% | Embedded into LanceDB; used for neighbor retrieval |
| **Eval** | ~20% | Simulates new incoming tickets; never embedded into LanceDB |

Eval tickets are passed through the pipeline as if entirely new. Ground-truth labels are stored separately in DuckDB and only accessed after prediction for evaluation.

---

## 3. Ontology and Proxy Labels

### Assignment ontology

The LLM is prompted to output structured fields from the assignment schema only:

```text
topic       — routing category (5 values)
urgency     — urgency level (Low / Medium / High)
next_action — deterministic action (6 values)
```

### Mapping layer

```text
priority  →  urgency          (direct 1:1 mapping)
  high    →  High
  medium  →  Medium
  low     →  Low

queue  →  topic               (curated mapping in ontology/ticket_ontology.yaml)
  "Technical Support"         →  "Technical / Online Access"
  "Billing and Payments"      →  "Billing / Payment"
  "Returns and Exchanges"     →  "Other"
  ...

topic  →  proxy_next_action   (from ontology/ticket_ontology.yaml)
  "Technical / Online Access" →  forward_to_technical_support
  "Billing / Payment"         →  forward_to_billing_team
  "Claims / Damage"           →  create_or_update_claim
  "Policy / Contract"         →  send_standard_faq_or_self_service_link
  "Other"                     →  ask_for_more_information
```

### Proxy label limitations

- `priority → urgency` is a strong proxy (same scale, direct mapping).
- `queue → topic` is a weaker proxy because Kaggle queue names are not insurance-specific.
- `next_action` has no ground truth in the dataset. Agreement is measured against a proxy action derived from the proxy topic mapping.

---

## 4. Architecture

The full pipeline:

```text
CSV
→ preprocessing (normalize, build representation_text)
→ DuckDB (historical_tickets table, reference/eval split)
→ reference split: multilingual embeddings → LanceDB reference vector index
→ eval split (simulates new tickets):
    → embed query (subject + body)
    → LanceDB top-k neighbor retrieval
    → weighted neighbor voting (queue, priority)
    → local LLM analyzer (topic, urgency, confidence, missing_info, note)
    → Pydantic validation + retry + fallback
    → deterministic validator (confidence check, missing info, LLM/kNN agreement)
    → optional conditional reviewer (second-pass LLM when trigger flags fire)
    → deterministic router (business rules → next_action)
    → simulated action executor (records action result)
→ CSV / JSONL outputs
→ evaluation metrics + run tracking in DuckDB
```

### Storage

| Store | Purpose |
|---|---|
| `data/tickets.duckdb` | Structured tables: tickets, predictions, metrics, run tracking |
| `data/lancedb/` | Vector store: one embedding per reference ticket, with label metadata |

---

## 5. Agentic Workflow

The system implements a multi-step agentic workflow:

1. **Retrieval** — embed the ticket query; fetch top-k similar historical tickets from LanceDB
2. **Aggregation** — weighted vote over neighbor labels to produce predicted queue and priority with confidence
3. **LLM Analysis** — local Ollama model produces structured triage output from subject + body + neighbor context
4. **Pydantic validation** — output is validated against strict Pydantic models; invalid output triggers retry; persistent failure triggers fallback
5. **Deterministic validation** — checks confidence thresholds, missing info flags, and LLM/kNN urgency agreement
6. **Conditional reviewer** — if validation raises trigger flags, a second-pass LLM is invoked to review and optionally revise the first analysis
7. **Deterministic routing** — business rules map validator signals to a final `next_action`
8. **Simulated action execution** — the action is passed to a local action executor that records a simulated result
9. **Trace logging** — full per-ticket evidence and decision trace written to JSONL

---

## 6. Local LLM Analyzer

The LLM analyzer uses Ollama. It receives:

- `subject` and `body` (prediction-time input only)
- Assignment ontology (allowed topics and urgency values)
- Retrieved neighbor context (snippets and labels from reference tickets)

It outputs a structured JSON response validated against a Pydantic model:

```json
{
  "topic": "Technical / Online Access",
  "urgency": "High",
  "confidence": 0.82,
  "missing_info": true,
  "missing_fields": ["customer_identifier"],
  "short_note": "Customer reports blocked portal access.",
  "clarification_question": "Please provide your customer number and the error shown."
}
```

The configured default analyzer is `llama3.2:3b`.

If the LLM returns invalid JSON or a schema mismatch, a retry is triggered. After `max_retries` failures, the ticket is flagged as `fallback=true` and routed to human review.

---

## 7. Conditional Reviewer

The conditional reviewer (`ConditionalLLMReviewer`) is an optional second-pass LLM invoked only when the deterministic validator raises trigger flags.

### Reviewer versus `requires_human_review`

These are distinct concepts:

- **Reviewer invocation** — a second LLM (a different model) is called to re-examine the first analysis. The reviewer may or may not change the output. This happens before routing.
- **`requires_human_review`** — a flag set by the deterministic validator and router indicating that a human agent should handle this ticket. It is set by Python business rules, not by the reviewer LLM.

### Reviewer inputs

The reviewer sees only:

- Original ticket (`subject` and `body` — no labels)
- First analysis produced by the primary LLM
- Validator flags and notes that triggered the review
- Historical neighbor evidence (from reference tickets — not the current ticket's evaluation labels)

### Acceptance rules

The agent accepts the reviewed output when `reviewed_analysis.confidence > 0`. If the reviewer LLM fails or returns `confidence=0.0`, the first analysis is retained unchanged.

### Reviewer trace fields (in `triage_trace.jsonl`)

| Field | Meaning |
|---|---|
| `reviewer_used` | True when the reviewer was invoked |
| `reviewer_model` | Model name used by the reviewer |
| `reviewer_changed_topic` | True when the reviewer changed the topic |
| `reviewer_changed_urgency` | True when the reviewer changed the urgency |
| `reviewer_note` | Short note from the reviewed analysis |
| `first_short_note` | Short note from the first (pre-reviewer) analysis |
| `reviewer_trigger_flags` | Subset of validator flags that triggered reviewer invocation |

### Current configuration

The reviewer is **disabled by default** in `config.yaml` (`reviewer.enabled: false`). The current configured reviewer model is `granite4.1:8b`. The CLI allows enabling or disabling the reviewer per operation without modifying `config.yaml`.

The latest controlled 200-ticket A/B experiment (`run_20260713_014052_ab_on` vs `run_20260713_014052_ab_off`) confirmed modest operational improvements when the `granite4.1:8b` reviewer is active: +1.5 pp urgency accuracy, +3.5 pp action proxy agreement, −4.0 pp human-review rate, at a +24% average latency cost. These results support conditional invocation rather than reviewing every ticket.

---

## 8. Deterministic Validator and Router

### Validator

The validator applies rule-based checks after the LLM output is parsed:

- If confidence is below the configured threshold → flag `low_llm_confidence`
- If `missing_info=true` → flag `missing_information`
- If LLM urgency and kNN predicted priority strongly disagree → flag `urgency_disagreement`
- If LLM topic and kNN predicted topic disagree → flag `topic_disagreement`

No LLM is involved in validation. The rules are explicit, auditable Python logic.

### Router

The router maps validator signals to a final `next_action` using deterministic business rules:

```text
if missing_info:
    next_action = ask_for_more_information
elif requires_human_review or urgency == High:
    next_action = escalate_to_human_supervisor
else:
    next_action = ontology.default_action(topic)
```

The routing decision is never an unconstrained LLM decision. This makes the system auditable and reproducible.

---

## 9. Evaluation Methodology

### Metrics

| Metric | Description | Notes |
|---|---|---|
| `urgency_accuracy` | Predicted urgency vs priority proxy label | Strongest signal — direct 1:1 mapping |
| `urgency_macro_f1` | Macro F1 for urgency | Per-class average |
| `topic_proxy_accuracy` | Predicted topic vs queue-mapped proxy topic | Weaker — queue mapping is indirect |
| `topic_macro_f1` | Macro F1 for topic | Per-class average |
| `next_action_proxy_agreement` | Agreement with proxy next action | **Not true accuracy** — no ground truth |
| `human_review_rate` | Share of tickets flagged for review | Operational safety metric |
| `missing_info_rate` | Share of tickets with missing info flag | Operational quality metric |
| `average_confidence` | Mean LLM confidence | Diagnostic |
| `avg_seconds_per_ticket` | Mean wall-clock processing time | Throughput |
| `p95_seconds_per_ticket` | 95th percentile processing time | Tail latency |

### Evaluation integrity

- Eval tickets are **never indexed** into LanceDB.
- Ground-truth labels are retrieved from DuckDB only after prediction.
- `actual_*` and `proxy_*` columns are evaluation metadata, not model input.
- `next_action_proxy_agreement` is explicitly named as proxy agreement — not a claim of true action accuracy.

---

## 10. Model Summary

### Analyzer models

| Model | Role | Notes |
|---|---|---|
| `llama3.2:3b` | **Selected default analyzer** | Best balance of speed and proxy KPIs in screening |
| `qwen3-coder:30b` | Screened | Highest urgency accuracy in screening; slower |
| `devstral-small-2:24b` | Screened | Highest urgency F1 in screening; slower |
| `deepseek-r1:14b` | Screened | Strong urgency; slower |

### Reviewer models

| Model | Role | Notes |
|---|---|---|
| `granite4.1:8b` | **Current configured reviewer** | Latest controlled A/B: +1.5 pp urgency accuracy, +3.5 pp action agreement, −4.0 pp human-review rate, +24% latency |
| `qwen3.5:9b` | Earlier reviewer experiment | Minimal quality change (+0.5 pp urgency accuracy); substantially higher latency (+55%); not the current headline result |

> **Note:** `qwen3.5:9b` was used as the reviewer in an earlier controlled A/B experiment. An earlier screening run attempted `qwen3.5:9b` as the **analyzer** and failed the structured-output contract at that time. That failure reflects that configuration, not a universal property of the model.

---

## 11. Run Tracking

Each batch run writes to DuckDB:

| Table | Contents |
|---|---|
| `triage_runs` | run_id, created_at, model_name, embedding model, config settings |
| `triage_predictions` | One row per ticket: predicted topic, urgency, next action, confidence, flags |
| `triage_metrics` | One row per metric per run |
| `triage_confusion_matrix` | Confusion counts for urgency and topic |

This is a lightweight local LLMOps-style approach. Runs are comparable by querying DuckDB directly.

---

## 12. Interactive CLI and Curated Evaluation

The interactive CLI (`python cli.py`) provides six top-level options:

- Triage a single ticket (with reviewer OFF or ON per operation)
- Run a batch evaluation (with reviewer OFF or ON)
- Compare reviewer OFF vs ON on a shared ticket sample
- Inspect evaluation results (curated leaderboard, full history, confusion matrix, per-ticket trace)
- Run a leakage audit (checks eval tickets are not in LanceDB; checks answer column is absent)
- Show runtime configuration

The curated leaderboard (CLI option 4→1) displays the seven runs configured in `config.yaml` grouped by methodology: controlled reviewer A/B, best observed reviewer configuration, and historical analyzer screening.

---

## 13. Limitations

- The source dataset is a generic IT and customer-support dataset, not native insurance data. Topic and action proxy labels are therefore approximate.
- `queue → topic` mapping is curated but imperfect. Some Kaggle queues do not map cleanly to the five insurance topics.
- `next_action` has no ground truth in the dataset. Action proxy agreement is a lower-bound diagnostic only.
- No live external systems are called. The action executor is fully simulated.
- No human feedback loop. The model does not improve from labelled corrections.
- Evaluation is based entirely on proxy labels derived from historical Kaggle metadata.

---

## 14. Future Work

- Integrate Phoenix, Langfuse, or MLflow for richer LLM observability and prompt experiment tracking.
- Add a human feedback loop and active learning to improve routing from labelled corrections.
- Expand the reference set with domain-specific insurance examples to improve neighbor retrieval quality.
- Add richer simulated action tools (e.g. claim draft generation, FAQ lookup).
- Wrap the existing sequential pipeline in LangGraph for graph-based orchestration.
- Fine-tune or prompt-tune the LLM on real labelled insurance ticket data if it becomes available.
- Benchmark additional multilingual embedding models.
