# Model Evaluation — Local Multilingual Ticket Triage Agent

---

## 1. Evaluation Setup

Evaluation is performed after each batch run. The batch runner iterates over the **eval split** (approximately 20% of the dataset), processes each ticket through the full pipeline, and compares predicted values to proxy ground-truth labels stored in DuckDB.

### Key integrity rules

- Eval tickets are never embedded into LanceDB. The LanceDB index contains only reference-split tickets.
- Ground-truth labels are not passed to the model. They are retrieved from DuckDB only after prediction.
- `actual_*` and `proxy_*` columns are evaluation metadata. They are never used as model input.
- `answer` is never read or used at any point in the pipeline.

---

## 2. Split Strategy

| Split | Approx. size | Role |
|---|---|---|
| **Reference** | ~80% | Indexed into LanceDB; used for neighbor retrieval and historical evidence |
| **Eval** | ~20% | Simulates new incoming tickets; processed ticket by ticket through the full agent pipeline |

The split is deterministic: it is based on a hash of `ticket_id`, so it is reproducible across runs without a random seed.

---

## 3. Metrics

| Metric | Computed from | Interpretation |
|---|---|---|
| `urgency_accuracy` | predicted urgency vs `proxy_urgency` (from `priority`) | Strongest evaluation signal — direct 1:1 label mapping |
| `urgency_macro_f1` | macro-averaged F1 over Low / Medium / High | Per-class average; not biased toward majority class |
| `topic_proxy_accuracy` | predicted topic vs `proxy_topic` (from `queue` mapping) | Proxy quality depends on quality of the queue→topic mapping |
| `topic_macro_f1` | macro-averaged F1 over the five topics | Per-class average |
| `next_action_proxy_agreement` | predicted next_action vs `proxy_next_action` (derived from proxy topic) | **Not true action accuracy** — see note below |
| `human_review_rate` | share of tickets with `requires_human_review=true` | Operational safety/quality metric |
| `missing_info_rate` | share of tickets with `missing_info=true` | Operational quality metric |
| `average_confidence` | mean of LLM-reported confidence scores | Diagnostic |
| `avg_seconds_per_ticket` | mean wall-clock time per ticket | Throughput |
| `p95_seconds_per_ticket` | 95th percentile wall-clock time | Tail latency |
| `reviewer_invocation_rate` | share of tickets where the conditional reviewer was called | Reviewer usage metric |

### Why urgency metrics are the strongest signal

The source dataset contains a `priority` column with three values: `high`, `medium`, `low`. These map directly to the assignment urgency levels `High`, `Medium`, `Low` with no ambiguity. The 1:1 mapping makes urgency the most reliable evaluation dimension.

### Why topic metrics are weaker

The source dataset has a `queue` column (e.g. "Technical Support", "Billing and Payments") that does not directly match the assignment insurance topics. A curated mapping in `ontology/ticket_ontology.yaml` bridges the two label spaces. Topic proxy accuracy reflects both model classification quality and the quality of this mapping.

### Why next_action_proxy_agreement is low

`next_action_proxy_agreement` measures how often the deterministic router's final `next_action` matches the proxy action derived from the proxy topic. It is expected to be low for two structural reasons:

1. The deterministic router overrides topic-based proxy actions for high-urgency or human-review cases. In runs with a 36–42% human review rate, a large share of tickets are escalated regardless of topic.
2. The proxy action is derived mechanically from the proxy topic, which may itself be incorrect.

The `human_review_rate` is the more meaningful operational metric for assessing routing behaviour.

---

## 4. Curated Evaluation — Seven Configured Runs

Seven runs are configured in `config.yaml` and displayed in the curated leaderboard (CLI option 4→1 or `outputs/curated_leaderboard.md`). They are grouped by methodology.

---

### Section A — Controlled Reviewer A/B

These two runs share identical settings:

- Analyzer: `llama3.2:3b`
- Eval sample: 200 tickets, balanced by proxy topic, seed 42
- Embedding model: `intfloat/multilingual-e5-large`
- top_k: 5
- Thresholds and trigger flags: identical

Only the conditional reviewer differs: OFF versus `granite4.1:8b` reviewer ON.

| Run ID | Reviewer | Reviewer rate | Urgency Acc | Urgency F1 | Topic Acc | Topic F1 | Action Agr. | Human Rev. | Avg Conf. | sec/ticket |
|---|---|---|---|---|---|---|---|---|---|---|
| `run_20260713_014052_ab_off` | — | 0.0% | 55.5% | 55.4% | 59.5% | 56.0% | 19.5% | 42.0% | 83.3% | 2.904 |
| `run_20260713_014052_ab_on` | granite4.1:8b | 21.0% | 57.0% | 56.0% | 60.0% | 56.0% | 23.0% | 38.0% | 84.9% | 3.599 |

**Deltas (ON − OFF):**

- Reviewer invocation rate: 21% of tickets triggered reviewer invocation
- Urgency accuracy: +1.5 percentage points (55.5% → 57.0%)
- Urgency macro F1: +0.65 percentage points (55.38% → 56.03%)
- Topic accuracy: +0.5 percentage points (59.5% → 60.0%)
- Topic macro F1: effectively unchanged (55.96% → 55.95%, −0.01 pp)
- Action proxy agreement: +3.5 percentage points (19.5% → 23.0%)
- Human-review rate: −4.0 percentage points (42.0% → 38.0%)
- Average confidence: +1.66 percentage points (83.27% → 84.93%)
- Average latency: +0.695 seconds/ticket (2.904 → 3.599 sec), approximately +24%
- P95 latency: +3.182 seconds (2.917 → 6.099 sec) — reviewed tickets incur an additional model call

**Conclusion from the controlled A/B:**

The conditional `granite4.1:8b` reviewer was invoked for 21% of tickets. It improved urgency accuracy (+1.5 pp), action proxy agreement (+3.5 pp), average confidence (+1.66 pp), and reduced the human-review rate (−4.0 pp). Topic macro F1 remained effectively unchanged. Average latency increased from 2.904 to 3.599 seconds per ticket, approximately 24%.

P95 latency increased from 2.917 to 6.099 seconds because reviewed tickets require an additional model call. The reviewer is therefore beneficial selectively, not free. Results support conditional invocation instead of reviewing every ticket.

**Methodological caveats:**
- Topic and action metrics are proxy-based; the `queue→topic` mapping is indirect.
- The reviewer was invoked selectively (21% of tickets), not for all tickets.
- One controlled run does not establish universal superiority.

---

### Earlier controlled reviewer experiment

This earlier run used `qwen3.5:9b` as reviewer (not analyzer) on a balanced 200-ticket sample. It showed minimal quality change with substantially higher latency, which motivated the decision to test `granite4.1:8b` as reviewer.

| Run ID | Reviewer | Reviewer rate | Urgency Acc | Urgency F1 | Topic Acc | Topic F1 | Action Agr. | Human Rev. | Avg Conf. | sec/ticket |
|---|---|---|---|---|---|---|---|---|---|---|
| `run_20260712_215929_ab_off` | — | 0.0% | 55.0% | 54.7% | 59.0% | 55.2% | 19.5% | 42.5% | 83.4% | 2.93 |
| `run_20260712_215929_ab_on` | qwen3.5:9b | 23.0% | 55.5% | 55.2% | 59.0% | 54.8% | 19.5% | 41.5% | 83.0% | 4.55 |

Urgency accuracy improved by only 0.5 pp; topic accuracy and action agreement were unchanged; latency increased +1.62 sec/ticket (+55%). The reviewer produced negligible quality change at significant latency cost.

> Note: `qwen3.5:9b` was used here as the **reviewer** — not the primary analyzer. An earlier screening attempt used `qwen3.5:9b` as the **analyzer** and failed the structured-output contract at that time. That failure reflects that configuration, not a universal property of the model.

---

### Section B — Earlier Best Observed Granite Reviewer Run (Different Sample)

This run used `llama3.2:3b` as analyzer and `granite4.1:8b` as reviewer on a different ticket sample. It is retained as historical context only. It is **not** the controlled A/B result.

| Run ID | Reviewer | Reviewer rate | Urgency Acc | Urgency F1 | Topic Acc | Topic F1 | Action Agr. | Human Rev. | Avg Conf. | sec/ticket |
|---|---|---|---|---|---|---|---|---|---|---|
| `run_20260711_200952` | granite4.1:8b | 20.5% | 57.0% | 55.7% | 61.5% | 58.5% | 24.0% | 37.0% | 85.0% | 3.55 |

This run produced strong topic proxy accuracy (61.5%) and action agreement (24.0%). Because it used a different ticket sample from Section A, it is not directly comparable to the Section A A/B results.

---

### Section C — Historical Analyzer Screening

These four runs document model screening across different analyzer choices, all on 200-ticket eval batches without the conditional reviewer. Some early screening runs predate complete sampling metadata, so they are informative model screening rather than a strict controlled A/B experiment.

| Run ID | Analyzer | Urgency Acc | Urgency F1 | Topic Acc | Topic F1 | Action Agr. | Human Rev. | Avg Conf. | sec/ticket |
|---|---|---|---|---|---|---|---|---|---|
| `run_20260710_032230` | qwen3-coder:30b | 63.0% | 57.9% | 63.5% | 61.7% | 20.0% | 34.5% | 80.5% | 4.96 |
| `run_20260710_025155` | devstral-small-2:24b | 65.5% | 62.6% | 57.0% | 52.0% | 18.0% | 34.0% | 91.0% | 4.41 |
| `run_20260710_015910` | deepseek-r1:14b | 65.0% | 61.7% | 58.0% | 53.4% | 18.0% | 34.5% | 87.8% | 3.87 |
| `run_20260710_013943` | llama3.2:3b | 61.5% | 58.7% | 60.5% | 56.7% | 17.0% | 36.0% | 85.0% | 2.90 |

**Observations from screening:**

- Larger models (`devstral-small-2:24b`, `deepseek-r1:14b`) achieved higher urgency accuracy/F1 but weaker topic proxy accuracy.
- `qwen3-coder:30b` showed the best topic proxy metrics in screening but is the slowest.
- `llama3.2:3b` achieved strong topic proxy accuracy (60.5%) with the best throughput (2.90 sec/ticket), making it the default analyzer choice.
- Human review rates are lower in the screening runs (34–36%) compared to the controlled A/B runs (42.5%). This likely reflects differences in sampling strategy across runs.

---

## 5. Model Selection Rationale

### Why llama3.2:3b is the selected default analyzer

- Strong topic proxy accuracy (60.5%) and urgency accuracy (61.5%) in screening.
- Fastest throughput at 2.90 sec/ticket — approximately 40% faster than `qwen3-coder:30b`.
- Produces valid structured output consistently (zero unresolvable validation failures in 200-ticket runs).
- Smaller footprint, practical for local deployment.

### Why granite4.1:8b is the configured reviewer

- The latest controlled A/B (`run_20260713_014052_ab_on`) demonstrated modest operational improvements: +1.5 pp urgency accuracy, +3.5 pp action agreement, −4.0 pp human-review rate, at a +24% latency cost.
- An earlier run (`run_20260711_200952`) on a different sample also produced strong results.
- The reviewer is invoked conditionally (21% of tickets), not for every ticket.

### Reviewer model selection

The latest controlled A/B showed that `granite4.1:8b` as reviewer produced modest but positive quality improvements with a 24% latency overhead. An earlier experiment with `qwen3.5:9b` as reviewer showed minimal quality change at a higher latency cost (+55%). Neither reviewer model produces a consistent improvement large enough to recommend automatic enablement; the reviewer remains disabled by default and is invoked only when validation flags fire.

---

## 6. SQL to Compare Runs

Run this query against `data/tickets.duckdb` using DBeaver, the DuckDB CLI, or a DuckDB Python connection:

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
  max(case when m.metric_name = 'avg_seconds_per_ticket' then m.metric_value end) as avg_seconds_per_ticket,
  max(case when m.metric_name = 'p95_seconds_per_ticket' then m.metric_value end) as p95_seconds_per_ticket
from triage_runs r
join triage_metrics m using (run_id)
group by r.run_id, r.created_at, r.model_name, r.limit_n
order by r.created_at desc;
```

### Latest runs

```sql
select run_id, created_at, model_name, limit_n
from triage_runs
order by created_at desc
limit 10;
```

### Confusion matrix for a specific run

```sql
select label_type, actual_label, predicted_label, count
from triage_confusion_matrix
where run_id = 'run_20260713_014052_ab_off'
order by label_type, actual_label, predicted_label;
```

---

## 7. DuckDB Tables

| Table | Description |
|---|---|
| `triage_runs` | One row per batch run: run_id, created_at, model_name, embedding model, config |
| `triage_predictions` | One row per ticket per run: predicted topic, urgency, next action, confidence, flags |
| `triage_metrics` | One row per KPI per run: metric_name, metric_value |
| `triage_confusion_matrix` | Confusion counts per run: label_type, actual_label, predicted_label, count |

All run history is queryable from a single local DuckDB file. No external observability platform is required.
