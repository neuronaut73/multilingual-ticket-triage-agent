# Curated Evaluation Leaderboard — HDI Ticket Triage Agent

This file contains only the seven configured runs grouped by methodology.
Use `outputs/leaderboard_database_snapshot.md` to inspect all stored runs.

Database was opened in **read-only** mode. No rows were modified.

---

## Section A: Controlled Reviewer A/B — Same Analyzer and Ticket Sample

_Note: A and B use the same llama3.2:3b analyzer, 200-ticket balanced eval_
_sample, seed 42, embedding model, retrieval configuration, thresholds, and_
_trigger flags. Only the conditional reviewer differs._

### run_20260713_014052_ab_off
- **triage_model**      : llama3.2:3b
- **reviewer**          : no  model=—
- **tickets (N)**       : 200
- **sec/ticket**        : 2.904
- **reviewer_rate**     : 0.0%
- **urgency_acc / f1**  : 55.5% / 55.4%
- **topic_acc / f1**    : 59.5% / 56.0%
- **action_agreement**  : 19.5%
- **human_review_rate** : 42.0%
- **avg_confidence**    : 83.3%

### run_20260713_014052_ab_on ★ Best observed reviewer configuration
- **triage_model**      : llama3.2:3b
- **reviewer**          : yes  model=granite4.1:8b
- **tickets (N)**       : 200
- **sec/ticket**        : 3.599
- **reviewer_rate**     : 21.0%
- **urgency_acc / f1**  : 57.0% / 56.0%
- **topic_acc / f1**    : 60.0% / 56.0%
- **action_agreement**  : 23.0%
- **human_review_rate** : 38.0%
- **avg_confidence**    : 84.9%


---

## Section B: Best Observed Reviewer Configuration

### run_20260711_200952
- **triage_model**      : llama3.2:3b
- **reviewer**          : yes  model=granite4.1:8b
- **tickets (N)**       : 200
- **sec/ticket**        : 3.545
- **reviewer_rate**     : 20.5%
- **urgency_acc / f1**  : 57.0% / 55.7%
- **topic_acc / f1**    : 61.5% / 58.5%
- **action_agreement**  : 24.0%
- **human_review_rate** : 37.0%
- **avg_confidence**    : 85.0%


---

## Section C: Historical Analyzer Screening — 200 Tickets

_Note: These runs document model screening. Some early runs predate_
_complete sampling metadata, so they are not presented as a strict_
_controlled A/B experiment._

### run_20260710_032230
- **triage_model**      : qwen3-coder:30b
- **reviewer**          : —  model=—
- **tickets (N)**       : 200
- **sec/ticket**        : 4.958
- **reviewer_rate**     : —
- **urgency_acc / f1**  : 63.0% / 57.9%
- **topic_acc / f1**    : 63.5% / 61.7%
- **action_agreement**  : 20.0%
- **human_review_rate** : 34.5%
- **avg_confidence**    : 80.5%

### run_20260710_025155
- **triage_model**      : devstral-small-2:24b
- **reviewer**          : —  model=—
- **tickets (N)**       : 200
- **sec/ticket**        : 4.405
- **reviewer_rate**     : —
- **urgency_acc / f1**  : 65.5% / 62.6%
- **topic_acc / f1**    : 57.0% / 52.0%
- **action_agreement**  : 18.0%
- **human_review_rate** : 34.0%
- **avg_confidence**    : 91.0%

### run_20260710_015910
- **triage_model**      : deepseek-r1:14b
- **reviewer**          : —  model=—
- **tickets (N)**       : 200
- **sec/ticket**        : 3.870
- **reviewer_rate**     : —
- **urgency_acc / f1**  : 65.0% / 61.7%
- **topic_acc / f1**    : 58.0% / 53.4%
- **action_agreement**  : 18.0%
- **human_review_rate** : 34.5%
- **avg_confidence**    : 87.8%

### run_20260710_013943
- **triage_model**      : llama3.2:3b
- **reviewer**          : —  model=—
- **tickets (N)**       : 200
- **sec/ticket**        : 2.898
- **reviewer_rate**     : —
- **urgency_acc / f1**  : 61.5% / 58.7%
- **topic_acc / f1**    : 60.5% / 56.7%
- **action_agreement**  : 17.0%
- **human_review_rate** : 36.0%
- **avg_confidence**    : 85.0%

---

Use 'Full experiment history' in the CLI to inspect all stored pilot,
smoke-test, failed, and repeated runs.

---

## Confirmation

- Database opened in **read-only** mode.
- No rows were inserted, updated, deleted, or truncated.
- No application source files were modified.
- No LLM inference was run.

