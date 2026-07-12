"""
Read-only leaderboard snapshot export.

Connects to data/tickets.duckdb in READ-ONLY mode.
Does NOT modify any table, row, or sequence.

Outputs:
  outputs/leaderboard_database_snapshot.csv  — one row per run, all fields
  outputs/leaderboard_database_snapshot.md   — human-readable review file

Usage:
  python scripts/export_leaderboard_snapshot.py
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone

import duckdb

DB_PATH      = "data/tickets.duckdb"
OUT_CSV      = "outputs/leaderboard_database_snapshot.csv"
OUT_MD       = "outputs/leaderboard_database_snapshot.md"


# ── SQL: mirrors show_kpi_leaderboard in src/application/cli_menu.py ──────────
#
# We use the identical pivot query so metric values match the CLI leaderboard
# exactly.  Comparability fields (random_seed, sample_strategy, etc.) come from
# triage_runs directly; reviewer trigger flags and thresholds are parsed from
# config_json.

LEADERBOARD_SQL = """
SELECT
    r.run_id,
    r.created_at,
    COALESCE(r.triage_model, r.model_name)  AS triage_model,
    r.reviewer_enabled,
    COALESCE(r.reviewer_model, '')           AS reviewer_model,
    r.split_name,
    r.limit_n,
    r.runtime_seconds,
    r.embedding_model,
    r.top_k,
    r.temperature,
    r.sample_strategy,
    r.stratify_by,
    r.random_seed,
    r.limit_per_label,
    r.config_json,
    MAX(CASE WHEN m.metric_name = 'avg_seconds_per_ticket'      THEN m.metric_value END) AS avg_sec_ticket,
    MAX(CASE WHEN m.metric_name = 'reviewer_invocation_rate'    THEN m.metric_value END) AS reviewer_inv_rate,
    MAX(CASE WHEN m.metric_name = 'urgency_accuracy'            THEN m.metric_value END) AS urgency_acc,
    MAX(CASE WHEN m.metric_name = 'urgency_macro_f1'            THEN m.metric_value END) AS urgency_f1,
    MAX(CASE WHEN m.metric_name = 'topic_proxy_accuracy'        THEN m.metric_value END) AS topic_acc,
    MAX(CASE WHEN m.metric_name = 'topic_macro_f1'              THEN m.metric_value END) AS topic_f1,
    MAX(CASE WHEN m.metric_name = 'next_action_proxy_agreement' THEN m.metric_value END) AS next_action_agr,
    MAX(CASE WHEN m.metric_name = 'human_review_rate'           THEN m.metric_value END) AS human_review_rt,
    MAX(CASE WHEN m.metric_name = 'average_confidence'          THEN m.metric_value END) AS avg_conf
FROM triage_runs r
LEFT JOIN triage_metrics m ON r.run_id = m.run_id
GROUP BY
    r.run_id, r.created_at, r.triage_model, r.model_name,
    r.reviewer_enabled, r.reviewer_model,
    r.split_name, r.limit_n, r.runtime_seconds,
    r.embedding_model, r.top_k, r.temperature,
    r.sample_strategy, r.stratify_by, r.random_seed, r.limit_per_label,
    r.config_json
ORDER BY r.created_at DESC NULLS LAST
"""


def _extract_config_fields(config_json: str | None) -> dict:
    """
    Parse config_json to extract comparability fields not stored as columns.

    Returns a dict with keys:
      cfg_low_confidence, cfg_trigger_flags, cfg_disagreement_ceiling,
      cfg_urgency_disagreement_ceiling, cfg_reviewer_temp, cfg_analyzer_temp
    """
    defaults = {
        "cfg_low_confidence":                    None,
        "cfg_trigger_flags":                     None,
        "cfg_disagreement_ceiling":              None,
        "cfg_urgency_disagreement_ceiling":      None,
        "cfg_reviewer_temp":                     None,
        "cfg_analyzer_temp":                     None,
    }
    if not config_json:
        return defaults
    try:
        cfg = json.loads(config_json)
    except (json.JSONDecodeError, TypeError):
        return defaults

    thresholds   = cfg.get("thresholds",  {})
    reviewer_cfg = cfg.get("reviewer",    {})
    llm_cfg      = cfg.get("llm",         {})

    return {
        "cfg_low_confidence":               thresholds.get("low_confidence"),
        "cfg_trigger_flags":                json.dumps(reviewer_cfg.get("trigger_flags", [])),
        "cfg_disagreement_ceiling":         reviewer_cfg.get("disagreement_confidence_ceiling"),
        "cfg_urgency_disagreement_ceiling": reviewer_cfg.get("urgency_disagreement_confidence_ceiling"),
        "cfg_reviewer_temp":                reviewer_cfg.get("temperature"),
        "cfg_analyzer_temp":                llm_cfg.get("temperature"),
    }


def _fmt(val, decimals: int = 4) -> str:
    """Format float or None for display."""
    if val is None:
        return "-"
    return f"{val:.{decimals}f}"


def _bool_str(val) -> str:
    if val is None:
        return "-"
    return "yes" if val else "no"


def load_runs(db_path: str) -> list[dict]:
    """
    Open DuckDB read-only and return all leaderboard rows as a list of dicts.
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "triage_runs" not in tables or "triage_metrics" not in tables:
            print("ERROR: triage_runs or triage_metrics not found.")
            return []

        raw_rows = conn.execute(LEADERBOARD_SQL).fetchall()
        col_names = [
            "run_id", "created_at", "triage_model", "reviewer_enabled", "reviewer_model",
            "split_name", "limit_n", "runtime_seconds",
            "embedding_model", "top_k", "temperature",
            "sample_strategy", "stratify_by", "random_seed", "limit_per_label",
            "config_json",
            "avg_sec_ticket", "reviewer_inv_rate",
            "urgency_acc", "urgency_f1",
            "topic_acc", "topic_f1",
            "next_action_agr", "human_review_rt", "avg_conf",
        ]
        rows = []
        for raw in raw_rows:
            row = dict(zip(col_names, raw))
            cfg_fields = _extract_config_fields(row.pop("config_json", None))
            row.update(cfg_fields)
            rows.append(row)
    finally:
        conn.close()
    return rows


# ── CSV export ─────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "run_id", "created_at", "triage_model",
    "reviewer_enabled", "reviewer_model",
    "split_name", "limit_n",
    "runtime_seconds", "avg_sec_ticket",
    "reviewer_inv_rate",
    "urgency_acc", "urgency_f1",
    "topic_acc", "topic_f1",
    "next_action_agr", "human_review_rt", "avg_conf",
    # comparability fields
    "embedding_model", "top_k", "temperature",
    "sample_strategy", "stratify_by", "random_seed", "limit_per_label",
    "cfg_low_confidence", "cfg_trigger_flags",
    "cfg_disagreement_ceiling", "cfg_urgency_disagreement_ceiling",
    "cfg_reviewer_temp", "cfg_analyzer_temp",
]


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Markdown helpers ────────────────────────────────────────────────────────────

def _md_row(r: dict) -> str:
    """Render one run as a compact markdown block."""
    lines = [
        f"### {r['run_id']}",
        f"- **created_at**        : {r['created_at']}",
        f"- **triage_model**      : {r['triage_model']}",
        f"- **reviewer**          : {_bool_str(r['reviewer_enabled'])}  "
        f"model={r['reviewer_model'] or '-'}",
        f"- **split / tickets**   : {r['split_name']} / {r['limit_n']}",
        f"- **runtime_s / avg_s** : {_fmt(r['runtime_seconds'], 1)} / {_fmt(r['avg_sec_ticket'])}",
        f"- **reviewer_inv_rate** : {_fmt(r['reviewer_inv_rate'])}",
        f"- **urgency_acc / f1**  : {_fmt(r['urgency_acc'])} / {_fmt(r['urgency_f1'])}",
        f"- **topic_acc / f1**    : {_fmt(r['topic_acc'])} / {_fmt(r['topic_f1'])}",
        f"- **next_action_agr**   : {_fmt(r['next_action_agr'])}",
        f"- **human_review_rt**   : {_fmt(r['human_review_rt'])}",
        f"- **avg_conf**          : {_fmt(r['avg_conf'])}",
        f"- **embedding_model**   : {r['embedding_model'] or '-'}",
        f"- **top_k**             : {r['top_k']}",
        f"- **sample_strategy**   : {r['sample_strategy'] or '-'}  "
        f"seed={r['random_seed']}  stratify={r['stratify_by'] or '-'}",
        f"- **low_conf_threshold**: {_fmt(r['cfg_low_confidence'])}",
        f"- **trigger_flags**     : {r['cfg_trigger_flags'] or '-'}",
        f"- **disagree_ceiling**  : {_fmt(r['cfg_disagreement_ceiling'])} / "
        f"{_fmt(r['cfg_urgency_disagreement_ceiling'])} (urgency)",
        "",
    ]
    return "\n".join(lines)


def _is_complete_eval_200(r: dict) -> bool:
    """True if run is a comparable 200-ticket eval run with full metrics."""
    return (
        r.get("split_name") == "eval"
        and r.get("limit_n") == 200
        and r.get("urgency_acc") is not None
        and r.get("urgency_f1")  is not None
        and r.get("topic_acc")   is not None
        and r.get("topic_f1")    is not None
    )


def _is_pilot(r: dict) -> bool:
    """True if run looks like a pilot, smoke, or incomplete run."""
    limit = r.get("limit_n") or 0
    if limit in (10, 20):
        return True
    # Missing core metrics
    if r.get("urgency_acc") is None or r.get("topic_acc") is None:
        return True
    # Zero confidence
    if r.get("avg_conf") == 0.0:
        return True
    return False


def _config_key(r: dict) -> tuple:
    """Tuple used to detect duplicate configurations."""
    return (
        r.get("triage_model") or "",
        _bool_str(r.get("reviewer_enabled")),
        r.get("reviewer_model") or "",
        r.get("split_name") or "",
        r.get("limit_n"),
        r.get("embedding_model") or "",
        r.get("top_k"),
        r.get("cfg_low_confidence"),
        r.get("cfg_trigger_flags") or "",
        r.get("sample_strategy") or "",
        r.get("random_seed"),
    )


def _suggest_candidates(rows: list[dict]) -> list[dict]:
    """
    Return up to 10 candidate runs for a curated leaderboard.

    Strategy:
      1. From comparable 200-ticket eval runs only.
      2. Include the best-metric analyzer-only baseline (reviewer=no).
      3. Include the same analyzer with reviewer enabled (reviewer=yes) if available.
      4. Include the best run per unique triage_model (analyzer-only).
      5. Include the best run per unique reviewer_model (reviewer-on).
      6. Prefer the run with the highest topic_acc when there are ties.
      7. Cap at 10.
    """
    eval_200 = [r for r in rows if _is_complete_eval_200(r)]
    if not eval_200:
        return []

    def score(r: dict) -> float:
        return (
            (r.get("urgency_acc") or 0) * 0.25
            + (r.get("urgency_f1") or 0) * 0.25
            + (r.get("topic_acc")  or 0) * 0.25
            + (r.get("topic_f1")   or 0) * 0.25
        )

    # Best analyzer-only baseline (reviewer disabled)
    no_reviewer = [r for r in eval_200 if not r.get("reviewer_enabled")]
    yes_reviewer = [r for r in eval_200 if r.get("reviewer_enabled")]

    candidates: list[dict] = []
    seen_run_ids: set = set()

    def add(r: dict, reason: str) -> None:
        if r["run_id"] not in seen_run_ids:
            seen_run_ids.add(r["run_id"])
            r = dict(r)
            r["_candidate_reason"] = reason
            candidates.append(r)

    # Best analyzer-only baseline
    if no_reviewer:
        best_baseline = max(no_reviewer, key=score)
        add(best_baseline, "Best analyzer-only baseline (highest avg KPI score)")

    # Same triage_model with reviewer enabled (if baseline exists)
    if candidates and yes_reviewer:
        baseline_model = candidates[0].get("triage_model", "")
        same_model_reviewed = [
            r for r in yes_reviewer if r.get("triage_model") == baseline_model
        ]
        if same_model_reviewed:
            best_reviewed = max(same_model_reviewed, key=score)
            add(best_reviewed, f"Best reviewer run for baseline model {baseline_model!r}")

    # Best run per unique triage_model (analyzer-only)
    seen_models: set = set()
    for r in sorted(no_reviewer, key=score, reverse=True):
        m = r.get("triage_model", "")
        if m not in seen_models:
            seen_models.add(m)
            add(r, f"Best analyzer-only run for model {m!r}")
        if len(candidates) >= 7:
            break

    # Best run per unique reviewer_model (reviewer-on)
    seen_rev_models: set = set()
    for r in sorted(yes_reviewer, key=score, reverse=True):
        rm = r.get("reviewer_model", "")
        if rm not in seen_rev_models:
            seen_rev_models.add(rm)
            add(r, f"Best reviewer run using reviewer model {rm!r}")
        if len(candidates) >= 10:
            break

    return candidates[:10]


def write_markdown(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total   = len(rows)
    eval200 = [r for r in rows if _is_complete_eval_200(r)]
    pilots  = [r for r in rows if _is_pilot(r)]

    # Duplicates: group by config key
    from collections import defaultdict
    config_groups: dict = defaultdict(list)
    for r in rows:
        config_groups[_config_key(r)].append(r["run_id"])
    duplicate_groups = {k: v for k, v in config_groups.items() if len(v) > 1}

    candidates = _suggest_candidates(rows)

    lines: list[str] = []

    lines += [
        "# Leaderboard Database Snapshot",
        "",
        f"Generated: {now}  ",
        f"Source DB: `{os.path.abspath(DB_PATH)}`  ",
        f"Total runs in DB: {total}  ",
        f"Comparable 200-ticket eval runs: {len(eval200)}  ",
        f"Pilot / smoke / incomplete: {len(pilots)}  ",
        "",
        "---",
        "",
        "## All Runs (newest first)",
        "",
    ]

    for r in rows:
        lines.append(_md_row(r))

    lines += [
        "---",
        "",
        "## Comparable 200-ticket eval runs",
        "",
        "Criteria: split=eval, limit_n=200, urgency and topic metrics present.",
        "",
    ]
    if eval200:
        for r in eval200:
            lines.append(_md_row(r))
    else:
        lines.append("_No comparable runs found._\n")

    lines += [
        "---",
        "",
        "## Pilot / smoke / incomplete runs",
        "",
        "Criteria: limit_n in {10, 20}, or missing core metrics, or avg_conf=0.",
        "",
    ]
    if pilots:
        for r in pilots:
            lines.append(_md_row(r))
    else:
        lines.append("_No pilot or incomplete runs found._\n")

    lines += [
        "---",
        "",
        "## Potential duplicate configurations",
        "",
        "Groups share: triage_model, reviewer state, reviewer_model, split, "
        "limit_n, embedding_model, top_k, low_confidence threshold, "
        "trigger_flags, sample_strategy, random_seed.",
        "",
    ]
    if duplicate_groups:
        for key, run_ids in sorted(duplicate_groups.items(), key=lambda x: -len(x[1])):
            (
                triage_model, reviewer, reviewer_model, split, limit_n,
                embedding_model, top_k, low_conf, trigger_flags,
                sample_strategy, random_seed,
            ) = key
            lines += [
                f"### Duplicate group ({len(run_ids)} runs)",
                f"- triage_model={triage_model}  reviewer={reviewer}  "
                f"reviewer_model={reviewer_model or '-'}",
                f"- split={split}  limit_n={limit_n}  "
                f"embedding_model={embedding_model or '-'}  top_k={top_k}",
                f"- low_conf={low_conf}  trigger_flags={trigger_flags or '-'}",
                f"- sample_strategy={sample_strategy}  seed={random_seed}",
                f"- run_ids:",
            ]
            for rid in run_ids:
                lines.append(f"  - {rid}")
            lines.append("")
    else:
        lines.append("_No duplicate configurations found._\n")

    lines += [
        "---",
        "",
        "## Recommendation: candidate runs for the curated leaderboard",
        "",
        "Suggested 6–10 runs for a curated leaderboard.  ",
        "Selection criteria: comparable 200-ticket eval runs, best average KPI score, "
        "diverse triage models, diverse reviewer models, no uninformative retries.",
        "",
    ]
    if candidates:
        for i, r in enumerate(candidates, 1):
            reason = r.get("_candidate_reason", "")
            lines += [
                f"### Candidate {i}: {r['run_id']}",
                f"- **Why**: {reason}",
                f"- triage_model={r['triage_model']}  reviewer={_bool_str(r['reviewer_enabled'])}  "
                f"reviewer_model={r['reviewer_model'] or '-'}",
                f"- urgency_acc={_fmt(r['urgency_acc'])}  urgency_f1={_fmt(r['urgency_f1'])}  "
                f"topic_acc={_fmt(r['topic_acc'])}  topic_f1={_fmt(r['topic_f1'])}",
                f"- next_action_agr={_fmt(r['next_action_agr'])}  "
                f"human_review_rt={_fmt(r['human_review_rt'])}  avg_conf={_fmt(r['avg_conf'])}",
                "",
            ]
    else:
        lines += [
            "_No comparable 200-ticket eval runs found. "
            "Run at least one batch evaluation (option 3) before generating recommendations._",
            "",
        ]

    lines += [
        "---",
        "",
        "## Confirmation",
        "",
        "- Database file was opened in **read-only** mode.",
        "- No rows were inserted, updated, deleted, or truncated.",
        "- No application source files were modified.",
        "- No LLM inference was run.",
        "",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(DB_PATH):
        print(f"ERROR: DuckDB not found at {os.path.abspath(DB_PATH)}")
        sys.exit(1)

    print(f"Opening {DB_PATH} in read-only mode …")
    rows = load_runs(DB_PATH)

    if not rows:
        print("No runs found in triage_runs. Nothing to export.")
        sys.exit(0)

    print(f"Loaded {len(rows)} run(s).")

    write_csv(rows, OUT_CSV)
    write_markdown(rows, OUT_MD)

    abs_csv = os.path.abspath(OUT_CSV)
    abs_md  = os.path.abspath(OUT_MD)

    print()
    print(f"CSV  : {abs_csv}")
    print(f"MD   : {abs_md}")
    print()
    print("Confirmation: database and application source files were NOT modified.")


if __name__ == "__main__":
    main()
