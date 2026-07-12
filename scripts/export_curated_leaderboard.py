"""
Read-only curated leaderboard snapshot export.

Connects to data/tickets.duckdb in READ-ONLY mode.
Does NOT modify any table, row, or sequence.

Reads the seven configured run IDs from config.yaml (leaderboard section).
Outputs:
  outputs/curated_leaderboard.csv  — one row per configured run with group label
  outputs/curated_leaderboard.md   — human-readable with methodological notes

Usage:
  python scripts/export_curated_leaderboard.py
"""

import csv
import os
import sys

import duckdb
import yaml

DB_PATH     = "data/tickets.duckdb"
CONFIG_PATH = "config.yaml"
OUT_CSV     = "outputs/curated_leaderboard.csv"
OUT_MD      = "outputs/curated_leaderboard.md"


CURATED_SQL = """
SELECT
    r.run_id,
    r.created_at,
    COALESCE(r.triage_model, r.model_name)  AS triage_model,
    r.reviewer_enabled,
    COALESCE(r.reviewer_model, '')           AS reviewer_model,
    r.limit_n,
    r.runtime_seconds,
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
WHERE r.run_id IN ({placeholders})
GROUP BY
    r.run_id, r.created_at, r.triage_model, r.model_name,
    r.reviewer_enabled, r.reviewer_model, r.limit_n, r.runtime_seconds
"""


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fmt(val, decimals: int = 4) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _pct(val) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def _bool_str(val) -> str:
    if val is None:
        return "—"
    return "yes" if val else "no"


def load_curated_rows(db_path: str, run_ids: list[str]) -> dict[str, dict]:
    """
    Open DuckDB read-only and return a dict of run rows keyed by run_id.
    Only run IDs that exist in the database are returned.
    """
    if not run_ids:
        return {}
    conn = duckdb.connect(db_path, read_only=True)
    try:
        placeholders = ", ".join("?" for _ in run_ids)
        sql = CURATED_SQL.format(placeholders=placeholders)
        raw = conn.execute(sql, run_ids).fetchall()
    finally:
        conn.close()

    col_names = [
        "run_id", "created_at", "triage_model", "reviewer_enabled",
        "reviewer_model", "limit_n", "runtime_seconds",
        "avg_sec_ticket", "reviewer_inv_rate",
        "urgency_acc", "urgency_f1", "topic_acc", "topic_f1",
        "next_action_agr", "human_review_rt", "avg_conf",
    ]
    return {row[0]: dict(zip(col_names, row)) for row in raw}


CSV_COLUMNS = [
    "group", "run_id", "triage_model", "reviewer_enabled", "reviewer_model",
    "limit_n", "runtime_seconds", "avg_sec_ticket", "reviewer_inv_rate",
    "urgency_acc", "urgency_f1", "topic_acc", "topic_f1",
    "next_action_agr", "human_review_rt", "avg_conf",
]


def write_csv(
    grouped_rows: list[tuple[str, dict]],
    path: str,
) -> None:
    """Write curated CSV with a 'group' label column."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for group_label, row in grouped_rows:
            out_row = dict(row)
            out_row["group"] = group_label
            writer.writerow(out_row)


def write_markdown(
    grouped_rows: list[tuple[str, dict]],
    featured_id: str,
    path: str,
) -> None:
    """Write curated markdown with methodological notes."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines: list[str] = [
        "# Curated Evaluation Leaderboard — HDI Ticket Triage Agent",
        "",
        "This file contains only the seven configured runs grouped by methodology.",
        "Use `outputs/leaderboard_database_snapshot.md` to inspect all stored runs.",
        "",
        "Database was opened in **read-only** mode. No rows were modified.",
        "",
        "---",
        "",
        "## Section A: Controlled Reviewer A/B — Same Analyzer and Ticket Sample",
        "",
        "_Note: A and B use the same llama3.2:3b analyzer, 200-ticket balanced eval_",
        "_sample, seed 42, embedding model, retrieval configuration, thresholds, and_",
        "_trigger flags. Only the conditional reviewer differs._",
        "",
    ]

    current_group = None
    for group_label, row in grouped_rows:
        if group_label != current_group:
            if current_group is not None:
                lines += ["", "---", ""]
            if group_label == "controlled_reviewer_ab":
                pass  # Already printed Section A header above
            elif group_label == "best_observed_reviewer":
                lines += [
                    "## Section B: Best Observed Reviewer Configuration",
                    "",
                ]
            elif group_label == "analyzer_screening":
                lines += [
                    "## Section C: Historical Analyzer Screening — 200 Tickets",
                    "",
                    "_Note: These runs document model screening. Some early runs predate_",
                    "_complete sampling metadata, so they are not presented as a strict_",
                    "_controlled A/B experiment._",
                    "",
                ]
            current_group = group_label

        rid      = row["run_id"]
        is_feat  = (rid == featured_id)
        star     = " ★ Best observed reviewer configuration" if is_feat else ""
        rev_str  = _bool_str(row.get("reviewer_enabled"))
        rev_mod  = row.get("reviewer_model") or "—"

        lines += [
            f"### {rid}{star}",
            f"- **triage_model**      : {row.get('triage_model', '—')}",
            f"- **reviewer**          : {rev_str}  model={rev_mod}",
            f"- **tickets (N)**       : {row.get('limit_n', '—')}",
            f"- **sec/ticket**        : {_fmt(row.get('avg_sec_ticket'), 3)}",
            f"- **reviewer_rate**     : {_pct(row.get('reviewer_inv_rate'))}",
            f"- **urgency_acc / f1**  : {_pct(row.get('urgency_acc'))} / {_pct(row.get('urgency_f1'))}",
            f"- **topic_acc / f1**    : {_pct(row.get('topic_acc'))} / {_pct(row.get('topic_f1'))}",
            f"- **action_agreement**  : {_pct(row.get('next_action_agr'))}",
            f"- **human_review_rate** : {_pct(row.get('human_review_rt'))}",
            f"- **avg_confidence**    : {_pct(row.get('avg_conf'))}",
            "",
        ]

    lines += [
        "---",
        "",
        "Use 'Full experiment history' in the CLI to inspect all stored pilot,",
        "smoke-test, failed, and repeated runs.",
        "",
        "---",
        "",
        "## Confirmation",
        "",
        "- Database opened in **read-only** mode.",
        "- No rows were inserted, updated, deleted, or truncated.",
        "- No application source files were modified.",
        "- No LLM inference was run.",
        "",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    if not os.path.exists(DB_PATH):
        print(f"ERROR: DuckDB not found at {os.path.abspath(DB_PATH)}")
        sys.exit(1)

    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: config.yaml not found at {os.path.abspath(CONFIG_PATH)}")
        sys.exit(1)

    cfg         = _load_config(CONFIG_PATH)
    lb_cfg      = cfg.get("leaderboard", {})
    if not lb_cfg:
        print("ERROR: 'leaderboard' section not found in config.yaml")
        sys.exit(1)

    controlled_ids = lb_cfg.get("controlled_reviewer_run_ids", [])
    featured_ids   = lb_cfg.get("featured_reviewer_run_ids", [])
    screening_ids  = lb_cfg.get("analyzer_screening_run_ids", [])
    featured_id    = lb_cfg.get("featured_run_id", "")

    all_ids = list(controlled_ids) + list(featured_ids) + list(screening_ids)
    print(f"Configured run IDs: {len(all_ids)}")

    print(f"Opening {DB_PATH} in read-only mode …")
    db_rows = load_curated_rows(DB_PATH, all_ids)
    print(f"Found {len(db_rows)} of {len(all_ids)} configured runs in DuckDB.")

    for rid in all_ids:
        if rid not in db_rows:
            print(f"  Warning: '{rid}' not found in DuckDB — skipping.")

    # Build ordered list of (group_label, row_dict)
    grouped: list[tuple[str, dict]] = []
    for rid in controlled_ids:
        if rid in db_rows:
            grouped.append(("controlled_reviewer_ab", db_rows[rid]))
    for rid in featured_ids:
        if rid in db_rows:
            grouped.append(("best_observed_reviewer", db_rows[rid]))
    for rid in screening_ids:
        if rid in db_rows:
            grouped.append(("analyzer_screening", db_rows[rid]))

    if not grouped:
        print("No configured runs found in DuckDB. Nothing to export.")
        sys.exit(0)

    write_csv(grouped, OUT_CSV)
    write_markdown(grouped, featured_id, OUT_MD)

    print()
    print(f"CSV : {os.path.abspath(OUT_CSV)}")
    print(f"MD  : {os.path.abspath(OUT_MD)}")
    print()
    print(f"Rows written: {len(grouped)}")
    print("Confirmation: database and application source files were NOT modified.")


if __name__ == "__main__":
    main()
