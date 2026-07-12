"""
Sprint 6B — Evaluation Repository.

Manages four DuckDB tables that store batch run results and evaluation KPIs:

  triage_runs            — one row per batch run (config snapshot + timing)
  triage_predictions     — one row per processed ticket (includes step timings)
  triage_metrics         — one row per scalar KPI per run
  triage_confusion_matrix — one row per (actual, predicted) pair per target

All four tables are created in the existing data/tickets.duckdb database.

Migration note:
  If triage_predictions already exists from a prior run without timing columns,
  create_tables() adds the three timing columns using ALTER TABLE ADD COLUMN
  IF NOT EXISTS (DuckDB 0.10+).
"""

import json
from datetime import datetime, timezone

import duckdb


class EvaluationRepository:
    """
    Write evaluation artifacts for a single batch run into DuckDB.

    Parameters
    ----------
    db_path:
        Path to the existing DuckDB file (data/tickets.duckdb).
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)

    def create_tables(self) -> None:
        """Create all four evaluation tables if they do not already exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS triage_runs (
                run_id           TEXT PRIMARY KEY,
                created_at       TIMESTAMP,
                model_name       TEXT,
                embedding_model  TEXT,
                dataset_path     TEXT,
                split_name       TEXT,
                limit_n          INTEGER,
                top_k            INTEGER,
                temperature      DOUBLE,
                max_retries      INTEGER,
                runtime_seconds  DOUBLE,
                config_json      TEXT,
                sample_strategy  TEXT,
                stratify_by      TEXT,
                random_seed      INTEGER,
                limit_per_label  INTEGER
            )
        """)

        # Backward-compatible migration for databases created before this sprint.
        _sampling_cols = [
            ("sample_strategy", "TEXT"),
            ("stratify_by",     "TEXT"),
            ("random_seed",     "INTEGER"),
            ("limit_per_label", "INTEGER"),
        ]
        for col, dtype in _sampling_cols:
            try:
                self.conn.execute(
                    f"ALTER TABLE triage_runs ADD COLUMN IF NOT EXISTS {col} {dtype}"
                )
            except Exception:
                pass  # column already present

        # Backward-compatible migration: add reviewer identity columns.
        # triage_model  — the primary analyzer model name (mirrors model_name).
        # reviewer_enabled — whether the reviewer loop was active for this run.
        # reviewer_model   — reviewer model name; empty string when disabled.
        _reviewer_run_cols = [
            ("triage_model",     "TEXT"),
            ("reviewer_enabled", "BOOLEAN"),
            ("reviewer_model",   "TEXT"),
        ]
        for col, dtype in _reviewer_run_cols:
            try:
                self.conn.execute(
                    f"ALTER TABLE triage_runs ADD COLUMN IF NOT EXISTS {col} {dtype}"
                )
            except Exception:
                pass  # column already present

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS triage_predictions (
                run_id                  TEXT,
                ticket_id               TEXT,
                text_snippet            TEXT,
                predicted_topic         TEXT,
                predicted_urgency       TEXT,
                predicted_next_action   TEXT,
                confidence              DOUBLE,
                missing_info            BOOLEAN,
                requires_human_review   BOOLEAN,
                short_note              TEXT,
                action_status           TEXT,
                action_target           TEXT,
                action_note             TEXT,
                actual_queue            TEXT,
                actual_priority         TEXT,
                actual_type             TEXT,
                proxy_topic             TEXT,
                proxy_urgency           TEXT,
                proxy_next_action       TEXT,
                proxy_topic_source      TEXT,
                retrieval_seconds       DOUBLE,
                llm_seconds             DOUBLE,
                total_ticket_seconds    DOUBLE
            )
        """)

        # Backward-compatible migration: add timing columns to tables created
        # before this sprint.  DuckDB raises if the column already exists, so
        # we suppress that specific error.
        _timing_cols = [
            ("retrieval_seconds",    "DOUBLE"),
            ("llm_seconds",          "DOUBLE"),
            ("total_ticket_seconds", "DOUBLE"),
        ]
        for col, dtype in _timing_cols:
            try:
                self.conn.execute(
                    f"ALTER TABLE triage_predictions ADD COLUMN IF NOT EXISTS {col} {dtype}"
                )
            except Exception:
                pass  # column already present or DB does not support IF NOT EXISTS

        # Backward-compatible migration: add reviewer columns.
        # These are absent in databases created before the reviewer sprint.
        _reviewer_cols = [
            ("reviewer_used",            "BOOLEAN"),
            ("reviewer_model",           "TEXT"),
            ("reviewer_changed_topic",   "BOOLEAN"),
            ("reviewer_changed_urgency", "BOOLEAN"),
            ("reviewer_seconds",         "DOUBLE"),
            ("first_topic",              "TEXT"),
            ("first_urgency",            "TEXT"),
            ("first_confidence",         "DOUBLE"),
            ("reviewer_trigger_flags",   "TEXT"),
        ]
        for col, dtype in _reviewer_cols:
            try:
                self.conn.execute(
                    f"ALTER TABLE triage_predictions ADD COLUMN IF NOT EXISTS {col} {dtype}"
                )
            except Exception:
                pass  # column already present

        # Backward-compatible migration: add explainability columns.
        # These are absent in databases created before this sprint.
        _explainability_cols = [
            ("first_short_note",           "TEXT"),
            ("reviewer_note",              "TEXT"),
            ("validator_flags",            "TEXT"),
            ("validator_notes",            "TEXT"),
            ("neighbor_predicted_topic",   "TEXT"),
            ("neighbor_topic_confidence",  "DOUBLE"),
            ("neighbor_predicted_priority", "TEXT"),
            ("neighbor_priority_confidence", "DOUBLE"),
        ]
        for col, dtype in _explainability_cols:
            try:
                self.conn.execute(
                    f"ALTER TABLE triage_predictions ADD COLUMN IF NOT EXISTS {col} {dtype}"
                )
            except Exception:
                pass  # column already present

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS triage_metrics (
                run_id        TEXT,
                metric_group  TEXT,
                metric_name   TEXT,
                metric_value  DOUBLE,
                created_at    TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS triage_confusion_matrix (
                run_id           TEXT,
                target_name      TEXT,
                actual_label     TEXT,
                predicted_label  TEXT,
                count            INTEGER
            )
        """)

    def insert_run(self, run_metadata: dict) -> None:
        """
        Insert one row into triage_runs.

        Uses INSERT OR REPLACE because run_id is the PRIMARY KEY.
        Re-running the same second would silently replace the previous entry.
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO triage_runs (
                run_id, created_at, model_name, embedding_model,
                dataset_path, split_name, limit_n, top_k,
                temperature, max_retries, runtime_seconds, config_json,
                sample_strategy, stratify_by, random_seed, limit_per_label,
                triage_model, reviewer_enabled, reviewer_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_metadata.get("run_id", ""),
                run_metadata.get("created_at", datetime.now(timezone.utc)),
                run_metadata.get("model_name", ""),
                run_metadata.get("embedding_model", ""),
                run_metadata.get("dataset_path", ""),
                run_metadata.get("split_name", ""),
                run_metadata.get("limit_n", 0),
                run_metadata.get("top_k", 0),
                run_metadata.get("temperature", 0.0),
                run_metadata.get("max_retries", 0),
                run_metadata.get("runtime_seconds", 0.0),
                run_metadata.get("config_json", "{}"),
                run_metadata.get("sample_strategy", "natural"),
                run_metadata.get("stratify_by", ""),
                run_metadata.get("random_seed", 42),
                run_metadata.get("limit_per_label", 0),
                run_metadata.get("triage_model", run_metadata.get("model_name", "")),
                bool(run_metadata.get("reviewer_enabled", False)),
                run_metadata.get("reviewer_model", ""),
            ],
        )

    def insert_predictions(self, run_id: str, rows: list[dict]) -> None:
        """
        Insert one row per processed ticket into triage_predictions.

        Maps result row keys (topic, urgency, next_action) to the
        predicted_* column names used in the table schema.
        Timing columns and reviewer trace columns are included with safe
        defaults when absent (e.g. when a non-timed agent is used).
        """
        for row in rows:
            self.conn.execute(
                """
                INSERT INTO triage_predictions (
                    run_id, ticket_id, text_snippet,
                    predicted_topic, predicted_urgency, predicted_next_action,
                    confidence, missing_info, requires_human_review, short_note,
                    action_status, action_target, action_note,
                    actual_queue, actual_priority, actual_type,
                    proxy_topic, proxy_urgency, proxy_next_action, proxy_topic_source,
                    retrieval_seconds, llm_seconds, total_ticket_seconds,
                    reviewer_used, reviewer_model,
                    reviewer_changed_topic, reviewer_changed_urgency,
                    reviewer_seconds,
                    first_topic, first_urgency, first_confidence,
                    reviewer_trigger_flags,
                    first_short_note, reviewer_note,
                    validator_flags, validator_notes,
                    neighbor_predicted_topic, neighbor_topic_confidence,
                    neighbor_predicted_priority, neighbor_priority_confidence
                ) VALUES (
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?,
                    ?, ?, ?,
                    ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?
                )
                """,
                [
                    run_id,
                    row.get("ticket_id", ""),
                    row.get("text_snippet", ""),
                    row.get("topic", ""),
                    row.get("urgency", ""),
                    row.get("next_action", ""),
                    float(row.get("confidence", 0.0)),
                    bool(row.get("missing_info", False)),
                    bool(row.get("requires_human_review", False)),
                    row.get("short_note", ""),
                    row.get("action_status", ""),
                    row.get("action_target", ""),
                    row.get("action_note", ""),
                    row.get("actual_queue", ""),
                    row.get("actual_priority", ""),
                    row.get("actual_type", ""),
                    row.get("proxy_topic", ""),
                    row.get("proxy_urgency", ""),
                    row.get("proxy_next_action", ""),
                    row.get("proxy_topic_source", ""),
                    float(row.get("retrieval_seconds", 0.0)),
                    float(row.get("llm_seconds", 0.0)),
                    float(row.get("total_ticket_seconds", 0.0)),
                    bool(row.get("reviewer_used", False)),
                    str(row.get("reviewer_model", "")),
                    bool(row.get("reviewer_changed_topic", False)),
                    bool(row.get("reviewer_changed_urgency", False)),
                    float(row.get("reviewer_seconds", 0.0)),
                    str(row.get("first_topic", "")),
                    str(row.get("first_urgency", "")),
                    float(row.get("first_confidence", 0.0)),
                    str(row.get("reviewer_trigger_flags", "[]")),
                    str(row.get("first_short_note", "")),
                    str(row.get("reviewer_note", "")),
                    str(row.get("validator_flags", "[]")),
                    str(row.get("validator_notes", "[]")),
                    str(row.get("neighbor_predicted_topic", "")),
                    float(row.get("neighbor_topic_confidence", 0.0)),
                    str(row.get("neighbor_predicted_priority", "")),
                    float(row.get("neighbor_priority_confidence", 0.0)),
                ],
            )

    def insert_metrics(self, run_id: str, metrics: dict) -> None:
        """
        Insert one row per scalar KPI into triage_metrics.

        The metric_group is inferred from the metric name prefix.
        Non-float values (e.g. nested dicts) are skipped.
        """
        now = datetime.now(timezone.utc)
        for name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            group = _metric_group(name)
            self.conn.execute(
                """
                INSERT INTO triage_metrics (
                    run_id, metric_group, metric_name, metric_value, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [run_id, group, name, float(value), now],
            )

    def insert_confusion_matrix(
        self,
        run_id: str,
        target_name: str,
        confusion_rows: list[dict],
    ) -> None:
        """
        Insert confusion matrix rows for one prediction target.

        Parameters
        ----------
        run_id:
            Current batch run identifier.
        target_name:
            Name of the prediction target (e.g. "urgency", "topic_proxy").
        confusion_rows:
            Output of confusion_counts() — list of dicts with
            actual_label, predicted_label, count.
        """
        for crow in confusion_rows:
            self.conn.execute(
                """
                INSERT INTO triage_confusion_matrix (
                    run_id, target_name, actual_label, predicted_label, count
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    target_name,
                    crow.get("actual_label", ""),
                    crow.get("predicted_label", ""),
                    int(crow.get("count", 0)),
                ],
            )

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()


# ── Helper ────────────────────────────────────────────────────────────────────

def _metric_group(name: str) -> str:
    """Infer the metric group from the metric name prefix."""
    if name.startswith("urgency"):
        return "urgency"
    if name.startswith("topic"):
        return "topic"
    if name.startswith("next_action"):
        return "next_action"
    return "operational"
