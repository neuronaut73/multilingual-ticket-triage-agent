"""
Tests for Sprint 6C.6 / 6C.7 — cli_menu.py and cli.py.

Fast unit tests only — no Ollama, no embedding model, no LanceDB calls.

All DuckDB tests use tmp_path to create a temporary database file.
The helpers seed only the tables and columns needed for each tested function.

Coverage — cli_menu.py:
  - show_config prints expected keys
  - show_analyzer_prompt prints role and schema keys
  - show_reviewer_prompt with reviewer enabled and disabled
  - lookup_ticket finds an existing row and reports missing cleanly
  - lookup_prediction finds a stored prediction and reports missing cleanly
  - show_kpi_leaderboard handles missing tables and a seeded run
  - show_run_details handles missing run_id and a seeded run
  - show_confusion_matrix handles missing data and a seeded matrix
  - run_leakage_audit prints audit sections without crashing
  - run_submission_checklist reports missing artifacts as failures

Coverage — cli.py navigation (Sprint 6C.7):
  - Top-level menu contains only options 0–6
  - Removed top-level items are no longer displayed
  - Option 1 opens the ticket submenu
  - Random-ticket and specific-ticket dispatch
  - Reviewer-off and conditional-reviewer selections (per-operation only)
  - Returning from submenu does not exit the whole application
  - Options 2, 3, 4, 5, 6 delegate to correct functions
  - Evaluation submenu delegates to leaderboard, run details, confusion matrix
  - Ticket prediction details reuses lookup_ticket + lookup_prediction
  - Invalid and empty input do not crash
  - Reviewer selection does not mutate cfg or write config.yaml
"""

import copy
import json

import duckdb
import pytest

import src.application.cli_menu as _cli_menu_mod
from src.application.cli_menu import (
    AB_COMPARISON_KPIS,
    _sample_tickets,
    apply_reviewer_override,
    lookup_prediction,
    lookup_ticket,
    print_effective_run_config,
    prompt_reviewer_override,
    run_leakage_audit,
    run_reviewer_ab_comparison,
    run_submission_checklist,
    show_analyzer_prompt,
    show_config,
    show_confusion_matrix,
    show_kpi_leaderboard,
    show_reviewer_prompt,
    show_run_details,
    toggle_reviewer_session_override,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_empty_db(tmp_path) -> str:
    """Return path to a valid empty DuckDB file (no tables)."""
    path = str(tmp_path / "empty.duckdb")
    conn = duckdb.connect(path)
    conn.close()
    return path


def _make_tickets_db(tmp_path) -> tuple[str, str]:
    """
    Return (db_path, ticket_id) for a DuckDB with one seeded historical ticket.

    Creates historical_tickets with the minimal schema expected by lookup_ticket.
    """
    path      = str(tmp_path / "tickets.duckdb")
    ticket_id = "abc123def456"
    conn = duckdb.connect(path)
    conn.execute("""
        CREATE TABLE historical_tickets (
            ticket_id              TEXT PRIMARY KEY,
            split_name             TEXT,
            subject                TEXT,
            body                   TEXT,
            raw_text               TEXT,
            cleaned_text           TEXT,
            representation_text    TEXT,
            text_snippet           TEXT,
            actual_queue           TEXT,
            actual_priority        TEXT,
            actual_type            TEXT,
            actual_tags_json       TEXT,
            language               TEXT,
            proxy_topic            TEXT,
            proxy_urgency          TEXT,
            proxy_next_action      TEXT,
            proxy_topic_source     TEXT,
            source_row_json        TEXT
        )
    """)
    conn.execute(
        """
        INSERT INTO historical_tickets VALUES (
            ?, 'eval', 'Login fails', 'I cannot log in since yesterday.',
            'Login fails I cannot log in since yesterday.',
            'login fails i cannot log in since yesterday',
            'Subject: Login fails\n\nBody: I cannot log in since yesterday.',
            'Login fails I cannot log in',
            'Technical Support', 'high', 'Incident',
            '["portal", "login"]', 'en',
            'Technical / Online Access', 'High', 'forward_to_technical_support',
            'queue_mapping', '{}'
        )
        """,
        [ticket_id],
    )
    conn.close()
    return path, ticket_id


def _make_eval_db_with_prediction(tmp_path) -> tuple[str, str, str]:
    """
    Return (db_path, ticket_id, run_id) for a DuckDB with a seeded prediction.

    Creates triage_predictions table and inserts one row.
    """
    path      = str(tmp_path / "eval.duckdb")
    ticket_id = "aabbccddeeff"
    run_id    = "run_20260101_120000"
    conn = duckdb.connect(path)
    conn.execute("""
        CREATE TABLE triage_predictions (
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
    conn.execute(
        """
        INSERT INTO triage_predictions VALUES (
            ?, ?, 'Login fails.', 'Technical / Online Access', 'High',
            'escalate_to_human_supervisor', 0.82, true, true,
            'Login issue.', 'simulated_success', 'supervisor_queue', 'Escalated.',
            'Technical Support', 'high', 'Incident',
            'Technical / Online Access', 'High', 'forward_to_technical_support',
            'queue_mapping', 1.1, 3.2, 4.5
        )
        """,
        [run_id, ticket_id],
    )
    conn.close()
    return path, ticket_id, run_id


def _make_runs_db(tmp_path) -> tuple[str, str]:
    """
    Return (db_path, run_id) for a DuckDB with seeded triage_runs and triage_metrics.

    Includes the reviewer identity columns (triage_model, reviewer_enabled,
    reviewer_model) so show_kpi_leaderboard can query them without error.
    """
    path   = str(tmp_path / "runs.duckdb")
    run_id = "run_20260101_120000"
    conn   = duckdb.connect(path)
    conn.execute("""
        CREATE TABLE triage_runs (
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
            triage_model     TEXT,
            reviewer_enabled BOOLEAN,
            reviewer_model   TEXT
        )
    """)
    conn.execute(
        "INSERT INTO triage_runs VALUES ("
        "?, NOW(), 'qwen2.5:7b', 'e5-large', "
        "'data/tickets.csv', 'eval', 10, 5, 0.1, 1, 42.5, '{}', "
        "'qwen2.5:7b', false, '')",
        [run_id],
    )
    conn.execute("""
        CREATE TABLE triage_metrics (
            run_id        TEXT,
            metric_group  TEXT,
            metric_name   TEXT,
            metric_value  DOUBLE,
            created_at    TIMESTAMP
        )
    """)
    for name, value in [
        ("urgency_accuracy", 0.75),
        ("topic_proxy_accuracy", 0.60),
        ("human_review_rate", 0.20),
    ]:
        conn.execute(
            "INSERT INTO triage_metrics VALUES (?, 'group', ?, ?, NOW())",
            [run_id, name, value],
        )
    conn.close()
    return path, run_id


def _make_confusion_db(tmp_path) -> tuple[str, str]:
    """
    Return (db_path, run_id) for a DuckDB with seeded triage_confusion_matrix.
    """
    path   = str(tmp_path / "confusion.duckdb")
    run_id = "run_20260101_120000"
    conn   = duckdb.connect(path)
    conn.execute("""
        CREATE TABLE triage_confusion_matrix (
            run_id           TEXT,
            target_name      TEXT,
            actual_label     TEXT,
            predicted_label  TEXT,
            count            INTEGER
        )
    """)
    for actual, predicted, count in [
        ("High", "High", 5),
        ("High", "Low",  1),
        ("Low",  "Low",  4),
    ]:
        conn.execute(
            "INSERT INTO triage_confusion_matrix VALUES (?, 'urgency', ?, ?, ?)",
            [run_id, actual, predicted, count],
        )
    conn.close()
    return path, run_id


# ── show_config ───────────────────────────────────────────────────────────────

class TestShowConfig:

    def test_prints_llm_section(self, capsys) -> None:
        cfg = {"llm": {"model_name": "qwen2.5:7b", "base_url": "http://localhost:11434"}}
        show_config(cfg)
        out = capsys.readouterr().out
        assert "llm" in out

    def test_prints_embedding_model_name(self, capsys) -> None:
        cfg = {"embedding": {"model_name": "multilingual-e5-large"}}
        show_config(cfg)
        out = capsys.readouterr().out
        assert "multilingual-e5-large" in out

    def test_prints_header_line(self, capsys) -> None:
        show_config({"k": "v"})
        out = capsys.readouterr().out
        assert "Config" in out or "config" in out


# ── show_analyzer_prompt ──────────────────────────────────────────────────────

class TestShowAnalyzerPrompt:

    def test_prints_role_line(self, capsys) -> None:
        show_analyzer_prompt()
        out = capsys.readouterr().out
        assert "ROLE" in out

    def test_prints_allowed_topics(self, capsys) -> None:
        show_analyzer_prompt()
        out = capsys.readouterr().out
        assert "Policy / Contract" in out
        assert "Claims / Damage" in out
        assert "Billing / Payment" in out
        assert "Technical / Online Access" in out

    def test_prints_json_schema(self, capsys) -> None:
        show_analyzer_prompt()
        out = capsys.readouterr().out
        assert '"topic"' in out
        assert '"urgency"' in out
        assert '"confidence"' in out

    def test_prints_urgency_values(self, capsys) -> None:
        show_analyzer_prompt()
        out = capsys.readouterr().out
        assert "Low" in out
        assert "Medium" in out
        assert "High" in out


# ── show_reviewer_prompt ──────────────────────────────────────────────────────

class TestShowReviewerPrompt:

    def test_disabled_reviewer_prints_message(self, capsys) -> None:
        cfg = {"reviewer": {"enabled": False}}
        show_reviewer_prompt(cfg)
        out = capsys.readouterr().out
        assert "disabled" in out.lower()

    def test_missing_reviewer_key_treated_as_disabled(self, capsys) -> None:
        show_reviewer_prompt({})
        out = capsys.readouterr().out
        assert "disabled" in out.lower()

    def test_enabled_reviewer_prints_role(self, capsys) -> None:
        cfg = {
            "reviewer": {
                "enabled": True,
                "model_name": "devstral:24b",
                "trigger_flags": ["low_llm_confidence"],
            }
        }
        show_reviewer_prompt(cfg)
        out = capsys.readouterr().out
        assert "ROLE" in out

    def test_enabled_reviewer_prints_trigger_flags(self, capsys) -> None:
        cfg = {
            "reviewer": {
                "enabled": True,
                "model_name": "devstral:24b",
                "trigger_flags": ["urgency_disagreement"],
            }
        }
        show_reviewer_prompt(cfg)
        out = capsys.readouterr().out
        assert "urgency_disagreement" in out

    def test_enabled_reviewer_prints_model_name(self, capsys) -> None:
        cfg = {
            "reviewer": {
                "enabled": True,
                "model_name": "test-model",
                "trigger_flags": [],
            }
        }
        show_reviewer_prompt(cfg)
        out = capsys.readouterr().out
        assert "test-model" in out


# ── lookup_ticket ─────────────────────────────────────────────────────────────

class TestLookupTicket:

    def test_found_ticket_prints_subject(self, capsys, tmp_path) -> None:
        db_path, ticket_id = _make_tickets_db(tmp_path)
        lookup_ticket(db_path, ticket_id)
        out = capsys.readouterr().out
        assert "Login fails" in out

    def test_found_ticket_prints_actual_queue(self, capsys, tmp_path) -> None:
        db_path, ticket_id = _make_tickets_db(tmp_path)
        lookup_ticket(db_path, ticket_id)
        out = capsys.readouterr().out
        assert "Technical Support" in out

    def test_found_ticket_prints_proxy_topic(self, capsys, tmp_path) -> None:
        db_path, ticket_id = _make_tickets_db(tmp_path)
        lookup_ticket(db_path, ticket_id)
        out = capsys.readouterr().out
        assert "Technical / Online Access" in out

    def test_missing_ticket_prints_not_found(self, capsys, tmp_path) -> None:
        db_path, _ = _make_tickets_db(tmp_path)
        lookup_ticket(db_path, "does_not_exist_id")
        out = capsys.readouterr().out
        assert "not found" in out.lower()

    def test_missing_db_prints_not_found(self, capsys, tmp_path) -> None:
        lookup_ticket(str(tmp_path / "nofile.duckdb"), "anyid")
        out = capsys.readouterr().out
        assert "not found" in out.lower()


# ── lookup_prediction ─────────────────────────────────────────────────────────

class TestLookupPrediction:

    def test_found_prediction_prints_topic(self, capsys, tmp_path) -> None:
        db_path, ticket_id, run_id = _make_eval_db_with_prediction(tmp_path)
        lookup_prediction(db_path, ticket_id, run_id)
        out = capsys.readouterr().out
        assert "Technical / Online Access" in out

    def test_found_prediction_prints_confidence(self, capsys, tmp_path) -> None:
        db_path, ticket_id, run_id = _make_eval_db_with_prediction(tmp_path)
        lookup_prediction(db_path, ticket_id, run_id)
        out = capsys.readouterr().out
        assert "0.82" in out

    def test_missing_prediction_prints_not_found(self, capsys, tmp_path) -> None:
        db_path, ticket_id, run_id = _make_eval_db_with_prediction(tmp_path)
        lookup_prediction(db_path, "unknown_ticket", run_id)
        out = capsys.readouterr().out
        assert "No prediction found" in out

    def test_empty_db_no_table_prints_message(self, capsys, tmp_path) -> None:
        db_path = _make_empty_db(tmp_path)
        lookup_prediction(db_path, "anyid", "run_xyz")
        out = capsys.readouterr().out
        assert "not found" in out.lower() or "triage_predictions" in out

    def test_missing_db_prints_not_found(self, capsys, tmp_path) -> None:
        lookup_prediction(str(tmp_path / "nofile.duckdb"), "anyid", "run_xyz")
        out = capsys.readouterr().out
        assert "not found" in out.lower()


# ── show_kpi_leaderboard ──────────────────────────────────────────────────────

class TestShowKpiLeaderboard:

    def test_no_db_prints_message(self, capsys, tmp_path) -> None:
        show_kpi_leaderboard(str(tmp_path / "nofile.duckdb"))
        out = capsys.readouterr().out
        assert "not found" in out.lower() or "DuckDB" in out

    def test_empty_db_no_tables_prints_message(self, capsys, tmp_path) -> None:
        db_path = _make_empty_db(tmp_path)
        show_kpi_leaderboard(db_path)
        out = capsys.readouterr().out
        assert "No evaluation tables" in out or "No runs" in out

    def test_seeded_run_shows_run_id(self, capsys, tmp_path) -> None:
        db_path, run_id = _make_runs_db(tmp_path)
        show_kpi_leaderboard(db_path)
        out = capsys.readouterr().out
        assert run_id[:18] in out

    def test_seeded_run_shows_urgency_accuracy(self, capsys, tmp_path) -> None:
        db_path, run_id = _make_runs_db(tmp_path)
        show_kpi_leaderboard(db_path)
        out = capsys.readouterr().out
        assert "0.7500" in out

    def test_seeded_run_shows_triage_model(self, capsys, tmp_path) -> None:
        """Leaderboard must display the triage_model column value."""
        db_path, run_id = _make_runs_db(tmp_path)
        show_kpi_leaderboard(db_path)
        out = capsys.readouterr().out
        assert "qwen2.5:7b" in out

    def test_seeded_run_shows_reviewer_column(self, capsys, tmp_path) -> None:
        """Leaderboard must display a reviewer yes/no column."""
        db_path, run_id = _make_runs_db(tmp_path)
        show_kpi_leaderboard(db_path)
        out = capsys.readouterr().out
        # reviewer_enabled=false in seeded DB → "no" should appear
        assert "no" in out.lower()

    def test_old_run_without_reviewer_columns_renders_safely(self, capsys, tmp_path) -> None:
        """
        A triage_runs table without triage_model/reviewer_enabled/reviewer_model
        (simulating a pre-migration DB) must not crash show_kpi_leaderboard.
        The existing except handler should print an error gracefully.
        """
        path   = str(tmp_path / "old.duckdb")
        run_id = "run_old_001"
        conn   = duckdb.connect(path)
        # Old schema — no reviewer columns
        conn.execute("""
            CREATE TABLE triage_runs (
                run_id          TEXT PRIMARY KEY,
                created_at      TIMESTAMP,
                model_name      TEXT,
                split_name      TEXT,
                limit_n         INTEGER,
                runtime_seconds DOUBLE
            )
        """)
        conn.execute(
            "INSERT INTO triage_runs VALUES (?, NOW(), 'old-model', 'eval', 5, 10.0)",
            [run_id],
        )
        conn.execute("""
            CREATE TABLE triage_metrics (
                run_id TEXT, metric_group TEXT,
                metric_name TEXT, metric_value DOUBLE, created_at TIMESTAMP
            )
        """)
        conn.close()
        # Must not raise — either prints rows or an error message
        show_kpi_leaderboard(path)
        out = capsys.readouterr().out
        # Any non-empty output is acceptable — the function did not crash
        assert len(out) > 0


# ── Full-history A/B run-ID display (truncation fix) ──────────────────────────

def _make_ab_runs_db(tmp_path) -> str:
    """
    Return db_path seeded with two A/B runs whose IDs have the _ab_off / _ab_on
    suffix and a third plain run.

    The A/B pair uses the real timestamp from the latest controlled experiment.
    created_at ordering: ab_on is most-recent, ab_off second, plain run oldest.
    """
    path = str(tmp_path / "ab_runs.duckdb")
    conn = duckdb.connect(path)
    conn.execute("""
        CREATE TABLE triage_runs (
            run_id           TEXT PRIMARY KEY,
            created_at       TIMESTAMP,
            model_name       TEXT,
            triage_model     TEXT,
            reviewer_enabled BOOLEAN,
            reviewer_model   TEXT,
            split_name       TEXT,
            limit_n          INTEGER,
            top_k            INTEGER,
            temperature      DOUBLE,
            max_retries      INTEGER,
            runtime_seconds  DOUBLE,
            config_json      TEXT
        )
    """)
    rows = [
        ("run_20260713_014052_ab_on",  "2026-07-13 01:41:00", "llama3.2:3b", True,  "granite4.1:8b"),
        ("run_20260713_014052_ab_off", "2026-07-13 01:40:52", "llama3.2:3b", False, ""),
        ("run_20260710_013943",        "2026-07-10 01:39:43", "llama3.2:3b", False, ""),
    ]
    for rid, ts, model, rev_enabled, rev_model in rows:
        conn.execute(
            "INSERT INTO triage_runs VALUES (?, ?, ?, ?, ?, ?, 'eval', 200, 5, 0.1, 1, 600.0, '{}')",
            [rid, ts, model, model, rev_enabled, rev_model],
        )
    conn.execute("""
        CREATE TABLE triage_metrics (
            run_id TEXT, metric_group TEXT,
            metric_name TEXT, metric_value DOUBLE, created_at TIMESTAMP
        )
    """)
    for rid in [r[0] for r in rows]:
        conn.execute(
            "INSERT INTO triage_metrics VALUES (?, 'group', 'urgency_accuracy', 0.55, NOW())",
            [rid],
        )
    conn.close()
    return path


class TestFullHistoryRunIdDisplay:
    """
    Focused tests proving A/B run IDs are displayed without truncation.

    Requirements:
      - _ab_off suffix is visible in full-history output
      - _ab_on suffix is visible in full-history output
      - Both IDs are distinguishable (neither is truncated to the same prefix)
      - The complete displayed run_id can be used with show_run_details
      - Ordering is newest created_at first (ab_on before ab_off)
    """

    def test_ab_off_suffix_displayed_completely(self, capsys, tmp_path) -> None:
        """run_20260713_014052_ab_off must appear in full in the full-history table."""
        db_path = _make_ab_runs_db(tmp_path)
        show_kpi_leaderboard(db_path)
        out = capsys.readouterr().out
        assert "run_20260713_014052_ab_off" in out, (
            "Full run ID 'run_20260713_014052_ab_off' must not be truncated"
        )

    def test_ab_on_suffix_displayed_completely(self, capsys, tmp_path) -> None:
        """run_20260713_014052_ab_on must appear in full in the full-history table."""
        db_path = _make_ab_runs_db(tmp_path)
        show_kpi_leaderboard(db_path)
        out = capsys.readouterr().out
        assert "run_20260713_014052_ab_on" in out, (
            "Full run ID 'run_20260713_014052_ab_on' must not be truncated"
        )

    def test_ab_off_and_ab_on_are_distinguishable(self, capsys, tmp_path) -> None:
        """
        Both IDs must appear as distinct strings.

        If both were truncated to the same prefix they would be indistinguishable.
        We verify that the full suffix strings 'ab_off' and 'ab_on' both appear.
        """
        db_path = _make_ab_runs_db(tmp_path)
        show_kpi_leaderboard(db_path)
        out = capsys.readouterr().out
        assert "ab_off" in out, "Suffix 'ab_off' must be visible"
        assert "ab_on" in out,  "Suffix 'ab_on' must be visible"

    def test_complete_ab_off_id_usable_with_run_details(self, capsys, tmp_path) -> None:
        """
        show_run_details must find the run when called with the complete displayed ID.

        This verifies that the displayed ID is not corrupted (no extra chars, no
        truncation) and can be used for an exact lookup.
        """
        db_path = _make_ab_runs_db(tmp_path)
        # The run was inserted with this exact ID — show_run_details must find it.
        show_run_details(db_path, "run_20260713_014052_ab_off")
        out = capsys.readouterr().out
        assert "not found" not in out.lower(), (
            "show_run_details must find 'run_20260713_014052_ab_off' by exact ID"
        )
        assert "run_20260713_014052_ab_off" in out

    def test_full_history_ordering_newest_first(self, capsys, tmp_path) -> None:
        """
        The full-history table must list runs in descending created_at order.

        In the seeded DB: ab_on > ab_off > plain run.
        So ab_on must appear before ab_off in the output.
        """
        db_path = _make_ab_runs_db(tmp_path)
        show_kpi_leaderboard(db_path)
        out = capsys.readouterr().out
        pos_on  = out.find("run_20260713_014052_ab_on")
        pos_off = out.find("run_20260713_014052_ab_off")
        assert pos_on != -1,  "ab_on must appear in output"
        assert pos_off != -1, "ab_off must appear in output"
        assert pos_on < pos_off, (
            "ab_on (newer created_at) must appear before ab_off in full-history table"
        )


# ── show_run_details ──────────────────────────────────────────────────────────

class TestShowRunDetails:

    def test_missing_run_id_prints_not_found(self, capsys, tmp_path) -> None:
        db_path, run_id = _make_runs_db(tmp_path)
        show_run_details(db_path, "run_does_not_exist")
        out = capsys.readouterr().out
        assert "not found" in out.lower()

    def test_found_run_prints_model_name(self, capsys, tmp_path) -> None:
        db_path, run_id = _make_runs_db(tmp_path)
        show_run_details(db_path, run_id)
        out = capsys.readouterr().out
        assert "qwen2.5:7b" in out

    def test_found_run_prints_kpi_metrics(self, capsys, tmp_path) -> None:
        db_path, run_id = _make_runs_db(tmp_path)
        show_run_details(db_path, run_id)
        out = capsys.readouterr().out
        assert "urgency_accuracy" in out

    def test_no_db_prints_message(self, capsys, tmp_path) -> None:
        show_run_details(str(tmp_path / "nofile.duckdb"), "run_xyz")
        out = capsys.readouterr().out
        assert "not found" in out.lower() or "DuckDB" in out


# ── show_confusion_matrix ─────────────────────────────────────────────────────

class TestShowConfusionMatrix:

    def test_missing_run_prints_not_found(self, capsys, tmp_path) -> None:
        db_path, run_id = _make_confusion_db(tmp_path)
        show_confusion_matrix(db_path, "run_does_not_exist")
        out = capsys.readouterr().out
        assert "No confusion matrix found" in out

    def test_found_matrix_prints_target_name(self, capsys, tmp_path) -> None:
        db_path, run_id = _make_confusion_db(tmp_path)
        show_confusion_matrix(db_path, run_id)
        out = capsys.readouterr().out
        assert "urgency" in out

    def test_found_matrix_prints_actual_labels(self, capsys, tmp_path) -> None:
        db_path, run_id = _make_confusion_db(tmp_path)
        show_confusion_matrix(db_path, run_id)
        out = capsys.readouterr().out
        assert "High" in out
        assert "Low" in out

    def test_found_matrix_prints_counts(self, capsys, tmp_path) -> None:
        db_path, run_id = _make_confusion_db(tmp_path)
        show_confusion_matrix(db_path, run_id)
        out = capsys.readouterr().out
        assert "5" in out

    def test_no_db_prints_message(self, capsys, tmp_path) -> None:
        show_confusion_matrix(str(tmp_path / "nofile.duckdb"), "run_xyz")
        out = capsys.readouterr().out
        assert "not found" in out.lower() or "DuckDB" in out


# ── run_leakage_audit ─────────────────────────────────────────────────────────

class TestRunLeakageAudit:

    def test_prints_leakage_rules_section(self, capsys, tmp_path) -> None:
        run_leakage_audit(str(tmp_path / "nofile.duckdb"), str(tmp_path / "nolance"))
        out = capsys.readouterr().out
        assert "Leakage Rules" in out or "Leakage Audit" in out

    def test_prints_prediction_input_fields(self, capsys, tmp_path) -> None:
        run_leakage_audit(str(tmp_path / "nofile.duckdb"), str(tmp_path / "nolance"))
        out = capsys.readouterr().out
        assert "subject" in out
        assert "body" in out

    def test_prints_evaluation_only_fields(self, capsys, tmp_path) -> None:
        run_leakage_audit(str(tmp_path / "nofile.duckdb"), str(tmp_path / "nolance"))
        out = capsys.readouterr().out
        assert "actual_queue" in out
        assert "proxy_topic" in out

    def test_prints_answer_not_read(self, capsys, tmp_path) -> None:
        run_leakage_audit(str(tmp_path / "nofile.duckdb"), str(tmp_path / "nolance"))
        out = capsys.readouterr().out
        assert "answer" in out

    def test_with_seeded_db_shows_split_counts(self, capsys, tmp_path) -> None:
        db_path, ticket_id = _make_tickets_db(tmp_path)
        run_leakage_audit(db_path, str(tmp_path / "nolance"))
        out = capsys.readouterr().out
        assert "eval" in out


# ── run_submission_checklist ──────────────────────────────────────────────────

class TestRunSubmissionChecklist:

    def _minimal_cfg(self) -> dict:
        return {
            "embedding":    {"model_name": "e5-large"},
            "vector_store": {"path": "data/lancedb", "table_name": "ticket_embeddings"},
            "retrieval":    {"top_k": 5},
            "llm":          {"model_name": "qwen2.5:7b"},
            "batch": {
                "output_csv":       "outputs/triage_results.csv",
                "trace_jsonl":      "outputs/triage_trace.jsonl",
                "run_summary_json": "outputs/run_summary.json",
            },
        }

    def test_missing_artifacts_show_fail(self, capsys, tmp_path) -> None:
        cfg = self._minimal_cfg()
        cfg["vector_store"]["path"] = str(tmp_path / "nolance")
        cfg["batch"]["output_csv"]  = str(tmp_path / "nocsv.csv")
        cfg["batch"]["trace_jsonl"] = str(tmp_path / "nojsonl.jsonl")
        cfg["batch"]["run_summary_json"] = str(tmp_path / "nojson.json")
        run_submission_checklist(cfg, str(tmp_path / "nodb.duckdb"))
        out = capsys.readouterr().out
        assert "[FAIL]" in out

    def test_existing_db_with_no_rows_shows_fail(self, capsys, tmp_path) -> None:
        cfg     = self._minimal_cfg()
        db_path = str(tmp_path / "empty.duckdb")
        conn    = duckdb.connect(db_path)
        conn.execute("""
            CREATE TABLE historical_tickets (
                ticket_id TEXT PRIMARY KEY,
                split_name TEXT, subject TEXT, body TEXT,
                raw_text TEXT, cleaned_text TEXT,
                representation_text TEXT, text_snippet TEXT
            )
        """)
        conn.close()
        cfg["vector_store"]["path"] = str(tmp_path / "nolance")
        cfg["batch"]["output_csv"]  = str(tmp_path / "nocsv.csv")
        cfg["batch"]["trace_jsonl"] = str(tmp_path / "nojsonl.jsonl")
        cfg["batch"]["run_summary_json"] = str(tmp_path / "nojson.json")
        run_submission_checklist(cfg, db_path)
        out = capsys.readouterr().out
        assert "[FAIL]" in out

    def test_config_section_check_passes(self, capsys, tmp_path) -> None:
        cfg = self._minimal_cfg()
        cfg["vector_store"]["path"] = str(tmp_path / "nolance")
        cfg["batch"]["output_csv"]  = str(tmp_path / "nocsv.csv")
        cfg["batch"]["trace_jsonl"] = str(tmp_path / "nojsonl.jsonl")
        cfg["batch"]["run_summary_json"] = str(tmp_path / "nojson.json")
        run_submission_checklist(cfg, str(tmp_path / "nodb.duckdb"))
        out = capsys.readouterr().out
        # config sections should all pass
        # status is "[OK]  " (2 trailing spaces) + " " separator = 3 spaces before label
        assert "config.yaml has [embedding]" in out
        assert "config.yaml has [llm]" in out
        # confirm they show as OK, not FAIL
        assert "[FAIL] config.yaml has [embedding]" not in out
        assert "[FAIL] config.yaml has [llm]" not in out

    def test_prints_summary_line(self, capsys, tmp_path) -> None:
        cfg = self._minimal_cfg()
        cfg["vector_store"]["path"] = str(tmp_path / "nolance")
        cfg["batch"]["output_csv"]  = str(tmp_path / "nocsv.csv")
        cfg["batch"]["trace_jsonl"] = str(tmp_path / "nojsonl.jsonl")
        cfg["batch"]["run_summary_json"] = str(tmp_path / "nojson.json")
        run_submission_checklist(cfg, str(tmp_path / "nodb.duckdb"))
        out = capsys.readouterr().out
        assert "check" in out.lower()


# ── prompt_reviewer_override ──────────────────────────────────────────────────

def _base_cfg() -> dict:
    """Minimal config with a reviewer section for override tests."""
    return {
        "llm": {"model_name": "llama3.2:3b"},
        "reviewer": {
            "enabled": True,
            "model_name": "qwen2.5:7b",
            "trigger_flags": ["low_llm_confidence"],
        },
        "batch": {"sample_strategy": "natural", "limit": 10},
    }


class TestPromptReviewerOverride:

    def test_reviewer_disabled_override(self, monkeypatch) -> None:
        """Answering 'n' sets reviewer.enabled = False in the returned copy."""
        monkeypatch.setattr("builtins.input", lambda _: "n")
        cfg = _base_cfg()
        result = prompt_reviewer_override(cfg)
        assert result["reviewer"]["enabled"] is False

    def test_reviewer_enabled_override_keeps_model(self, monkeypatch) -> None:
        """Answering 'y' then Enter keeps the existing reviewer model."""
        responses = iter(["y", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        cfg = _base_cfg()
        result = prompt_reviewer_override(cfg)
        assert result["reviewer"]["enabled"] is True
        assert result["reviewer"]["model_name"] == "qwen2.5:7b"

    def test_reviewer_model_override(self, monkeypatch) -> None:
        """Answering 'y' then a model name replaces reviewer.model_name."""
        responses = iter(["y", "new-model:7b"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        cfg = _base_cfg()
        result = prompt_reviewer_override(cfg)
        assert result["reviewer"]["enabled"] is True
        assert result["reviewer"]["model_name"] == "new-model:7b"

    def test_config_not_mutated(self, monkeypatch) -> None:
        """The original cfg dict must not be modified by the override."""
        monkeypatch.setattr("builtins.input", lambda _: "n")
        cfg = _base_cfg()
        original_enabled = cfg["reviewer"]["enabled"]
        original_model   = cfg["reviewer"]["model_name"]
        prompt_reviewer_override(cfg)
        assert cfg["reviewer"]["enabled"]    == original_enabled
        assert cfg["reviewer"]["model_name"] == original_model

    def test_empty_answer_disables_reviewer(self, monkeypatch) -> None:
        """Pressing Enter (empty input) should default to disabled."""
        monkeypatch.setattr("builtins.input", lambda _: "")
        cfg = _base_cfg()
        result = prompt_reviewer_override(cfg)
        assert result["reviewer"]["enabled"] is False


# ── print_effective_run_config ────────────────────────────────────────────────

class TestPrintEffectiveRunConfig:

    def test_reviewer_disabled_shows_no(self, capsys) -> None:
        cfg = _base_cfg()
        cfg["reviewer"]["enabled"] = False
        print_effective_run_config(cfg, limit=20)
        out = capsys.readouterr().out
        assert "No" in out

    def test_reviewer_enabled_shows_model_name(self, capsys) -> None:
        cfg = _base_cfg()
        cfg["reviewer"]["enabled"] = True
        print_effective_run_config(cfg, limit=20)
        out = capsys.readouterr().out
        assert "qwen2.5:7b" in out
        assert "Yes" in out

    def test_shows_analyzer_model(self, capsys) -> None:
        cfg = _base_cfg()
        print_effective_run_config(cfg, limit=5)
        out = capsys.readouterr().out
        assert "llama3.2:3b" in out

    def test_shows_batch_limit(self, capsys) -> None:
        cfg = _base_cfg()
        print_effective_run_config(cfg, limit=42)
        out = capsys.readouterr().out
        assert "42" in out

    def test_reviewer_model_not_shown_when_disabled(self, capsys) -> None:
        cfg = _base_cfg()
        cfg["reviewer"]["enabled"] = False
        print_effective_run_config(cfg, limit=10)
        out = capsys.readouterr().out
        assert "qwen2.5:7b" not in out


# ── apply_reviewer_override ───────────────────────────────────────────────────

class TestApplyReviewerOverride:

    def test_override_true_forces_reviewer_enabled(self) -> None:
        cfg = {"reviewer": {"enabled": False, "model_name": "test-model"}}
        result = apply_reviewer_override(cfg, True)
        assert result["reviewer"]["enabled"] is True

    def test_override_false_forces_reviewer_disabled(self) -> None:
        cfg = {"reviewer": {"enabled": True, "model_name": "test-model"}}
        result = apply_reviewer_override(cfg, False)
        assert result["reviewer"]["enabled"] is False

    def test_override_none_leaves_config_value_unchanged(self) -> None:
        cfg = {"reviewer": {"enabled": True, "model_name": "test-model"}}
        result = apply_reviewer_override(cfg, None)
        assert result["reviewer"]["enabled"] is True

    def test_does_not_mutate_original_config(self) -> None:
        cfg = {"reviewer": {"enabled": True, "model_name": "test-model"}}
        apply_reviewer_override(cfg, False)
        assert cfg["reviewer"]["enabled"] is True

    def test_reviewer_section_created_when_absent_and_override_true(self) -> None:
        cfg: dict = {}
        result = apply_reviewer_override(cfg, True)
        assert result["reviewer"]["enabled"] is True
        # Original must be untouched.
        assert "reviewer" not in cfg

    def test_reviewer_section_created_when_absent_and_override_false(self) -> None:
        cfg: dict = {}
        result = apply_reviewer_override(cfg, False)
        assert result["reviewer"]["enabled"] is False
        assert "reviewer" not in cfg


# ── toggle_reviewer_session_override ─────────────────────────────────────────

class TestToggleReviewerSessionOverride:

    def test_choice_1_returns_true(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "1")
        cfg = {"reviewer": {"enabled": False}}
        result = toggle_reviewer_session_override(cfg, None)
        assert result is True

    def test_choice_2_returns_false(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "2")
        cfg = {"reviewer": {"enabled": True}}
        result = toggle_reviewer_session_override(cfg, None)
        assert result is False

    def test_choice_3_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "3")
        cfg = {"reviewer": {"enabled": True}}
        result = toggle_reviewer_session_override(cfg, True)
        assert result is None

    def test_invalid_choice_returns_current_override(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "X")
        cfg = {"reviewer": {"enabled": True}}
        result = toggle_reviewer_session_override(cfg, False)
        assert result is False

    def test_empty_input_returns_current_override(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "")
        cfg = {"reviewer": {"enabled": True}}
        result = toggle_reviewer_session_override(cfg, True)
        assert result is True

    def test_prints_effective_state(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "1")
        cfg = {"reviewer": {"enabled": False}}
        toggle_reviewer_session_override(cfg, None)
        out = capsys.readouterr().out
        assert "Effective reviewer state" in out


# ── run_reviewer_ab_comparison ────────────────────────────────────────────────

def _make_fake_records(n: int = 3) -> list[dict]:
    """Return n minimal ticket dicts suitable for A/B comparison tests."""
    return [
        {
            "ticket_id": f"ticket_{i:04d}",
            "subject": "Test subject",
            "body": "Test body",
            "raw_text": "Test subject Test body",
            "cleaned_text": "test subject test body",
            "representation_text": "Subject: Test subject\n\nBody: Test body",
            "text_snippet": "Test subject Test body",
            "actual_queue": "Technical Support",
            "actual_priority": "medium",
            "actual_type": "Incident",
            "actual_tags_json": "[]",
            "proxy_topic": "Other",
            "proxy_urgency": "Medium",
            "proxy_next_action": "ask_for_more_information",
            "proxy_topic_source": "fallback",
        }
        for i in range(n)
    ]


def _ab_cfg() -> dict:
    """Minimal config for A/B comparison tests."""
    return {
        "llm": {"model_name": "llama3.2:3b", "base_url": "http://localhost:11434"},
        "reviewer": {
            "enabled": True,
            "model_name": "qwen2.5:7b",
            "trigger_flags": ["low_llm_confidence"],
        },
        "batch": {
            "limit": 3,
            "sample_strategy": "natural",
            "split": "eval",
            "random_seed": 42,
            "limit_per_label": 50,
        },
    }


class TestRunReviewerAbComparison:
    """
    Tests for run_reviewer_ab_comparison.

    _fetch_all_split_records and _execute_batch_on_records are replaced by
    lightweight stubs via monkeypatch so no DuckDB, Ollama, or embedding model
    is required.
    """

    def _stub_execute(self, calls: list, cfg, db_path, records, embedding_model,
                      run_id=None, write_files=True):
        """Record call args and return stub metrics."""
        calls.append({"cfg": copy.deepcopy(cfg), "records": records, "run_id": run_id})
        metrics = {kpi: 0.5 for kpi in AB_COMPARISON_KPIS}
        return run_id, metrics, []

    def test_same_records_passed_to_both_runs(self, monkeypatch, tmp_path) -> None:
        fake_records = _make_fake_records(3)
        monkeypatch.setattr(
            _cli_menu_mod, "_fetch_all_split_records",
            lambda db_path, split: fake_records,
        )
        calls: list = []
        monkeypatch.setattr(
            _cli_menu_mod, "_execute_batch_on_records",
            lambda cfg, db_path, records, em, run_id=None, write_files=True:
                self._stub_execute(calls, cfg, db_path, records, em, run_id, write_files),
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        cfg = _ab_cfg()
        run_reviewer_ab_comparison(cfg, str(tmp_path / "fake.duckdb"), object())

        assert len(calls) == 2
        # Both calls received the same sampled list object.
        assert calls[0]["records"] is calls[1]["records"]
        # Each record in that list has the expected ticket_id.
        sampled_ids = [r["ticket_id"] for r in calls[0]["records"]]
        assert sampled_ids == [r["ticket_id"] for r in calls[1]["records"]]

    def test_two_distinct_ab_suffixed_run_ids(self, monkeypatch, tmp_path) -> None:
        fake_records = _make_fake_records(2)
        monkeypatch.setattr(
            _cli_menu_mod, "_fetch_all_split_records",
            lambda db_path, split: fake_records,
        )
        run_ids: list = []
        monkeypatch.setattr(
            _cli_menu_mod, "_execute_batch_on_records",
            lambda cfg, db_path, records, em, run_id=None, write_files=True:
                run_ids.append(run_id) or (run_id, {kpi: 0.0 for kpi in AB_COMPARISON_KPIS}, []),
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        run_reviewer_ab_comparison(_ab_cfg(), str(tmp_path / "fake.duckdb"), object())

        assert len(run_ids) == 2
        assert run_ids[0] != run_ids[1]
        assert run_ids[0].endswith("_ab_off")
        assert run_ids[1].endswith("_ab_on")

    def test_prints_all_kpi_names(self, monkeypatch, tmp_path, capsys) -> None:
        fake_records = _make_fake_records(1)
        monkeypatch.setattr(
            _cli_menu_mod, "_fetch_all_split_records",
            lambda db_path, split: fake_records,
        )
        monkeypatch.setattr(
            _cli_menu_mod, "_execute_batch_on_records",
            lambda cfg, db_path, records, em, run_id=None, write_files=True:
                (run_id, {kpi: 0.0 for kpi in AB_COMPARISON_KPIS}, []),
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        run_reviewer_ab_comparison(_ab_cfg(), str(tmp_path / "fake.duckdb"), object())

        out = capsys.readouterr().out
        for kpi in AB_COMPARISON_KPIS:
            assert kpi in out, f"Expected KPI '{kpi}' not found in output"

    def test_run_a_has_reviewer_disabled(self, monkeypatch, tmp_path) -> None:
        fake_records = _make_fake_records(2)
        monkeypatch.setattr(
            _cli_menu_mod, "_fetch_all_split_records",
            lambda db_path, split: fake_records,
        )
        calls: list = []
        monkeypatch.setattr(
            _cli_menu_mod, "_execute_batch_on_records",
            lambda cfg, db_path, records, em, run_id=None, write_files=True:
                self._stub_execute(calls, cfg, db_path, records, em, run_id, write_files),
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        run_reviewer_ab_comparison(_ab_cfg(), str(tmp_path / "fake.duckdb"), object())

        # First call = Run A (reviewer off).
        assert calls[0]["cfg"]["reviewer"]["enabled"] is False
        # Second call = Run B (reviewer on).
        assert calls[1]["cfg"]["reviewer"]["enabled"] is True

    def test_no_records_prints_message(self, monkeypatch, tmp_path, capsys) -> None:
        monkeypatch.setattr(
            _cli_menu_mod, "_fetch_all_split_records",
            lambda db_path, split: [],
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        run_reviewer_ab_comparison(_ab_cfg(), str(tmp_path / "fake.duckdb"), object())

        out = capsys.readouterr().out
        assert "No records found" in out

    def test_effective_config_not_mutated(self, monkeypatch, tmp_path) -> None:
        """Original cfg passed to run_reviewer_ab_comparison must not be modified."""
        fake_records = _make_fake_records(2)
        monkeypatch.setattr(
            _cli_menu_mod, "_fetch_all_split_records",
            lambda db_path, split: fake_records,
        )
        monkeypatch.setattr(
            _cli_menu_mod, "_execute_batch_on_records",
            lambda cfg, db_path, records, em, run_id=None, write_files=True:
                (run_id, {kpi: 0.0 for kpi in AB_COMPARISON_KPIS}, []),
        )
        monkeypatch.setattr("builtins.input", lambda _: "")

        original_cfg = _ab_cfg()
        original_enabled = original_cfg["reviewer"]["enabled"]
        run_reviewer_ab_comparison(original_cfg, str(tmp_path / "fake.duckdb"), object())
        assert original_cfg["reviewer"]["enabled"] == original_enabled


# ── Balanced total-limit sampling ─────────────────────────────────────────────

def _make_labeled_records(classes: list[str], per_class: int) -> list[dict]:
    """
    Return fake ticket dicts with known proxy_topic / proxy_urgency labels.

    Each record has a stable ticket_id so determinism tests can compare ID lists.
    """
    records = []
    for cls in classes:
        for i in range(per_class):
            records.append({
                "ticket_id":            f"{cls}_{i:04d}",
                "subject":              "Test subject",
                "body":                 "Test body",
                "raw_text":             "Test subject Test body",
                "cleaned_text":         "test subject test body",
                "representation_text":  "Subject: Test subject\n\nBody: Test body",
                "text_snippet":         "Test subject Test body",
                "actual_queue":         "Technical Support",
                "actual_priority":      "medium",
                "actual_type":          "Incident",
                "actual_tags_json":     "[]",
                "proxy_topic":          cls,
                "proxy_urgency":        "Medium",
                "proxy_next_action":    "ask_for_more_information",
                "proxy_topic_source":   "fallback",
            })
    return records


class TestBalancedSamplingTotal:
    """
    Tests for _sample_tickets with balanced strategies.

    Verifies that `limit` is always the TOTAL cap (not per-class),
    that sampling is deterministic, and that config.batch.limit_per_label
    is ignored by the A/B comparison.
    """

    def test_balanced_proxy_topic_at_most_limit(self) -> None:
        """balanced_proxy_topic with limit=20 must return at most 20 rows."""
        records = _make_labeled_records(
            ["Claims / Damage", "Billing / Payment", "Technical / Online Access", "Other"],
            per_class=30,
        )
        result = _sample_tickets(records, "balanced_proxy_topic", limit=20, random_seed=42)
        assert len(result) <= 20

    def test_balanced_proxy_urgency_at_most_limit(self) -> None:
        """balanced_proxy_urgency with limit=20 must return at most 20 rows."""
        records = []
        for urgency, i in [("High", 0), ("Medium", 1), ("Low", 2)]:
            for j in range(30):
                rec = dict(_make_fake_records(1)[0])
                rec["ticket_id"]     = f"urg_{urgency}_{j:04d}"
                rec["proxy_urgency"] = urgency
                records.append(rec)
        result = _sample_tickets(records, "balanced_proxy_urgency", limit=20, random_seed=42)
        assert len(result) <= 20

    def test_balanced_proxy_topic_exact_count(self) -> None:
        """With enough rows in every class the result is exactly limit."""
        records = _make_labeled_records(["A", "B", "C", "D"], per_class=20)
        result = _sample_tickets(records, "balanced_proxy_topic", limit=12, random_seed=42)
        assert len(result) == 12

    def test_balanced_sampling_deterministic_same_seed(self) -> None:
        """Same seed must produce the same ticket_id order."""
        records = _make_labeled_records(["A", "B", "C"], per_class=20)
        ids1 = [r["ticket_id"] for r in _sample_tickets(
            records, "balanced_proxy_topic", limit=15, random_seed=42
        )]
        ids2 = [r["ticket_id"] for r in _sample_tickets(
            records, "balanced_proxy_topic", limit=15, random_seed=42
        )]
        assert ids1 == ids2

    def test_balanced_sampling_different_seeds_differ(self) -> None:
        """Different seeds must produce different orderings."""
        records = _make_labeled_records(["A", "B", "C"], per_class=20)
        ids1 = [r["ticket_id"] for r in _sample_tickets(
            records, "balanced_proxy_topic", limit=15, random_seed=42
        )]
        ids2 = [r["ticket_id"] for r in _sample_tickets(
            records, "balanced_proxy_topic", limit=15, random_seed=99
        )]
        assert ids1 != ids2

    def test_balanced_fewer_rows_than_limit_takes_all(self) -> None:
        """If total rows < limit, all available rows are returned."""
        records = _make_labeled_records(["A", "B"], per_class=3)  # 6 total
        result = _sample_tickets(records, "balanced_proxy_topic", limit=20, random_seed=42)
        assert len(result) == 6

    def test_limit_per_label_in_config_does_not_override_ab_limit(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        config.batch.limit_per_label must NOT control how many tickets are
        processed in the A/B comparison.  With limit=5 and 5 classes × 10 rows
        (50 total), both runs must receive at most 5 records.
        """
        # 5 classes × 10 rows = 50 records
        records = _make_labeled_records(["A", "B", "C", "D", "E"], per_class=10)
        monkeypatch.setattr(
            _cli_menu_mod, "_fetch_all_split_records",
            lambda db_path, split: records,
        )
        captured: list[list] = []

        def stub_execute(cfg, db_path, recs, em, run_id=None, write_files=True):
            captured.append(list(recs))
            return run_id, {kpi: 0.5 for kpi in AB_COMPARISON_KPIS}, []

        monkeypatch.setattr(_cli_menu_mod, "_execute_batch_on_records", stub_execute)

        # limit=5 as default; strategy=balanced_proxy_topic; limit_per_label=1000 in config
        cfg = _ab_cfg()
        cfg["batch"]["limit"]           = 5
        cfg["batch"]["limit_per_label"] = 1000
        cfg["batch"]["sample_strategy"] = "balanced_proxy_topic"
        monkeypatch.setattr("builtins.input", lambda _: "")

        run_reviewer_ab_comparison(cfg, str(tmp_path / "fake.duckdb"), object())

        assert len(captured) == 2, "Expected exactly two batch runs (A and B)"
        assert len(captured[0]) <= 5, (
            f"Run A received {len(captured[0])} records — must be ≤ 5"
        )
        assert len(captured[1]) <= 5, (
            f"Run B received {len(captured[1])} records — must be ≤ 5"
        )

    def test_ab_comparison_same_ticket_ids_for_balanced_strategy(
        self, monkeypatch, tmp_path
    ) -> None:
        """Both A and B runs must receive the exact same ticket_ids when using balanced sampling."""
        records = _make_labeled_records(["X", "Y", "Z"], per_class=10)
        monkeypatch.setattr(
            _cli_menu_mod, "_fetch_all_split_records",
            lambda db_path, split: records,
        )
        captured: list[list] = []

        def stub_execute(cfg, db_path, recs, em, run_id=None, write_files=True):
            captured.append(list(recs))
            return run_id, {kpi: 0.5 for kpi in AB_COMPARISON_KPIS}, []

        monkeypatch.setattr(_cli_menu_mod, "_execute_batch_on_records", stub_execute)

        cfg = _ab_cfg()
        cfg["batch"]["limit"]           = 9
        cfg["batch"]["sample_strategy"] = "balanced_proxy_topic"
        monkeypatch.setattr("builtins.input", lambda _: "")

        run_reviewer_ab_comparison(cfg, str(tmp_path / "fake.duckdb"), object())

        assert len(captured) == 2
        ids_a = [r["ticket_id"] for r in captured[0]]
        ids_b = [r["ticket_id"] for r in captured[1]]
        assert ids_a == ids_b


# ══════════════════════════════════════════════════════════════════════════════
# Sprint 6C.7 — CLI navigation tests (cli.py)
# ══════════════════════════════════════════════════════════════════════════════

import cli  # noqa: E402 — import after fixtures to avoid circular at collection time
from cli import (  # noqa: E402
    MENU as CLI_MENU,
    _demo_loop,
    _select_reviewer_mode,
    _triage_ticket_menu,
    _evaluation_results_menu,
    _ticket_prediction_details,
)


def _nav_cfg() -> dict:
    """Minimal config for navigation tests."""
    return {
        "llm": {"model_name": "llama3.2:3b", "base_url": "http://localhost:11434"},
        "reviewer": {
            "enabled": False,
            "model_name": "reviewer-model:7b",
            "trigger_flags": [],
        },
        "embedding": {"model_name": "e5-large", "device": "auto",
                      "normalize_embeddings": True},
        "vector_store": {"path": "data/lancedb", "table_name": "ticket_embeddings"},
        "retrieval": {"top_k": 5},
        "batch": {
            "limit": 5,
            "sample_strategy": "natural",
            "split": "eval",
            "random_seed": 42,
        },
        "thresholds": {"low_confidence": 0.60},
    }


def _make_inputs(*values):
    """Return a function that cycles through values then always returns '0'."""
    it = iter(values)
    return lambda _: next(it, "0")


def _stub_noop(*args, **kwargs):
    return None


def _stub_embedding(_cfg):
    return object()


# ── Test 1 & 2: Menu content ──────────────────────────────────────────────────

class TestTopLevelMenuContent:
    """Test 1 & 2: Menu string has exactly options 0–6; removed items are gone."""

    def test_menu_has_option_zero(self):
        assert "0." in CLI_MENU

    def test_menu_has_option_six(self):
        assert "6." in CLI_MENU

    def test_menu_does_not_have_option_seven(self):
        # No 7th numbered option in the top-level menu.
        assert " 7." not in CLI_MENU
        assert "7." not in CLI_MENU

    def test_menu_does_not_have_option_fifteen(self):
        assert "15." not in CLI_MENU

    def test_removed_toggle_reviewer_not_in_menu(self):
        assert "Toggle reviewer" not in CLI_MENU
        assert "toggle reviewer" not in CLI_MENU.lower()

    def test_removed_analyzer_prompt_not_in_menu(self):
        assert "analyzer prompt" not in CLI_MENU.lower()

    def test_removed_reviewer_prompt_not_in_menu(self):
        assert "reviewer prompt" not in CLI_MENU.lower()

    def test_removed_submission_checklist_not_in_menu(self):
        assert "checklist" not in CLI_MENU.lower()

    def test_removed_ticket_lookup_not_top_level(self):
        # "Look up original ticket" was option 7 — should be gone from top menu.
        assert "Look up original" not in CLI_MENU

    def test_menu_contains_triage_a_ticket(self):
        assert "Triage a ticket" in CLI_MENU or "triage a ticket" in CLI_MENU.lower()

    def test_menu_contains_batch_evaluation(self):
        assert "batch evaluation" in CLI_MENU.lower()

    def test_menu_contains_reviewer_comparison(self):
        assert "reviewer" in CLI_MENU.lower()

    def test_menu_contains_inspect_evaluation(self):
        assert "evaluation results" in CLI_MENU.lower()

    def test_menu_contains_leakage_audit(self):
        assert "leakage" in CLI_MENU.lower()

    def test_menu_contains_runtime_configuration(self):
        assert "configuration" in CLI_MENU.lower()


# ── Test 3, 8: Submenu navigation ─────────────────────────────────────────────

class TestTriageSubmenuNavigation:
    """Tests 3 & 8: Option 1 opens the ticket submenu; back returns to main."""

    def test_option_1_opens_triage_submenu(self, monkeypatch, capsys, tmp_path):
        """Selecting 1 then 0 (back) then 0 (exit) must not crash."""
        monkeypatch.setattr("builtins.input", _make_inputs("1", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")
        out = capsys.readouterr().out
        # Triage submenu header must have appeared.
        assert "Triage a Ticket" in out or "Triage" in out

    def test_back_from_triage_returns_to_main(self, monkeypatch, tmp_path):
        """Selecting 1 → 0 (back) → 0 (exit main) must complete without error."""
        monkeypatch.setattr("builtins.input", _make_inputs("1", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")
        # No exception = pass.


# ── Tests 4, 5: Triage dispatch ───────────────────────────────────────────────

class TestTriageDispatch:
    """Tests 4 & 5: Random and specific ticket dispatch to existing functions."""

    def test_random_ticket_delegates_to_existing_function(
        self, monkeypatch, tmp_path
    ) -> None:
        called = []
        # Must patch on the cli module — it imports the name directly.
        monkeypatch.setattr(cli, "run_random_eval_ticket",
                            lambda *a, **kw: called.append("random"))
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)

        # 1 (triage) → 1 (random) → 1 (reviewer off) → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("1", "1", "1", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

        assert called == ["random"], "run_random_eval_ticket must be called once"

    def test_specific_ticket_delegates_to_existing_function(
        self, monkeypatch, tmp_path
    ) -> None:
        called = []
        monkeypatch.setattr(cli, "run_specific_ticket",
                            lambda *a, **kw: called.append("specific"))
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)

        # 1 (triage) → 2 (specific) → ticket_id → 1 (reviewer off) → 0 (back) → 0 (exit)
        monkeypatch.setattr(
            "builtins.input",
            _make_inputs("1", "2", "abc123def456", "1", "0", "0"),
        )
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

        assert called == ["specific"], "run_specific_ticket must be called once"


# ── Tests 6, 7: Per-operation reviewer mode ───────────────────────────────────

class TestReviewerModePerOperation:
    """Tests 6 & 7: Reviewer mode is applied only to the current operation."""

    def test_reviewer_off_passed_to_triage(self, monkeypatch, tmp_path) -> None:
        received_cfgs: list[dict] = []

        def capture_cfg(effective_cfg, db_path, em):
            received_cfgs.append(copy.deepcopy(effective_cfg))

        monkeypatch.setattr(cli, "run_random_eval_ticket", capture_cfg)
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)

        # 1 (triage) → 1 (random) → 1 (reviewer off) → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("1", "1", "1", "0", "0"))
        original_cfg = _nav_cfg()
        _demo_loop(original_cfg, str(tmp_path / "t.duckdb"), "data/lancedb")

        assert len(received_cfgs) == 1
        assert received_cfgs[0]["reviewer"]["enabled"] is False

    def test_reviewer_on_passed_to_triage(self, monkeypatch, tmp_path) -> None:
        received_cfgs: list[dict] = []

        def capture_cfg(effective_cfg, db_path, em):
            received_cfgs.append(copy.deepcopy(effective_cfg))

        monkeypatch.setattr(cli, "run_random_eval_ticket", capture_cfg)
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)

        # 1 (triage) → 1 (random) → 2 (conditional reviewer) → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("1", "1", "2", "0", "0"))
        cfg = _nav_cfg()
        # reviewer.model_name is set in _nav_cfg(), so conditional reviewer is valid.
        _demo_loop(cfg, str(tmp_path / "t.duckdb"), "data/lancedb")

        assert len(received_cfgs) == 1
        assert received_cfgs[0]["reviewer"]["enabled"] is True

    def test_reviewer_selection_does_not_mutate_original_cfg(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(cli, "run_random_eval_ticket", _stub_noop)
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)

        # 1 (triage) → 1 (random) → 2 (conditional reviewer) → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("1", "1", "2", "0", "0"))
        cfg = _nav_cfg()
        original_enabled = cfg["reviewer"]["enabled"]
        _demo_loop(cfg, str(tmp_path / "t.duckdb"), "data/lancedb")

        # Original cfg must be unchanged.
        assert cfg["reviewer"]["enabled"] == original_enabled

    def test_no_reviewer_model_prints_explanation(
        self, monkeypatch, capsys, tmp_path
    ) -> None:
        """Selecting conditional reviewer with no model_name prints an explanation."""
        monkeypatch.setattr(cli, "run_random_eval_ticket", _stub_noop)
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)

        cfg = _nav_cfg()
        cfg["reviewer"]["model_name"] = ""  # no model configured

        # 1 (triage) → 1 (random) → 2 (conditional reviewer — no model) → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("1", "1", "2", "0", "0"))
        _demo_loop(cfg, str(tmp_path / "t.duckdb"), "data/lancedb")

        out = capsys.readouterr().out
        assert "No reviewer model" in out or "not configured" in out.lower()


# ── Test 9: Batch evaluation ──────────────────────────────────────────────────

class TestBatchEvaluationDispatch:
    """Test 9: Option 2 delegates to the existing batch workflow."""

    def test_batch_delegates_to_run_batch_evaluation(
        self, monkeypatch, tmp_path
    ) -> None:
        called = []
        monkeypatch.setattr(cli, "run_batch_evaluation",
                            lambda *a, **kw: called.append("batch"))
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)
        monkeypatch.setattr(cli, "print_effective_run_config", _stub_noop)

        # 2 (batch) → "" (default limit) → 1 (reviewer off) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("2", "", "1", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

        assert called == ["batch"]

    def test_batch_reviewer_off_passes_disabled_cfg(
        self, monkeypatch, tmp_path
    ) -> None:
        received: list[dict] = []
        monkeypatch.setattr(
            cli, "run_batch_evaluation",
            lambda cfg, *a, **kw: received.append(copy.deepcopy(cfg)),
        )
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)
        monkeypatch.setattr(cli, "print_effective_run_config", _stub_noop)

        monkeypatch.setattr("builtins.input", _make_inputs("2", "", "1", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

        assert received[0]["reviewer"]["enabled"] is False

    def test_batch_does_not_mutate_original_cfg(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(cli, "run_batch_evaluation", _stub_noop)
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)
        monkeypatch.setattr(cli, "print_effective_run_config", _stub_noop)

        monkeypatch.setattr("builtins.input", _make_inputs("2", "", "1", "0"))
        cfg = _nav_cfg()
        original_enabled = cfg["reviewer"]["enabled"]
        _demo_loop(cfg, str(tmp_path / "t.duckdb"), "data/lancedb")
        assert cfg["reviewer"]["enabled"] == original_enabled


# ── Test 10: A/B comparison ───────────────────────────────────────────────────

class TestAbComparisonDispatch:
    """Test 10: Option 3 delegates to the reviewer A/B workflow."""

    def test_option_3_delegates_to_ab_comparison(
        self, monkeypatch, tmp_path
    ) -> None:
        called = []
        monkeypatch.setattr(cli, "run_reviewer_ab_comparison",
                            lambda *a, **kw: called.append("ab"))
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)

        # 3 (A/B) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("3", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

        assert called == ["ab"]

    def test_option_3_does_not_ask_reviewer_mode(
        self, monkeypatch, capsys, tmp_path
    ) -> None:
        """A/B comparison must NOT show the reviewer mode selection prompt."""
        monkeypatch.setattr(cli, "run_reviewer_ab_comparison", _stub_noop)
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)

        monkeypatch.setattr("builtins.input", _make_inputs("3", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

        out = capsys.readouterr().out
        # The per-operation reviewer mode prompt must not appear for A/B.
        assert "Reviewer Mode for This Operation" not in out


# ── Tests 11, 12: Evaluation results submenu ──────────────────────────────────

class TestEvaluationResultsSubmenu:
    """Tests 11 & 12: Option 4 opens submenu; submenu delegates to correct functions."""

    def test_option_4_opens_eval_submenu(self, monkeypatch, capsys, tmp_path):
        """Selecting 4 then 0 (back) then 0 (exit) must show the eval submenu."""
        monkeypatch.setattr("builtins.input", _make_inputs("4", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")
        out = capsys.readouterr().out
        assert "Inspect Evaluation Results" in out or "evaluation" in out.lower()

    def test_back_from_eval_submenu_returns_to_main(self, monkeypatch, tmp_path):
        """4 → 0 (back) → 0 (exit main) must complete without error."""
        monkeypatch.setattr("builtins.input", _make_inputs("4", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

    def test_eval_curated_delegates_to_function(
        self, monkeypatch, tmp_path
    ) -> None:
        """Test 2: Submenu option 1 routes to show_curated_leaderboard."""
        called = []
        monkeypatch.setattr(cli, "show_curated_leaderboard",
                            lambda *a, **kw: called.append("curated"))
        # 4 (eval submenu) → 1 (curated) → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("4", "1", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")
        assert called == ["curated"]

    def test_eval_full_history_delegates_to_function(
        self, monkeypatch, tmp_path
    ) -> None:
        """Full experiment history (option 2) routes to show_kpi_leaderboard."""
        called = []
        monkeypatch.setattr(cli, "show_kpi_leaderboard",
                            lambda *a, **kw: called.append("leaderboard"))
        # 4 (eval submenu) → 2 (full history) → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("4", "2", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")
        assert called == ["leaderboard"]

    def test_eval_run_details_delegates_to_function(
        self, monkeypatch, tmp_path
    ) -> None:
        called = []
        monkeypatch.setattr(cli, "show_run_details",
                            lambda *a, **kw: called.append("run_details"))
        # 4 (eval submenu) → 3 (run details) → run_id → 0 (back) → 0 (exit)
        monkeypatch.setattr(
            "builtins.input",
            _make_inputs("4", "3", "run_20260101_120000", "0", "0"),
        )
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")
        assert called == ["run_details"]

    def test_eval_confusion_matrix_delegates_to_function(
        self, monkeypatch, tmp_path
    ) -> None:
        called = []
        monkeypatch.setattr(cli, "show_confusion_matrix",
                            lambda *a, **kw: called.append("confusion"))
        # 4 (eval submenu) → 4 (confusion) → run_id → 0 (back) → 0 (exit)
        monkeypatch.setattr(
            "builtins.input",
            _make_inputs("4", "4", "run_20260101_120000", "0", "0"),
        )
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")
        assert called == ["confusion"]


# ── Test 13: Ticket prediction details ────────────────────────────────────────

class TestTicketPredictionDetails:
    """Test 13: Ticket prediction details reuses lookup_ticket + lookup_prediction."""

    def test_prediction_details_calls_lookup_ticket_and_prediction(
        self, monkeypatch, tmp_path
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(cli, "lookup_ticket",
                            lambda *a, **kw: calls.append("ticket"))
        monkeypatch.setattr(cli, "lookup_prediction",
                            lambda *a, **kw: calls.append("prediction"))

        # 4 (eval) → 5 (prediction details) → ticket_id → run_id → 0 (back) → 0 (exit)
        monkeypatch.setattr(
            "builtins.input",
            _make_inputs("4", "5", "abc123", "run_20260101_120000", "0", "0"),
        )
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

        assert "ticket" in calls
        assert "prediction" in calls
        assert calls.index("ticket") < calls.index("prediction"), (
            "lookup_ticket must be called before lookup_prediction"
        )

    def test_prediction_details_no_run_id_shows_ticket_only(
        self, monkeypatch, tmp_path
    ) -> None:
        """If run_id is empty, lookup_ticket is still called; lookup_prediction is not."""
        calls: list[str] = []
        monkeypatch.setattr(cli, "lookup_ticket",
                            lambda *a, **kw: calls.append("ticket"))
        monkeypatch.setattr(cli, "lookup_prediction",
                            lambda *a, **kw: calls.append("prediction"))

        # 4 → 5 → ticket_id → "" (no run_id) → 0 (back) → 0 (exit)
        monkeypatch.setattr(
            "builtins.input",
            _make_inputs("4", "5", "abc123", "", "0", "0"),
        )
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

        assert "ticket" in calls
        assert "prediction" not in calls


# ── Tests 14, 15: Leakage audit and config display ───────────────────────────

class TestLeakageAuditAndConfig:
    """Tests 14 & 15: Options 5 and 6 delegate to existing functions."""

    def test_option_5_delegates_to_leakage_audit(
        self, monkeypatch, tmp_path
    ) -> None:
        called = []
        monkeypatch.setattr(cli, "run_leakage_audit",
                            lambda *a, **kw: called.append("audit"))
        monkeypatch.setattr("builtins.input", _make_inputs("5", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")
        assert called == ["audit"]

    def test_option_6_delegates_to_show_config(
        self, monkeypatch, capsys, tmp_path
    ) -> None:
        called = []
        monkeypatch.setattr(cli, "show_config",
                            lambda *a, **kw: called.append("config"))
        monkeypatch.setattr("builtins.input", _make_inputs("6", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")
        assert called == ["config"]


# ── Test 16: Invalid and empty input ─────────────────────────────────────────

class TestInvalidInput:
    """Test 16: Invalid and empty input must not crash."""

    def test_invalid_main_menu_input_does_not_crash(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr("builtins.input", _make_inputs("99", "abc", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

    def test_empty_main_menu_input_does_not_crash(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr("builtins.input", _make_inputs("", "", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

    def test_invalid_triage_submenu_input_does_not_crash(
        self, monkeypatch, tmp_path
    ) -> None:
        # 1 (triage) → invalid → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("1", "99", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

    def test_invalid_eval_submenu_input_does_not_crash(
        self, monkeypatch, tmp_path
    ) -> None:
        # 4 (eval) → invalid → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("4", "99", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

    def test_invalid_reviewer_mode_does_not_crash(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(cli, "run_random_eval_ticket", _stub_noop)
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)
        # 1 (triage) → 1 (random) → invalid reviewer → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("1", "1", "X", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

    def test_missing_ticket_id_in_triage_does_not_crash(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(cli, "run_specific_ticket", _stub_noop)
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)
        # 1 (triage) → 2 (specific) → "" (no id) → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("1", "2", "", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")


# ── Test 17: config.yaml not written ─────────────────────────────────────────

class TestConfigYamlNotWritten:
    """Test 17: Reviewer selection never writes config.yaml."""

    def test_reviewer_off_does_not_write_config_yaml(
        self, monkeypatch, tmp_path
    ) -> None:
        """apply_reviewer_override must never open config.yaml for writing."""
        monkeypatch.setattr(cli, "run_random_eval_ticket", _stub_noop)
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)

        write_calls: list[str] = []
        real_open = open

        def patched_open(path, mode="r", *args, **kwargs):
            if "config.yaml" in str(path) and "w" in mode:
                write_calls.append(str(path))
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", patched_open)

        # 1 (triage) → 1 (random) → 1 (reviewer off) → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("1", "1", "1", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

        assert write_calls == [], (
            f"config.yaml must never be opened for writing; got: {write_calls}"
        )

    def test_reviewer_on_does_not_write_config_yaml(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(cli, "run_random_eval_ticket", _stub_noop)
        monkeypatch.setattr(cli, "load_embedding_model", _stub_embedding)

        write_calls: list[str] = []
        real_open = open

        def patched_open(path, mode="r", *args, **kwargs):
            if "config.yaml" in str(path) and "w" in mode:
                write_calls.append(str(path))
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", patched_open)

        # 1 (triage) → 1 (random) → 2 (conditional reviewer) → 0 (back) → 0 (exit)
        monkeypatch.setattr("builtins.input", _make_inputs("1", "1", "2", "0", "0"))
        _demo_loop(_nav_cfg(), str(tmp_path / "t.duckdb"), "data/lancedb")

        assert write_calls == []


# ── _select_reviewer_mode unit tests ─────────────────────────────────────────

class TestSelectReviewerMode:
    """Direct unit tests for the _select_reviewer_mode helper in cli.py."""

    def test_choice_1_returns_reviewer_disabled(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "1")
        result = _select_reviewer_mode(_nav_cfg())
        assert result is not None
        assert result["reviewer"]["enabled"] is False

    def test_choice_2_with_model_returns_reviewer_enabled(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "2")
        result = _select_reviewer_mode(_nav_cfg())
        assert result is not None
        assert result["reviewer"]["enabled"] is True

    def test_choice_2_without_model_returns_none(
        self, monkeypatch, capsys
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "2")
        cfg = _nav_cfg()
        cfg["reviewer"]["model_name"] = ""
        result = _select_reviewer_mode(cfg)
        assert result is None
        out = capsys.readouterr().out
        assert "No reviewer model" in out or "not configured" in out.lower()

    def test_choice_0_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "0")
        result = _select_reviewer_mode(_nav_cfg())
        assert result is None

    def test_empty_input_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "")
        result = _select_reviewer_mode(_nav_cfg())
        assert result is None

    def test_invalid_input_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "X")
        result = _select_reviewer_mode(_nav_cfg())
        assert result is None

    def test_does_not_mutate_original_cfg(self, monkeypatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "2")
        cfg = _nav_cfg()
        original_enabled = cfg["reviewer"]["enabled"]
        _select_reviewer_mode(cfg)
        assert cfg["reviewer"]["enabled"] == original_enabled


# ══════════════════════════════════════════════════════════════════════════════
# Curated leaderboard tests
# ══════════════════════════════════════════════════════════════════════════════

import src.application.cli_menu as _cli_menu_curated_mod
from src.application.cli_menu import show_curated_leaderboard


def _make_curated_db(tmp_path) -> tuple[str, list[str]]:
    """
    Return (db_path, run_ids) with triage_runs and triage_metrics seeded
    for four distinct runs used by curated leaderboard tests.

    Two runs have reviewer enabled, two have it disabled.
    One run is designated as the featured run.
    """
    path = str(tmp_path / "curated.duckdb")
    run_ids = [
        "run_20260712_215929_ab_off",
        "run_20260712_215929_ab_on",
        "run_20260711_200952",
        "run_20260710_032230",
    ]
    conn = duckdb.connect(path)
    conn.execute("""
        CREATE TABLE triage_runs (
            run_id           TEXT PRIMARY KEY,
            created_at       TIMESTAMP,
            model_name       TEXT,
            triage_model     TEXT,
            reviewer_enabled BOOLEAN,
            reviewer_model   TEXT,
            split_name       TEXT,
            limit_n          INTEGER,
            runtime_seconds  DOUBLE
        )
    """)
    data = [
        (run_ids[0], "llama3.2:3b", False, "",             200, 585.0),
        (run_ids[1], "llama3.2:3b", True,  "qwen3.5:9b",  200, 910.0),
        (run_ids[2], "llama3.2:3b", True,  "granite4.1:8b", 200, 709.0),
        (run_ids[3], "llama3.2:3b", False, "",             200, 600.0),
    ]
    for rid, model, rev_enabled, rev_model, limit, runtime in data:
        conn.execute(
            "INSERT INTO triage_runs VALUES (?, NOW(), ?, ?, ?, ?, 'eval', ?, ?)",
            [rid, model, model, rev_enabled, rev_model, limit, runtime],
        )
    conn.execute("""
        CREATE TABLE triage_metrics (
            run_id        TEXT,
            metric_group  TEXT,
            metric_name   TEXT,
            metric_value  DOUBLE,
            created_at    TIMESTAMP
        )
    """)
    for rid in run_ids:
        for name, value in [
            ("urgency_accuracy",            0.55),
            ("urgency_macro_f1",            0.54),
            ("topic_proxy_accuracy",        0.59),
            ("topic_macro_f1",              0.55),
            ("next_action_proxy_agreement", 0.20),
            ("human_review_rate",           0.40),
            ("average_confidence",          0.83),
            ("avg_seconds_per_ticket",      3.5),
            ("reviewer_invocation_rate",    0.23 if "ab_on" in rid or rid == run_ids[2] else 0.0),
        ]:
            conn.execute(
                "INSERT INTO triage_metrics VALUES (?, 'group', ?, ?, NOW())",
                [rid, name, value],
            )
    conn.close()
    return path, run_ids


def _curated_cfg(db_path: str = "", featured: str = "run_20260711_200952") -> dict:
    """Minimal config with a leaderboard section for curated leaderboard tests."""
    return {
        "leaderboard": {
            "controlled_reviewer_run_ids": [
                "run_20260712_215929_ab_off",
                "run_20260712_215929_ab_on",
            ],
            "featured_reviewer_run_ids": [
                "run_20260711_200952",
            ],
            "analyzer_screening_run_ids": [
                "run_20260710_032230",
            ],
            "featured_run_id": featured,
        }
    }


class TestCuratedLeaderboard:
    """
    Focused tests for show_curated_leaderboard and the curated submenu routing.

    All database access uses seeded tmp_path DuckDB files.
    No Ollama, embedding model, or LanceDB calls.
    """

    # Test 1: Top-level six-option menu remains unchanged
    def test_top_level_menu_has_exactly_six_numbered_options(self) -> None:
        from cli import MENU as CLI_MENU
        for i in range(7):
            assert f"{i}." in CLI_MENU, f"Option {i}. missing from top-level menu"
        assert "7." not in CLI_MENU

    # Test 2: Evaluation submenu contains curated and full-history options
    def test_eval_submenu_contains_curated_option(self) -> None:
        from cli import _EVAL_MENU
        assert "Curated evaluation" in _EVAL_MENU or "curated" in _EVAL_MENU.lower()

    def test_eval_submenu_contains_full_history_option(self) -> None:
        from cli import _EVAL_MENU
        assert "Full experiment history" in _EVAL_MENU or "full experiment" in _EVAL_MENU.lower()

    # Test 3: Curated groups read run IDs from config, not hardcoded
    def test_curated_groups_read_from_config(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        cfg = _curated_cfg()
        # Use a different controlled run ID in config — only that ID should appear
        cfg["leaderboard"]["controlled_reviewer_run_ids"] = ["run_20260712_215929_ab_off"]
        show_curated_leaderboard(db_path, cfg)
        out = capsys.readouterr().out
        assert "20260712_215929_ab_off" in out

    # Test 4: Configured ordering is preserved
    def test_section_a_order_matches_config(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        cfg = _curated_cfg()
        show_curated_leaderboard(db_path, cfg)
        out = capsys.readouterr().out
        pos_off = out.find("20260712_215929_ab_off")
        pos_on  = out.find("20260712_215929_ab_on")
        assert pos_off < pos_on, "ab_off must appear before ab_on (configured order)"

    # Test 5: Missing run IDs are skipped safely
    def test_missing_run_id_skipped_with_warning(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        cfg = _curated_cfg()
        cfg["leaderboard"]["controlled_reviewer_run_ids"].append("run_does_not_exist")
        show_curated_leaderboard(db_path, cfg)
        out = capsys.readouterr().out
        assert "Warning" in out or "warning" in out.lower() or "not found" in out.lower()

    def test_missing_run_id_does_not_crash(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        cfg = _curated_cfg()
        cfg["leaderboard"]["analyzer_screening_run_ids"] = ["run_totally_absent"]
        show_curated_leaderboard(db_path, cfg)  # must not raise

    # Test 6: Featured run is marked
    def test_featured_run_is_marked(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        cfg = _curated_cfg(featured="run_20260711_200952")
        show_curated_leaderboard(db_path, cfg)
        out = capsys.readouterr().out
        assert "★" in out, "Featured run must be marked with ★"

    # Test 7: Curated view retains urgency accuracy and F1
    def test_curated_view_shows_urgency_acc(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        show_curated_leaderboard(db_path, _curated_cfg())
        out = capsys.readouterr().out
        assert "Urg Acc" in out or "urg" in out.lower()
        assert "55.0%" in out  # 0.55 → 55.0%

    def test_curated_view_shows_urgency_f1(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        show_curated_leaderboard(db_path, _curated_cfg())
        out = capsys.readouterr().out
        assert "Urg F1" in out or "f1" in out.lower()

    # Test 8: Curated view retains topic accuracy and F1
    def test_curated_view_shows_topic_acc(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        show_curated_leaderboard(db_path, _curated_cfg())
        out = capsys.readouterr().out
        assert "Top Acc" in out or "topic" in out.lower()

    def test_curated_view_shows_topic_f1(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        show_curated_leaderboard(db_path, _curated_cfg())
        out = capsys.readouterr().out
        assert "Top F1" in out or "topic" in out.lower()

    # Test 9: Curated view retains action agreement, HR rate, reviewer rate, confidence, sec/ticket
    def test_curated_view_shows_action_agreement(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        show_curated_leaderboard(db_path, _curated_cfg())
        out = capsys.readouterr().out
        assert "Action" in out or "action" in out.lower()

    def test_curated_view_shows_human_review_rate(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        show_curated_leaderboard(db_path, _curated_cfg())
        out = capsys.readouterr().out
        assert "HR Rate" in out or "human" in out.lower()

    def test_curated_view_shows_reviewer_rate(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        show_curated_leaderboard(db_path, _curated_cfg())
        out = capsys.readouterr().out
        assert "Rev Rate" in out or "reviewer" in out.lower()

    def test_curated_view_shows_avg_confidence(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        show_curated_leaderboard(db_path, _curated_cfg())
        out = capsys.readouterr().out
        assert "Avg Conf" in out or "confidence" in out.lower()

    def test_curated_view_shows_sec_per_ticket(self, capsys, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        show_curated_leaderboard(db_path, _curated_cfg())
        out = capsys.readouterr().out
        assert "Sec/Tick" in out or "sec" in out.lower()

    # Test 10: Full-history view (option 2) preserves existing behavior
    def test_full_history_shows_all_seeded_runs(self, capsys, tmp_path) -> None:
        db_path, run_ids = _make_curated_db(tmp_path)
        show_kpi_leaderboard(db_path)
        out = capsys.readouterr().out
        for rid in run_ids:
            assert rid[:18] in out, f"run_id {rid[:18]} missing from full history"

    # Test 11: No database write statements executed
    def test_no_db_write_during_curated_view(self, monkeypatch, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        write_calls: list[str] = []
        real_connect = duckdb.connect

        def patched_connect(path, read_only=False, **kwargs):
            if not read_only and path == db_path:
                write_calls.append(path)
            return real_connect(path, read_only=read_only, **kwargs)

        monkeypatch.setattr(duckdb, "connect", patched_connect)
        show_curated_leaderboard(db_path, _curated_cfg())
        assert write_calls == [], (
            f"show_curated_leaderboard opened DuckDB in write mode: {write_calls}"
        )

    # Test 12: config.yaml is not rewritten at runtime
    def test_config_yaml_not_rewritten(self, monkeypatch, tmp_path) -> None:
        db_path, _ = _make_curated_db(tmp_path)
        write_calls: list[str] = []
        real_open = open

        def patched_open(path, mode="r", *args, **kwargs):
            if "config.yaml" in str(path) and "w" in mode:
                write_calls.append(str(path))
            return real_open(path, mode, *args, **kwargs)

        monkeypatch.setattr("builtins.open", patched_open)
        show_curated_leaderboard(db_path, _curated_cfg())
        assert write_calls == []

    # Test 13: Wide output — two-table fallback avoids wrapping
    def test_two_table_fallback_each_section(self, capsys, tmp_path) -> None:
        """
        Each section prints two tables.  Verify both Table 1 and Table 2 headers
        appear for the controlled section (Section A).
        """
        db_path, _ = _make_curated_db(tmp_path)
        show_curated_leaderboard(db_path, _curated_cfg())
        out = capsys.readouterr().out
        # Both table headers must be present
        assert "Table 1" in out or "Configuration" in out
        assert "Table 2" in out or "Quality" in out

    # Fallback: absent leaderboard section falls back to full history
    def test_no_leaderboard_section_falls_back_to_full_history(
        self, monkeypatch, capsys, tmp_path
    ) -> None:
        db_path, run_ids = _make_curated_db(tmp_path)
        called = []
        monkeypatch.setattr(
            _cli_menu_curated_mod,
            "show_kpi_leaderboard",
            lambda db: called.append("full"),
        )
        show_curated_leaderboard(db_path, {})  # empty cfg — no leaderboard section
        assert called == ["full"]
