"""
Tests for Sprint 6B — evaluation_repository.py.

Uses a temporary DuckDB file via pytest's tmp_path fixture.
No Ollama, embeddings, or LanceDB calls.

Test coverage:
  - create_tables() creates all four required tables
  - insert_run() stores run metadata that can be read back
  - insert_predictions() stores all result rows
  - insert_metrics() stores scalar KPIs with correct group assignments
  - insert_confusion_matrix() stores confusion rows
"""

import pytest

from src.infrastructure.evaluation_repository import EvaluationRepository, _metric_group


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_repo(tmp_path) -> EvaluationRepository:
    db = str(tmp_path / "test_eval.duckdb")
    repo = EvaluationRepository(db)
    repo.create_tables()
    return repo


def _make_run_metadata(run_id: str = "run_20260101_120000") -> dict:
    return {
        "run_id":          run_id,
        "created_at":      None,
        "model_name":      "qwen2.5:7b",
        "embedding_model": "multilingual-e5-large",
        "dataset_path":    "data/tickets.csv",
        "split_name":      "eval",
        "limit_n":         10,
        "top_k":           5,
        "temperature":     0.1,
        "max_retries":     1,
        "runtime_seconds": 42.5,
        "config_json":     "{}",
    }


def _make_prediction_row(ticket_id: str = "t001") -> dict:
    return {
        "ticket_id":               ticket_id,
        "text_snippet":            "Cannot log in.",
        "topic":                   "Technical / Online Access",
        "urgency":                 "High",
        "next_action":             "escalate_to_human_supervisor",
        "confidence":              0.82,
        "missing_info":            True,
        "requires_human_review":   True,
        "short_note":              "Login blocked.",
        "action_status":           "simulated_success",
        "action_target":           "human_supervisor_queue",
        "action_note":             "Escalated.",
        "actual_queue":            "Technical Support",
        "actual_priority":         "high",
        "actual_type":             "Incident",
        "proxy_topic":             "Technical / Online Access",
        "proxy_urgency":           "High",
        "proxy_next_action":       "forward_to_technical_support",
        "proxy_topic_source":      "queue_mapping",
        # reviewer fields
        "reviewer_used":           True,
        "reviewer_model":          "qwen3-coder:30b",
        "reviewer_changed_topic":  False,
        "reviewer_changed_urgency": True,
        "reviewer_seconds":        1.23,
        "first_topic":             "Other",
        "first_urgency":           "Medium",
        "first_confidence":        0.55,
        "reviewer_trigger_flags":  '["low_llm_confidence"]',
        # explainability fields
        "first_short_note":              "First analyzer: login access issue.",
        "reviewer_note":                 "Reviewer kept technical topic after evidence check.",
        "validator_flags":               '["low_llm_confidence", "topic_disagreement"]',
        "validator_notes":               '["Confidence below threshold."]',
        "neighbor_predicted_topic":      "Technical / Online Access",
        "neighbor_topic_confidence":     0.74,
        "neighbor_predicted_priority":   "high",
        "neighbor_priority_confidence":  0.69,
    }


# ─── Table creation ───────────────────────────────────────────────────────────

class TestCreateTables:

    def test_creates_triage_runs(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        tables = _get_table_names(repo)
        repo.close()
        assert "triage_runs" in tables

    def test_creates_triage_predictions(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        tables = _get_table_names(repo)
        repo.close()
        assert "triage_predictions" in tables

    def test_creates_triage_metrics(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        tables = _get_table_names(repo)
        repo.close()
        assert "triage_metrics" in tables

    def test_creates_triage_confusion_matrix(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        tables = _get_table_names(repo)
        repo.close()
        assert "triage_confusion_matrix" in tables

    def test_create_tables_idempotent(self, tmp_path) -> None:
        """Calling create_tables() twice must not raise."""
        repo = _make_repo(tmp_path)
        repo.create_tables()
        repo.close()


def _get_table_names(repo: EvaluationRepository) -> list[str]:
    rows = repo.conn.execute("SHOW TABLES").fetchall()
    return [row[0] for row in rows]


# ─── insert_run ───────────────────────────────────────────────────────────────

class TestInsertRun:

    def test_run_is_stored(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        repo.insert_run(_make_run_metadata("run_test_001"))
        count = repo.conn.execute(
            "SELECT COUNT(*) FROM triage_runs WHERE run_id = ?", ["run_test_001"]
        ).fetchone()[0]
        repo.close()
        assert count == 1

    def test_model_name_is_stored(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        meta = _make_run_metadata("run_test_002")
        meta["model_name"] = "test-model"
        repo.insert_run(meta)
        name = repo.conn.execute(
            "SELECT model_name FROM triage_runs WHERE run_id = ?", ["run_test_002"]
        ).fetchone()[0]
        repo.close()
        assert name == "test-model"

    def test_runtime_seconds_is_stored(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        meta = _make_run_metadata("run_test_003")
        meta["runtime_seconds"] = 99.9
        repo.insert_run(meta)
        secs = repo.conn.execute(
            "SELECT runtime_seconds FROM triage_runs WHERE run_id = ?", ["run_test_003"]
        ).fetchone()[0]
        repo.close()
        assert abs(secs - 99.9) < 0.01

    def test_insert_or_replace_does_not_raise_on_duplicate(self, tmp_path) -> None:
        """INSERT OR REPLACE: inserting the same run_id twice must not raise."""
        repo = _make_repo(tmp_path)
        repo.insert_run(_make_run_metadata("run_dup"))
        repo.insert_run(_make_run_metadata("run_dup"))  # should replace, not error
        count = repo.conn.execute(
            "SELECT COUNT(*) FROM triage_runs WHERE run_id = ?", ["run_dup"]
        ).fetchone()[0]
        repo.close()
        assert count == 1


# ─── insert_predictions ───────────────────────────────────────────────────────

class TestInsertPredictions:

    def test_all_rows_stored(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        rows = [_make_prediction_row(f"t{i}") for i in range(5)]
        repo.insert_predictions("run_x", rows)
        count = repo.conn.execute(
            "SELECT COUNT(*) FROM triage_predictions WHERE run_id = ?", ["run_x"]
        ).fetchone()[0]
        repo.close()
        assert count == 5

    def test_predicted_topic_mapped_from_topic(self, tmp_path) -> None:
        """Result row key 'topic' is stored as 'predicted_topic'."""
        repo = _make_repo(tmp_path)
        row = _make_prediction_row("t001")
        row["topic"] = "Billing / Payment"
        repo.insert_predictions("run_y", [row])
        topic = repo.conn.execute(
            "SELECT predicted_topic FROM triage_predictions WHERE run_id = ? AND ticket_id = ?",
            ["run_y", "t001"],
        ).fetchone()[0]
        repo.close()
        assert topic == "Billing / Payment"

    def test_proxy_columns_stored(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        repo.insert_predictions("run_z", [_make_prediction_row("t001")])
        proxy_topic = repo.conn.execute(
            "SELECT proxy_topic FROM triage_predictions WHERE run_id = ?", ["run_z"]
        ).fetchone()[0]
        repo.close()
        assert proxy_topic == "Technical / Online Access"


# ─── insert_metrics ───────────────────────────────────────────────────────────

class TestInsertMetrics:

    def _sample_metrics(self) -> dict:
        return {
            "urgency_accuracy":            0.75,
            "urgency_macro_f1":            0.70,
            "topic_proxy_accuracy":        0.60,
            "next_action_proxy_agreement": 0.55,
            "human_review_rate":           0.20,
            "average_confidence":          0.82,
        }

    def test_all_metrics_stored(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        metrics = self._sample_metrics()
        repo.insert_metrics("run_m1", metrics)
        count = repo.conn.execute(
            "SELECT COUNT(*) FROM triage_metrics WHERE run_id = ?", ["run_m1"]
        ).fetchone()[0]
        repo.close()
        assert count == len(metrics)

    def test_metric_value_stored_correctly(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        repo.insert_metrics("run_m2", {"urgency_accuracy": 0.75})
        value = repo.conn.execute(
            "SELECT metric_value FROM triage_metrics WHERE run_id = ? AND metric_name = ?",
            ["run_m2", "urgency_accuracy"],
        ).fetchone()[0]
        repo.close()
        assert abs(value - 0.75) < 0.0001

    def test_non_float_values_skipped(self, tmp_path) -> None:
        """Dict or list values must not be stored in triage_metrics."""
        repo = _make_repo(tmp_path)
        repo.insert_metrics("run_m3", {"urgency_accuracy": 0.5, "nested": {"a": 1}})
        count = repo.conn.execute(
            "SELECT COUNT(*) FROM triage_metrics WHERE run_id = ?", ["run_m3"]
        ).fetchone()[0]
        repo.close()
        assert count == 1  # only urgency_accuracy


class TestMetricGroup:

    def test_urgency_prefix(self) -> None:
        assert _metric_group("urgency_accuracy") == "urgency"
        assert _metric_group("urgency_macro_f1") == "urgency"

    def test_topic_prefix(self) -> None:
        assert _metric_group("topic_proxy_accuracy") == "topic"
        assert _metric_group("topic_macro_f1") == "topic"

    def test_next_action_prefix(self) -> None:
        assert _metric_group("next_action_proxy_agreement") == "next_action"

    def test_operational_default(self) -> None:
        assert _metric_group("human_review_rate") == "operational"
        assert _metric_group("average_confidence") == "operational"
        assert _metric_group("escalation_rate") == "operational"


# ─── insert_confusion_matrix ─────────────────────────────────────────────────

class TestInsertConfusionMatrix:

    def test_rows_stored(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        confusion_rows = [
            {"actual_label": "High", "predicted_label": "High", "count": 3},
            {"actual_label": "High", "predicted_label": "Low",  "count": 1},
        ]
        repo.insert_confusion_matrix("run_c1", "urgency", confusion_rows)
        count = repo.conn.execute(
            "SELECT COUNT(*) FROM triage_confusion_matrix WHERE run_id = ? AND target_name = ?",
            ["run_c1", "urgency"],
        ).fetchone()[0]
        repo.close()
        assert count == 2

    def test_target_name_stored(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        repo.insert_confusion_matrix(
            "run_c2",
            "topic_proxy",
            [{"actual_label": "Billing / Payment", "predicted_label": "Other", "count": 2}],
        )
        target = repo.conn.execute(
            "SELECT target_name FROM triage_confusion_matrix WHERE run_id = ?", ["run_c2"]
        ).fetchone()[0]
        repo.close()
        assert target == "topic_proxy"

    def test_count_stored_correctly(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        repo.insert_confusion_matrix(
            "run_c3",
            "urgency",
            [{"actual_label": "Medium", "predicted_label": "High", "count": 7}],
        )
        cnt = repo.conn.execute(
            "SELECT count FROM triage_confusion_matrix WHERE run_id = ? AND actual_label = ?",
            ["run_c3", "Medium"],
        ).fetchone()[0]
        repo.close()
        assert cnt == 7

    def test_empty_confusion_rows_stores_nothing(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        repo.insert_confusion_matrix("run_c4", "urgency", [])
        count = repo.conn.execute(
            "SELECT COUNT(*) FROM triage_confusion_matrix WHERE run_id = ?", ["run_c4"]
        ).fetchone()[0]
        repo.close()
        assert count == 0


# ─── Reviewer columns in triage_predictions ───────────────────────────────────

def _get_column_names(repo: EvaluationRepository) -> list[str]:
    """Return column names of triage_predictions as a list."""
    rows = repo.conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'triage_predictions'"
    ).fetchall()
    return [r[0] for r in rows]


class TestReviewerColumnsExist:
    """create_tables() must add all reviewer columns to triage_predictions."""

    def test_reviewer_used_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "reviewer_used" in cols

    def test_reviewer_model_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "reviewer_model" in cols

    def test_reviewer_changed_topic_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "reviewer_changed_topic" in cols

    def test_reviewer_changed_urgency_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "reviewer_changed_urgency" in cols

    def test_reviewer_seconds_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "reviewer_seconds" in cols

    def test_first_topic_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "first_topic" in cols

    def test_first_urgency_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "first_urgency" in cols

    def test_first_confidence_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "first_confidence" in cols

    def test_reviewer_trigger_flags_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "reviewer_trigger_flags" in cols


class TestInsertPredictionsReviewerFields:
    """insert_predictions() must persist all reviewer fields."""

    def _insert_and_fetch(self, tmp_path) -> dict:
        repo = _make_repo(tmp_path)
        row = _make_prediction_row("rv001")
        repo.insert_predictions("run_rv", [row])
        cols = [d[0] for d in repo.conn.execute(
            "SELECT * FROM triage_predictions WHERE run_id = ? AND ticket_id = ?",
            ["run_rv", "rv001"],
        ).description]
        values = repo.conn.execute(
            "SELECT * FROM triage_predictions WHERE run_id = ? AND ticket_id = ?",
            ["run_rv", "rv001"],
        ).fetchone()
        repo.close()
        return dict(zip(cols, values))

    def test_reviewer_used_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["reviewer_used"] is True

    def test_reviewer_model_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["reviewer_model"] == "qwen3-coder:30b"

    def test_reviewer_changed_topic_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["reviewer_changed_topic"] is False

    def test_reviewer_changed_urgency_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["reviewer_changed_urgency"] is True

    def test_reviewer_seconds_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert abs(pred["reviewer_seconds"] - 1.23) < 0.001

    def test_first_topic_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["first_topic"] == "Other"

    def test_first_urgency_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["first_urgency"] == "Medium"

    def test_first_confidence_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert abs(pred["first_confidence"] - 0.55) < 0.001

    def test_reviewer_trigger_flags_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["reviewer_trigger_flags"] == '["low_llm_confidence"]'

    def test_reviewer_used_defaults_to_false_when_absent(self, tmp_path) -> None:
        """A prediction row with no reviewer keys stores False for reviewer_used."""
        repo = _make_repo(tmp_path)
        minimal_row = {
            "ticket_id": "t_min",
            "text_snippet": "Test.",
            "topic": "Other",
            "urgency": "Low",
            "next_action": "ask_for_more_information",
            "confidence": 0.5,
            "missing_info": False,
            "requires_human_review": False,
            "short_note": "",
            "action_status": "",
            "action_target": "",
            "action_note": "",
            "actual_queue": "",
            "actual_priority": "",
            "actual_type": "",
            "proxy_topic": "",
            "proxy_urgency": "",
            "proxy_next_action": "",
            "proxy_topic_source": "",
        }
        repo.insert_predictions("run_min", [minimal_row])
        rv = repo.conn.execute(
            "SELECT reviewer_used FROM triage_predictions WHERE ticket_id = ?", ["t_min"]
        ).fetchone()[0]
        repo.close()
        assert rv is False


# ─── triage_runs: new reviewer identity columns ───────────────────────────────

def _get_run_column_names(repo: EvaluationRepository) -> list[str]:
    """Return column names of triage_runs as a list."""
    rows = repo.conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'triage_runs'"
    ).fetchall()
    return [r[0] for r in rows]


class TestTriageRunsReviewerColumns:
    """create_tables() must add triage_model, reviewer_enabled, reviewer_model to triage_runs."""

    def test_triage_model_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_run_column_names(repo)
        repo.close()
        assert "triage_model" in cols

    def test_reviewer_enabled_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_run_column_names(repo)
        repo.close()
        assert "reviewer_enabled" in cols

    def test_reviewer_model_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_run_column_names(repo)
        repo.close()
        assert "reviewer_model" in cols


class TestInsertRunReviewerFields:
    """insert_run() must persist triage_model, reviewer_enabled, reviewer_model."""

    def test_triage_model_stored(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        meta = _make_run_metadata("run_tm_001")
        meta["triage_model"] = "qwen2.5:7b"
        repo.insert_run(meta)
        val = repo.conn.execute(
            "SELECT triage_model FROM triage_runs WHERE run_id = ?", ["run_tm_001"]
        ).fetchone()[0]
        repo.close()
        assert val == "qwen2.5:7b"

    def test_reviewer_enabled_false_stored_when_no_reviewer(self, tmp_path) -> None:
        """When reviewer_enabled is absent from metadata, it defaults to False."""
        repo = _make_repo(tmp_path)
        meta = _make_run_metadata("run_re_off")
        # do not set reviewer_enabled — should default to False
        repo.insert_run(meta)
        val = repo.conn.execute(
            "SELECT reviewer_enabled FROM triage_runs WHERE run_id = ?", ["run_re_off"]
        ).fetchone()[0]
        repo.close()
        assert val is False

    def test_reviewer_enabled_true_stored(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        meta = _make_run_metadata("run_re_on")
        meta["reviewer_enabled"] = True
        meta["reviewer_model"]   = "devstral:24b"
        repo.insert_run(meta)
        val = repo.conn.execute(
            "SELECT reviewer_enabled FROM triage_runs WHERE run_id = ?", ["run_re_on"]
        ).fetchone()[0]
        repo.close()
        assert val is True

    def test_reviewer_model_stored_when_reviewer_enabled(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        meta = _make_run_metadata("run_rm_001")
        meta["reviewer_enabled"] = True
        meta["reviewer_model"]   = "devstral:24b"
        repo.insert_run(meta)
        val = repo.conn.execute(
            "SELECT reviewer_model FROM triage_runs WHERE run_id = ?", ["run_rm_001"]
        ).fetchone()[0]
        repo.close()
        assert val == "devstral:24b"

    def test_reviewer_model_empty_when_reviewer_disabled(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        meta = _make_run_metadata("run_rm_off")
        meta["reviewer_enabled"] = False
        meta["reviewer_model"]   = ""
        repo.insert_run(meta)
        val = repo.conn.execute(
            "SELECT reviewer_model FROM triage_runs WHERE run_id = ?", ["run_rm_off"]
        ).fetchone()[0]
        repo.close()
        assert val == ""

    def test_triage_model_falls_back_to_model_name(self, tmp_path) -> None:
        """When triage_model is absent, insert_run falls back to model_name."""
        repo = _make_repo(tmp_path)
        meta = _make_run_metadata("run_fb_001")
        meta["model_name"] = "fallback-model"
        # triage_model intentionally absent
        repo.insert_run(meta)
        val = repo.conn.execute(
            "SELECT triage_model FROM triage_runs WHERE run_id = ?", ["run_fb_001"]
        ).fetchone()[0]
        repo.close()
        assert val == "fallback-model"


# ─── Reviewer wiring in main._build_agent and cli_menu.build_agent_from_config ─

class TestReviewerWiring:
    """_build_agent and build_agent_from_config must wire the reviewer from config."""

    def _make_minimal_cfg(self, reviewer_enabled: bool) -> dict:
        cfg = {
            "llm": {
                "base_url":        "http://localhost:11434",
                "model_name":      "test-model",
                "temperature":     0.1,
                "timeout_seconds": 60,
                "max_retries":     1,
            },
            "vector_store": {"path": "data/lancedb", "table_name": "ticket_embeddings"},
            "retrieval":    {"top_k": 5},
            "thresholds":   {"low_confidence": 0.60},
            "reviewer": {
                "enabled":       reviewer_enabled,
                "model_name":    "reviewer-model",
                "base_url":      "http://localhost:11434",
                "temperature":   0.1,
                "max_retries":   1,
                "trigger_flags": ["low_llm_confidence", "urgency_disagreement"],
            },
        }
        return cfg

    def test_reviewer_none_when_disabled(self) -> None:
        """When reviewer.enabled=false, agent.reviewer must be None."""
        import sys, os
        sys.path.insert(0, os.path.abspath("."))
        import main as m
        from unittest.mock import MagicMock

        cfg = self._make_minimal_cfg(reviewer_enabled=False)
        fake_model = MagicMock()
        agent = m._build_agent(cfg, fake_model)
        assert agent.reviewer is None

    def test_reviewer_not_none_when_enabled(self) -> None:
        """When reviewer.enabled=true, agent.reviewer must be a ConditionalLLMReviewer."""
        import sys, os
        sys.path.insert(0, os.path.abspath("."))
        import main as m
        from unittest.mock import MagicMock
        from src.application.reviewer import ConditionalLLMReviewer

        cfg = self._make_minimal_cfg(reviewer_enabled=True)
        fake_model = MagicMock()
        agent = m._build_agent(cfg, fake_model)
        assert isinstance(agent.reviewer, ConditionalLLMReviewer)

    def test_reviewer_trigger_flags_wired_from_config(self) -> None:
        """ConditionalLLMReviewer.trigger_flags must match config trigger_flags."""
        import sys, os
        sys.path.insert(0, os.path.abspath("."))
        import main as m
        from unittest.mock import MagicMock

        cfg = self._make_minimal_cfg(reviewer_enabled=True)
        fake_model = MagicMock()
        agent = m._build_agent(cfg, fake_model)
        assert "low_llm_confidence" in agent.reviewer.trigger_flags
        assert "urgency_disagreement" in agent.reviewer.trigger_flags

    def test_reviewer_model_name_wired_from_config(self) -> None:
        import sys, os
        sys.path.insert(0, os.path.abspath("."))
        import main as m
        from unittest.mock import MagicMock

        cfg = self._make_minimal_cfg(reviewer_enabled=True)
        fake_model = MagicMock()
        agent = m._build_agent(cfg, fake_model)
        assert agent.reviewer.model_name == "reviewer-model"


# ─── Explainability columns in triage_predictions ─────────────────────────────

class TestExplainabilityColumnsExist:
    """create_tables() must add all explainability columns to triage_predictions."""

    def test_first_short_note_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "first_short_note" in cols

    def test_reviewer_note_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "reviewer_note" in cols

    def test_validator_flags_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "validator_flags" in cols

    def test_validator_notes_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "validator_notes" in cols

    def test_neighbor_predicted_topic_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "neighbor_predicted_topic" in cols

    def test_neighbor_topic_confidence_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "neighbor_topic_confidence" in cols

    def test_neighbor_predicted_priority_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "neighbor_predicted_priority" in cols

    def test_neighbor_priority_confidence_column_exists(self, tmp_path) -> None:
        repo = _make_repo(tmp_path)
        cols = _get_column_names(repo)
        repo.close()
        assert "neighbor_priority_confidence" in cols


class TestInsertPredictionsExplainabilityFields:
    """insert_predictions() must persist all explainability fields."""

    def _insert_and_fetch(self, tmp_path) -> dict:
        repo = _make_repo(tmp_path)
        row = _make_prediction_row("ex001")
        repo.insert_predictions("run_ex", [row])
        cols = [d[0] for d in repo.conn.execute(
            "SELECT * FROM triage_predictions WHERE run_id = ? AND ticket_id = ?",
            ["run_ex", "ex001"],
        ).description]
        values = repo.conn.execute(
            "SELECT * FROM triage_predictions WHERE run_id = ? AND ticket_id = ?",
            ["run_ex", "ex001"],
        ).fetchone()
        repo.close()
        return dict(zip(cols, values))

    def test_first_short_note_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["first_short_note"] == "First analyzer: login access issue."

    def test_reviewer_note_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["reviewer_note"] == "Reviewer kept technical topic after evidence check."

    def test_validator_flags_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["validator_flags"] == '["low_llm_confidence", "topic_disagreement"]'

    def test_validator_notes_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["validator_notes"] == '["Confidence below threshold."]'

    def test_neighbor_predicted_topic_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["neighbor_predicted_topic"] == "Technical / Online Access"

    def test_neighbor_topic_confidence_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert abs(pred["neighbor_topic_confidence"] - 0.74) < 0.001

    def test_neighbor_predicted_priority_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert pred["neighbor_predicted_priority"] == "high"

    def test_neighbor_priority_confidence_stored(self, tmp_path) -> None:
        pred = self._insert_and_fetch(tmp_path)
        assert abs(pred["neighbor_priority_confidence"] - 0.69) < 0.001

    def test_reviewer_note_empty_when_reviewer_not_used(self, tmp_path) -> None:
        """A row with reviewer_used=False should store empty reviewer_note."""
        repo = _make_repo(tmp_path)
        row = _make_prediction_row("ex_noreview")
        row["reviewer_used"]  = False
        row["reviewer_note"]  = ""
        row["first_short_note"] = "Analyzer note only."
        repo.insert_predictions("run_nr", [row])
        pred = repo.conn.execute(
            "SELECT reviewer_note, first_short_note FROM triage_predictions "
            "WHERE ticket_id = ?", ["ex_noreview"]
        ).fetchone()
        repo.close()
        assert pred[0] == ""
        assert pred[1] == "Analyzer note only."
