"""
Tests for Sprint 6A — BatchRunner.

All tests are fully deterministic.
No Ollama, embeddings, LanceDB, or DuckDB calls.

Test coverage:
  - BatchRunner processes multiple ticket records.
  - Result rows contain all assignment-required fields.
  - Evaluation / reference metadata columns are preserved.
  - actual_* and proxy_* values do NOT appear in the TicketInput passed to the agent.
  - action_status, action_target, action_note are extracted from ActionExecutionResult.
  - missing_fields is preserved as a list in the result row.
  - BatchRunner returns one row per input record.
  Streaming (trace_path):
  - Without trace_path, no file is created.
  - With trace_path, JSONL file is created before any ticket is processed.
  - One line is appended per ticket immediately after processing.
  - Final line count equals number of processed tickets.
"""

import json
import os

import pytest

from src.application.batch_runner import BatchRunner, _build_result_row
from src.application.agent import TicketTriageAgent
from src.domain.enums import NextAction, Topic, Urgency
from src.domain.models import (
    ActionExecutionResult,
    LLMAnalysis,
    NeighborPrediction,
    TicketInput,
    TriageResult,
    ValidationResult,
)


# ─── Fixed test values ────────────────────────────────────────────────────────

_FIXED_NEIGHBOR_PREDICTION = NeighborPrediction(
    predicted_queue="Technical Support",
    queue_confidence=0.75,
    predicted_priority="high",
    priority_confidence=0.70,
    predicted_proxy_topic="Technical / Online Access",
    proxy_topic_confidence=0.72,
    predicted_tags=["login"],
    neighbors=[],
)

_FIXED_ANALYSIS = LLMAnalysis(
    topic=Topic.TECHNICAL,
    urgency=Urgency.HIGH,
    missing_info=True,
    missing_fields=["customer_identifier"],
    confidence=0.82,
    short_note="Customer blocked from portal.",
)

_FIXED_VALIDATION = ValidationResult(
    is_valid=True,
    requires_human_review=True,
    flags=["high_urgency"],
    notes=[],
)

_FIXED_NEXT_ACTION = NextAction.ESCALATE

_FIXED_ACTION_RESULT = ActionExecutionResult(
    selected_action=NextAction.ESCALATE,
    action_status="simulated_success",
    action_note="Escalated to supervisor.",
    target="human_supervisor_queue",
)


# ─── Fake agent ───────────────────────────────────────────────────────────────

class FakeAgent:
    """
    Returns a fixed TriageResult for every ticket.
    Records which TicketInputs were received so tests can verify
    that actual_*/proxy_* values never enter the ticket passed to the agent.
    """

    def __init__(self) -> None:
        self.received_tickets: list[TicketInput] = []

    def process_ticket(self, ticket: TicketInput) -> TriageResult:
        self.received_tickets.append(ticket)
        return TriageResult(
            ticket_id=ticket.ticket_id,
            text_snippet=ticket.text_snippet,
            topic=_FIXED_ANALYSIS.topic,
            urgency=_FIXED_ANALYSIS.urgency,
            next_action=_FIXED_NEXT_ACTION,
            confidence=_FIXED_ANALYSIS.confidence,
            missing_info=_FIXED_ANALYSIS.missing_info,
            missing_fields=_FIXED_ANALYSIS.missing_fields,
            requires_human_review=_FIXED_VALIDATION.requires_human_review,
            short_note=_FIXED_ANALYSIS.short_note,
            action_result=_FIXED_ACTION_RESULT,
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_record(ticket_id: str = "eval_001") -> dict:
    """Build a minimal ticket record as returned by _fetch_batch_tickets."""
    return {
        "ticket_id":             ticket_id,
        "subject":               "Cannot log in",
        "body":                  "I keep getting an error when I try to log in.",
        "raw_text":              "Cannot log in I keep getting an error.",
        "cleaned_text":          "Cannot log in I keep getting an error.",
        "representation_text":   "Subject: Cannot log in\n\nBody: I keep getting an error.",
        "text_snippet":          "Cannot log in I keep getting an error.",
        "actual_queue":          "Technical Support",
        "actual_priority":       "high",
        "actual_type":           "Incident",
        "actual_tags_json":      '["login", "portal"]',
        "proxy_topic":           "Technical / Online Access",
        "proxy_urgency":         "High",
        "proxy_next_action":     "forward_to_technical_support",
        "proxy_topic_source":    "queue_mapping",
    }


def _make_runner() -> tuple[BatchRunner, FakeAgent]:
    fake_agent = FakeAgent()
    runner     = BatchRunner(fake_agent)
    return runner, fake_agent


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestBatchRunnerBasic:

    def test_returns_one_row_per_record(self) -> None:
        runner, _ = _make_runner()
        records = [_make_record("t1"), _make_record("t2"), _make_record("t3")]
        rows = runner.process_tickets(records)
        assert len(rows) == 3

    def test_empty_input_returns_empty_list(self) -> None:
        runner, _ = _make_runner()
        rows = runner.process_tickets([])
        assert rows == []

    def test_ticket_id_matches_input_record(self) -> None:
        runner, _ = _make_runner()
        rows = runner.process_tickets([_make_record("eval_xyz")])
        assert rows[0]["ticket_id"] == "eval_xyz"

    def test_agent_called_once_per_record(self) -> None:
        runner, fake_agent = _make_runner()
        runner.process_tickets([_make_record("t1"), _make_record("t2")])
        assert len(fake_agent.received_tickets) == 2


class TestAssignmentRequiredFields:
    """Result rows must contain every assignment-required field."""

    def _get_row(self) -> dict:
        runner, _ = _make_runner()
        return runner.process_tickets([_make_record()])[0]

    def test_has_ticket_id(self) -> None:
        assert "ticket_id" in self._get_row()

    def test_has_text_snippet(self) -> None:
        assert "text_snippet" in self._get_row()

    def test_has_topic(self) -> None:
        assert "topic" in self._get_row()

    def test_has_urgency(self) -> None:
        assert "urgency" in self._get_row()

    def test_has_next_action(self) -> None:
        assert "next_action" in self._get_row()

    def test_has_short_note(self) -> None:
        assert "short_note" in self._get_row()

    def test_has_confidence(self) -> None:
        assert "confidence" in self._get_row()

    def test_has_missing_info(self) -> None:
        assert "missing_info" in self._get_row()

    def test_has_missing_fields(self) -> None:
        assert "missing_fields" in self._get_row()

    def test_has_requires_human_review(self) -> None:
        assert "requires_human_review" in self._get_row()

    def test_has_action_status(self) -> None:
        assert "action_status" in self._get_row()

    def test_has_action_target(self) -> None:
        assert "action_target" in self._get_row()

    def test_has_action_note(self) -> None:
        assert "action_note" in self._get_row()


class TestEvaluationMetadata:
    """Evaluation metadata from the source record must appear in result rows."""

    def _get_row(self) -> dict:
        runner, _ = _make_runner()
        return runner.process_tickets([_make_record()])[0]

    def test_actual_queue_preserved(self) -> None:
        assert self._get_row()["actual_queue"] == "Technical Support"

    def test_actual_priority_preserved(self) -> None:
        assert self._get_row()["actual_priority"] == "high"

    def test_actual_type_preserved(self) -> None:
        assert self._get_row()["actual_type"] == "Incident"

    def test_proxy_topic_preserved(self) -> None:
        assert self._get_row()["proxy_topic"] == "Technical / Online Access"

    def test_proxy_urgency_preserved(self) -> None:
        assert self._get_row()["proxy_urgency"] == "High"

    def test_proxy_next_action_preserved(self) -> None:
        assert self._get_row()["proxy_next_action"] == "forward_to_technical_support"

    def test_proxy_topic_source_preserved(self) -> None:
        assert self._get_row()["proxy_topic_source"] == "queue_mapping"


class TestNoLabelLeakIntoAgent:
    """actual_* and proxy_* must NOT appear in the TicketInput sent to the agent."""

    def test_actual_queue_not_in_ticket_input(self) -> None:
        runner, fake_agent = _make_runner()
        runner.process_tickets([_make_record()])
        ticket = fake_agent.received_tickets[0]
        assert not hasattr(ticket, "actual_queue")

    def test_actual_priority_not_in_ticket_input(self) -> None:
        runner, fake_agent = _make_runner()
        runner.process_tickets([_make_record()])
        ticket = fake_agent.received_tickets[0]
        assert not hasattr(ticket, "actual_priority")

    def test_proxy_topic_not_in_ticket_input(self) -> None:
        runner, fake_agent = _make_runner()
        runner.process_tickets([_make_record()])
        ticket = fake_agent.received_tickets[0]
        assert not hasattr(ticket, "proxy_topic")

    def test_ticket_input_has_only_text_fields(self) -> None:
        runner, fake_agent = _make_runner()
        runner.process_tickets([_make_record()])
        ticket = fake_agent.received_tickets[0]
        allowed = {
            "ticket_id", "subject", "body", "raw_text",
            "cleaned_text", "representation_text", "text_snippet",
        }
        # Pydantic model_fields holds the declared fields
        declared = set(ticket.model_fields.keys())
        assert declared == allowed


class TestMissingFields:

    def test_missing_fields_is_list_in_result(self) -> None:
        runner, _ = _make_runner()
        rows = runner.process_tickets([_make_record()])
        assert isinstance(rows[0]["missing_fields"], list)

    def test_missing_fields_contains_expected_value(self) -> None:
        runner, _ = _make_runner()
        rows = runner.process_tickets([_make_record()])
        assert "customer_identifier" in rows[0]["missing_fields"]


class TestActionResultFields:

    def test_action_status_from_action_result(self) -> None:
        runner, _ = _make_runner()
        rows = runner.process_tickets([_make_record()])
        assert rows[0]["action_status"] == "simulated_success"

    def test_action_target_from_action_result(self) -> None:
        runner, _ = _make_runner()
        rows = runner.process_tickets([_make_record()])
        assert rows[0]["action_target"] == "human_supervisor_queue"

    def test_action_note_from_action_result(self) -> None:
        runner, _ = _make_runner()
        rows = runner.process_tickets([_make_record()])
        assert rows[0]["action_note"] == "Escalated to supervisor."

    def test_action_fields_empty_when_no_action_result(self) -> None:
        """If agent returns no action_result, action fields default to empty string."""
        class NoActionAgent:
            def process_ticket(self, ticket: TicketInput) -> TriageResult:
                return TriageResult(
                    ticket_id=ticket.ticket_id,
                    text_snippet=ticket.text_snippet,
                    topic=Topic.OTHER,
                    urgency=Urgency.LOW,
                    next_action=NextAction.ASK_MORE_INFO,
                    confidence=0.5,
                    missing_info=False,
                    missing_fields=[],
                    requires_human_review=False,
                    short_note="No action.",
                    action_result=None,
                )

        runner = BatchRunner(NoActionAgent())
        rows = runner.process_tickets([_make_record()])
        assert rows[0]["action_status"] == ""
        assert rows[0]["action_target"] == ""
        assert rows[0]["action_note"]   == ""


class TestStreamingTrace:
    """Verify that trace_path causes incremental JSONL writes during processing."""

    def test_without_trace_path_no_file_created(self, tmp_path) -> None:
        runner, _ = _make_runner()
        runner.process_tickets([_make_record()])
        # No trace file should be created when trace_path is not given.
        assert not any(tmp_path.iterdir())

    def test_with_trace_path_file_is_created(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        runner, _ = _make_runner()
        runner.process_tickets([_make_record()], trace_path=path)
        assert os.path.exists(path)

    def test_line_count_equals_record_count(self, tmp_path) -> None:
        import json
        path = str(tmp_path / "trace.jsonl")
        runner, _ = _make_runner()
        records = [_make_record(ticket_id=f"t{i}") for i in range(4)]
        runner.process_tickets(records, trace_path=path)
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 4

    def test_all_lines_are_valid_json(self, tmp_path) -> None:
        import json
        path = str(tmp_path / "trace.jsonl")
        runner, _ = _make_runner()
        runner.process_tickets([_make_record(ticket_id=f"t{i}") for i in range(3)], trace_path=path)
        with open(path, encoding="utf-8") as f:
            for raw in f:
                if raw.strip():
                    json.loads(raw)  # must not raise

    def test_trace_file_truncated_on_new_run(self, tmp_path) -> None:
        """A second call to process_tickets with the same path must start fresh."""
        import json
        path = str(tmp_path / "trace.jsonl")
        runner, _ = _make_runner()
        runner.process_tickets([_make_record(ticket_id="first")], trace_path=path)
        runner.process_tickets([_make_record(ticket_id="second")], trace_path=path)
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["ticket_id"] == "second"

    def test_returns_all_rows_regardless_of_trace_path(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        runner, _ = _make_runner()
        records = [_make_record(ticket_id=f"t{i}") for i in range(5)]
        rows = runner.process_tickets(records, trace_path=path)
        assert len(rows) == 5


class TestRunSummaryDistributions:
    """Verify that _write_run_summary computes correct distributions."""

    def test_topic_distribution(self, tmp_path) -> None:
        """topic_distribution must count tickets by topic value."""
        import json, sys, os
        sys.path.insert(0, os.path.abspath("."))
        import main as m

        rows = [
            {"topic": "Technical / Online Access", "urgency": "High",   "next_action": "escalate_to_human_supervisor",       "requires_human_review": True,  "missing_info": True},
            {"topic": "Technical / Online Access", "urgency": "Medium", "next_action": "forward_to_technical_support",        "requires_human_review": False, "missing_info": False},
            {"topic": "Billing / Payment",         "urgency": "Low",    "next_action": "forward_to_billing_team",             "requires_human_review": False, "missing_info": False},
        ]
        path = str(tmp_path / "summary.json")
        cfg  = {"llm": {"model_name": "test-model"}, "embedding": {"model_name": "test-emb"}, "batch": {"split": "eval", "limit": 3}}
        m._write_run_summary(path, rows, cfg)

        with open(path, encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["topic_distribution"]["Technical / Online Access"] == 2
        assert summary["topic_distribution"]["Billing / Payment"] == 1

    def test_human_review_rate(self, tmp_path) -> None:
        import json, sys, os
        sys.path.insert(0, os.path.abspath("."))
        import main as m

        rows = [
            {"topic": "Other", "urgency": "Low", "next_action": "ask_for_more_information", "requires_human_review": True,  "missing_info": False},
            {"topic": "Other", "urgency": "Low", "next_action": "ask_for_more_information", "requires_human_review": False, "missing_info": False},
            {"topic": "Other", "urgency": "Low", "next_action": "ask_for_more_information", "requires_human_review": False, "missing_info": False},
            {"topic": "Other", "urgency": "Low", "next_action": "ask_for_more_information", "requires_human_review": False, "missing_info": False},
        ]
        path = str(tmp_path / "summary.json")
        cfg  = {"llm": {"model_name": "m"}, "embedding": {"model_name": "e"}, "batch": {"split": "eval", "limit": 4}}
        m._write_run_summary(path, rows, cfg)

        with open(path, encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["human_review_rate"] == 0.25

    def test_number_processed(self, tmp_path) -> None:
        import json, sys, os
        sys.path.insert(0, os.path.abspath("."))
        import main as m

        rows = [
            {"topic": "Other", "urgency": "Low", "next_action": "ask_for_more_information", "requires_human_review": False, "missing_info": False},
        ]
        path = str(tmp_path / "summary.json")
        cfg  = {"llm": {"model_name": "m"}, "embedding": {"model_name": "e"}, "batch": {"split": "eval", "limit": 1}}
        m._write_run_summary(path, rows, cfg)

        with open(path, encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["number_processed"] == 1


# ─── Timed agent path ─────────────────────────────────────────────────────────

class FakeTimedAgent:
    """
    Agent that implements process_ticket_timed.
    Returns a fixed TriageResult and a fixed timing dict.
    """

    TIMING = {
        "retrieval_seconds":        0.12,
        "llm_seconds":              1.50,
        "validation_seconds":       0.01,
        "routing_seconds":          0.001,
        "action_execution_seconds": 0.002,
        "total_ticket_seconds":     1.633,
    }

    def process_ticket(self, ticket: TicketInput) -> TriageResult:
        return TriageResult(
            ticket_id=ticket.ticket_id,
            text_snippet=ticket.text_snippet,
            topic=Topic.OTHER,
            urgency=Urgency.LOW,
            next_action=NextAction.ASK_MORE_INFO,
            confidence=0.5,
            missing_info=False,
            missing_fields=[],
            requires_human_review=False,
            short_note="",
            action_result=None,
        )

    def process_ticket_timed(
        self, ticket: TicketInput
    ) -> tuple[TriageResult, dict]:
        return self.process_ticket(ticket), dict(self.TIMING)


class TestTimedAgentPath:
    """BatchRunner uses process_ticket_timed when the agent exposes it."""

    def _run(self) -> dict:
        runner = BatchRunner(FakeTimedAgent())
        rows = runner.process_tickets([_make_record()])
        return rows[0]

    def test_retrieval_seconds_in_result_row(self) -> None:
        assert self._run()["retrieval_seconds"] == 0.12

    def test_llm_seconds_in_result_row(self) -> None:
        assert self._run()["llm_seconds"] == 1.50

    def test_total_ticket_seconds_in_result_row(self) -> None:
        assert self._run()["total_ticket_seconds"] == 1.633

    def test_validation_seconds_in_result_row(self) -> None:
        assert self._run()["validation_seconds"] == 0.01

    def test_routing_seconds_in_result_row(self) -> None:
        assert self._run()["routing_seconds"] == 0.001

    def test_action_execution_seconds_in_result_row(self) -> None:
        assert self._run()["action_execution_seconds"] == 0.002

    def test_fallback_agent_has_zero_timing(self) -> None:
        """FakeAgent (no process_ticket_timed) produces zero timing fields."""
        runner, _ = _make_runner()
        row = runner.process_tickets([_make_record()])[0]
        assert row["total_ticket_seconds"] == 0.0
        assert row["retrieval_seconds"] == 0.0
        assert row["llm_seconds"] == 0.0


# ─── Reviewer metadata fields ──────────────────────────────────────────────────

class FakeTimedAgentWithReviewer:
    """
    Agent that includes reviewer trace fields in its timing dict.
    Used to verify that BatchRunner passes reviewer metadata to result rows.
    """

    TIMING = {
        "retrieval_seconds":          0.10,
        "llm_seconds":                1.20,
        "validation_seconds":         0.01,
        "reviewer_seconds":           0.95,
        "routing_seconds":            0.001,
        "action_execution_seconds":   0.002,
        "total_ticket_seconds":       2.263,
        "reviewer_used":              True,
        "reviewer_model":             "devstral-small-2:24b",
        "reviewer_changed_topic":     True,
        "reviewer_changed_urgency":   False,
        # Pre-reviewer LLM outputs stored by the agent before reviewer runs.
        "first_topic":                "Technical / Online Access",
        "first_urgency":              "High",
        "first_confidence":           0.55,
    }

    def process_ticket_timed(
        self, ticket: TicketInput
    ) -> tuple[TriageResult, dict]:
        result = TriageResult(
            ticket_id=ticket.ticket_id,
            text_snippet=ticket.text_snippet,
            topic=Topic.BILLING,
            urgency=Urgency.MEDIUM,
            next_action=NextAction.ASK_MORE_INFO,
            confidence=0.78,
            missing_info=False,
            missing_fields=[],
            requires_human_review=False,
            short_note="Reviewer revised topic.",
            action_result=None,
        )
        return result, dict(self.TIMING)


class TestReviewerMetadataFields:
    """Result rows must include reviewer trace fields from the timing dict."""

    def _run(self) -> dict:
        runner = BatchRunner(FakeTimedAgentWithReviewer())
        return runner.process_tickets([_make_record()])[0]

    def test_reviewer_used_in_result_row(self) -> None:
        assert self._run()["reviewer_used"] is True

    def test_reviewer_model_in_result_row(self) -> None:
        assert self._run()["reviewer_model"] == "devstral-small-2:24b"

    def test_reviewer_changed_topic_in_result_row(self) -> None:
        assert self._run()["reviewer_changed_topic"] is True

    def test_reviewer_changed_urgency_in_result_row(self) -> None:
        assert self._run()["reviewer_changed_urgency"] is False

    def test_reviewer_seconds_in_result_row(self) -> None:
        assert self._run()["reviewer_seconds"] == pytest.approx(0.95)

    def test_first_topic_in_result_row(self) -> None:
        assert self._run()["first_topic"] == "Technical / Online Access"

    def test_first_urgency_in_result_row(self) -> None:
        assert self._run()["first_urgency"] == "High"

    def test_first_confidence_in_result_row(self) -> None:
        assert self._run()["first_confidence"] == pytest.approx(0.55)

    def test_reviewer_trigger_flags_in_result_row(self) -> None:
        runner = BatchRunner(FakeTimedAgentWithReviewer())
        row = runner.process_tickets([_make_record()])[0]
        # FakeTimedAgentWithReviewer does not set reviewer_trigger_flags,
        # so it should default to "[]".
        assert "reviewer_trigger_flags" in row

    def test_reviewer_trigger_flags_default_empty_json(self) -> None:
        """When timing dict has no reviewer_trigger_flags, defaults to '[]'."""
        runner, _ = _make_runner()
        row = runner.process_tickets([_make_record()])[0]
        assert row["reviewer_trigger_flags"] == "[]"

    def test_reviewer_fields_default_false_when_not_in_timing(self) -> None:
        """When timing dict has no reviewer keys (legacy agent), defaults to False/0.0."""
        runner, _ = _make_runner()   # FakeAgent has no reviewer fields
        row = runner.process_tickets([_make_record()])[0]
        assert row["reviewer_used"]            is False
        assert row["reviewer_model"]           == ""
        assert row["reviewer_changed_topic"]   is False
        assert row["reviewer_changed_urgency"] is False
        assert row["reviewer_seconds"]         == 0.0
        assert row["first_topic"]              == ""
        assert row["first_urgency"]            == ""
        assert row["first_confidence"]         == 0.0


class TestConsoleTerminology:
    """Console output must use human_review and reviewer as separate fields."""

    def _capture_output(self, runner: BatchRunner) -> str:
        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            runner.process_tickets([_make_record()])
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def test_decision_line_has_human_review_field(self) -> None:
        runner, _ = _make_runner()
        output = self._capture_output(runner)
        assert "human_review=" in output

    def test_decision_line_has_reviewer_field(self) -> None:
        runner, _ = _make_runner()
        output = self._capture_output(runner)
        assert "reviewer=" in output

    def test_decision_line_does_not_use_ambiguous_review_equals(self) -> None:
        """The old ambiguous 'review=yes/no' must not appear."""
        runner, _ = _make_runner()
        output = self._capture_output(runner)
        # Allowed: human_review=... or reviewer=...
        # Not allowed: bare ', review=' which was the old ambiguous format.
        lines = [l for l in output.splitlines() if "Processing ticket" in l]
        for line in lines:
            assert ", review=" not in line

    def test_reviewer_no_when_reviewer_not_used(self) -> None:
        runner, _ = _make_runner()  # FakeAgent — reviewer_used=False
        output = self._capture_output(runner)
        lines = [l for l in output.splitlines() if "Processing ticket" in l]
        assert any("reviewer=no" in l for l in lines)


# ─── New explainability fields in result rows ─────────────────────────────────

class FakeTimedAgentWithExplainability:
    """
    Agent that includes all new explainability and neighbor evidence fields
    in its timing dict.
    """

    TIMING = {
        "retrieval_seconds":          0.10,
        "llm_seconds":                1.20,
        "validation_seconds":         0.01,
        "reviewer_seconds":           0.95,
        "routing_seconds":            0.001,
        "action_execution_seconds":   0.002,
        "total_ticket_seconds":       2.263,
        "reviewer_used":              True,
        "reviewer_model":             "devstral:24b",
        "reviewer_changed_topic":     True,
        "reviewer_changed_urgency":   False,
        "reviewer_trigger_flags":     '["low_llm_confidence"]',
        "first_topic":                "Other",
        "first_urgency":              "Medium",
        "first_confidence":           0.55,
        # New explainability fields.
        "first_short_note":           "First analyzer note: possibly billing issue.",
        "reviewer_note":              "Reviewer revised to billing after checking evidence.",
        "validator_flags":            '["low_llm_confidence", "topic_disagreement"]',
        "validator_notes":            '["Confidence below threshold."]',
        # Neighbor retrieval evidence.
        "neighbor_predicted_topic":    "Billing / Payment",
        "neighbor_topic_confidence":   0.73,
        "neighbor_predicted_priority": "medium",
        "neighbor_priority_confidence": 0.68,
    }

    def process_ticket_timed(
        self, ticket: TicketInput
    ) -> tuple[TriageResult, dict]:
        result = TriageResult(
            ticket_id=ticket.ticket_id,
            text_snippet=ticket.text_snippet,
            topic=Topic.BILLING,
            urgency=Urgency.MEDIUM,
            next_action=NextAction.ASK_MORE_INFO,
            confidence=0.78,
            missing_info=False,
            missing_fields=[],
            requires_human_review=True,
            short_note="Final: billing invoice dispute.",
            action_result=None,
        )
        return result, dict(self.TIMING)


class TestExplainabilityFieldsInResultRow:
    """Result rows must include all explainability and neighbor evidence fields."""

    def _run(self) -> dict:
        runner = BatchRunner(FakeTimedAgentWithExplainability())
        return runner.process_tickets([_make_record()])[0]

    def test_first_short_note_in_result_row(self) -> None:
        assert self._run()["first_short_note"] == "First analyzer note: possibly billing issue."

    def test_reviewer_note_in_result_row(self) -> None:
        assert self._run()["reviewer_note"] == "Reviewer revised to billing after checking evidence."

    def test_validator_flags_in_result_row(self) -> None:
        assert self._run()["validator_flags"] == '["low_llm_confidence", "topic_disagreement"]'

    def test_validator_notes_in_result_row(self) -> None:
        assert self._run()["validator_notes"] == '["Confidence below threshold."]'

    def test_neighbor_predicted_topic_in_result_row(self) -> None:
        assert self._run()["neighbor_predicted_topic"] == "Billing / Payment"

    def test_neighbor_topic_confidence_in_result_row(self) -> None:
        assert self._run()["neighbor_topic_confidence"] == pytest.approx(0.73)

    def test_neighbor_predicted_priority_in_result_row(self) -> None:
        assert self._run()["neighbor_predicted_priority"] == "medium"

    def test_neighbor_priority_confidence_in_result_row(self) -> None:
        assert self._run()["neighbor_priority_confidence"] == pytest.approx(0.68)

    def test_explainability_fields_default_empty_when_absent(self) -> None:
        """FakeAgent (no timing dict) — all new fields default to safe values."""
        runner, _ = _make_runner()
        row = runner.process_tickets([_make_record()])[0]
        assert row["first_short_note"]            == ""
        assert row["reviewer_note"]               == ""
        assert row["validator_flags"]             == "[]"
        assert row["validator_notes"]             == "[]"
        assert row["neighbor_predicted_topic"]    == ""
        assert row["neighbor_topic_confidence"]   == 0.0
        assert row["neighbor_predicted_priority"] == ""
        assert row["neighbor_priority_confidence"] == 0.0


# ─── Console output: validator flags and reviewer block ───────────────────────

def _capture(runner: BatchRunner, records: list[dict] | None = None) -> str:
    import io, sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        runner.process_tickets(records or [_make_record()])
    finally:
        sys.stdout = old
    return buf.getvalue()


class TestConsoleValidatorFlags:
    """
    Validator flags must be printed selectively — only for human_review or reviewer cases.
    Normal clean tickets must remain one compact line.
    """

    def test_normal_clean_ticket_no_validator_flags_line(self) -> None:
        """A ticket with human_review=no and reviewer=no must produce one compact line."""
        class CleanFakeAgent:
            def process_ticket_timed(self, ticket: TicketInput):
                result = TriageResult(
                    ticket_id=ticket.ticket_id,
                    text_snippet=ticket.text_snippet,
                    topic=Topic.OTHER,
                    urgency=Urgency.LOW,
                    next_action=NextAction.ASK_MORE_INFO,
                    confidence=0.85,
                    missing_info=False,
                    missing_fields=[],
                    requires_human_review=False,
                    short_note="Clean ticket.",
                    action_result=None,
                )
                timing = {
                    "reviewer_used": False,
                    "requires_human_review": False,
                    "validator_flags": "[]",
                    "reviewer_trigger_flags": "[]",
                }
                return result, timing

        output = _capture(BatchRunner(CleanFakeAgent()))
        lines = output.splitlines()
        # Only the decision line — no validator flags line.
        assert len(lines) == 1
        assert "validator flags" not in output

    def test_human_review_ticket_prints_validator_flags_line(self) -> None:
        """human_review=yes but reviewer=no: must print validator flags line."""
        class HumanReviewAgent:
            def process_ticket_timed(self, ticket: TicketInput):
                result = TriageResult(
                    ticket_id=ticket.ticket_id,
                    text_snippet=ticket.text_snippet,
                    topic=Topic.CLAIMS,
                    urgency=Urgency.HIGH,
                    next_action=NextAction.ESCALATE,
                    confidence=0.60,
                    missing_info=False,
                    missing_fields=[],
                    requires_human_review=True,
                    short_note="High urgency claim.",
                    action_result=None,
                )
                timing = {
                    "reviewer_used": False,
                    "validator_flags": '["low_llm_confidence"]',
                    "reviewer_trigger_flags": "[]",
                    "validator_notes": "[]",
                    "neighbor_predicted_topic": "",
                    "neighbor_topic_confidence": 0.0,
                    "neighbor_predicted_priority": "",
                    "neighbor_priority_confidence": 0.0,
                }
                return result, timing

        output = _capture(BatchRunner(HumanReviewAgent()))
        assert "validator flags:" in output
        assert "low_llm_confidence" in output

    def test_reviewer_ticket_no_full_block_prints_both_flag_lines(self) -> None:
        """reviewer=yes but log_reviewer_events=False: print validator flags + trigger flags."""
        runner = BatchRunner(
            FakeTimedAgentWithExplainability(),
            log_decisions=True,
            log_reviewer_events=False,
        )
        output = _capture(runner)
        assert "validator flags" in output
        assert "trigger flags" in output
        # Full reviewer block must NOT appear.
        assert "REVIEWER used:" not in output

    def test_reviewer_ticket_full_block_when_log_reviewer_events(self) -> None:
        """reviewer=yes and log_reviewer_events=True: print full reviewer block."""
        runner = BatchRunner(
            FakeTimedAgentWithExplainability(),
            log_decisions=True,
            log_reviewer_events=True,
        )
        output = _capture(runner)
        assert "REVIEWER used:" in output
        assert "validator flags" in output
        assert "trigger flags" in output
        assert "neighbor evidence:" in output
        assert "before:" in output
        assert "after :" in output
        assert "note  :" in output
        assert "changed:" in output

    def test_reviewer_block_contains_reviewer_note(self) -> None:
        runner = BatchRunner(
            FakeTimedAgentWithExplainability(),
            log_decisions=True,
            log_reviewer_events=True,
        )
        output = _capture(runner)
        assert "Reviewer revised to billing after checking evidence." in output

    def test_reviewer_block_contains_neighbor_evidence(self) -> None:
        runner = BatchRunner(
            FakeTimedAgentWithExplainability(),
            log_decisions=True,
            log_reviewer_events=True,
        )
        output = _capture(runner)
        assert "Billing / Payment" in output
        assert "medium" in output

    def test_no_actual_labels_in_console_output(self) -> None:
        """actual_* and proxy_* values from the record must NOT appear in console output."""
        runner = BatchRunner(FakeTimedAgentWithExplainability())
        output = _capture(runner)
        # Record has actual_queue="Technical Support", actual_priority="high"
        # These must not appear in the live decision output.
        assert "actual_queue" not in output
        assert "actual_priority" not in output
        assert "proxy_topic" not in output

    def test_no_raw_llm_responses_in_console_output(self) -> None:
        """Raw LLM prompts or responses must never be printed."""
        runner = BatchRunner(FakeTimedAgentWithExplainability())
        output = _capture(runner)
        # Check that the prompt keywords do not appear in the raw form.
        assert "ALLOWED TOPICS" not in output
        assert "OUTPUT (JSON only" not in output
