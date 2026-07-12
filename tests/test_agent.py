"""
Tests for Sprint 5E — TicketTriageAgent.

All tests are fully deterministic.  No Ollama, embeddings, LanceDB, or
DuckDB calls.

Fake components return fixed values so that each test can assert on one
specific aspect of the agent's orchestration behaviour:

  - process_ticket calls all five components.
  - TriageResult fields come from the correct source components.
  - requires_human_review is copied from ValidationResult.
  - The agent duplicates no business logic.

FakeNeighborRetriever, FakeAnalyzer, FakeValidator, FakeRouter, and
FakeActionExecutor are minimal stubs.  Each records whether it was called
so tests can verify call order without inspecting private state.
"""

import pytest

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

_TICKET_ID   = "eval_001"
_TEXT_SNIPPET = "Customer cannot log in to the portal."

_FIXED_NEIGHBOR_PREDICTION = NeighborPrediction(
    predicted_queue="Technical Support",
    queue_confidence=0.75,
    predicted_priority="high",
    priority_confidence=0.70,
    predicted_proxy_topic="Technical / Online Access",
    proxy_topic_confidence=0.72,
    predicted_tags=["login", "portal"],
    neighbors=[],
)

_FIXED_ANALYSIS = LLMAnalysis(
    topic=Topic.TECHNICAL,
    urgency=Urgency.HIGH,
    missing_info=False,
    missing_fields=[],
    confidence=0.88,
    short_note="Customer blocked from portal after login failure.",
)

_FIXED_VALIDATION = ValidationResult(
    is_valid=True,
    requires_human_review=True,
    flags=["high_urgency_with_limited_confidence"],
    notes=[],
)

_FIXED_NEXT_ACTION = NextAction.ESCALATE

_FIXED_ACTION_RESULT = ActionExecutionResult(
    selected_action=NextAction.ESCALATE,
    action_status="simulated_success",
    action_note="Simulated action: escalate ticket to a human supervisor.",
    target="human_supervisor_queue",
)


# ─── Fake components ──────────────────────────────────────────────────────────

class FakeNeighborRetriever:
    """Returns a fixed NeighborPrediction and records the call."""

    def __init__(self) -> None:
        self.called = False
        self.received_ticket_id = None
        self.received_text = None

    def retrieve_and_predict(self, ticket_id: str, representation_text: str) -> NeighborPrediction:
        self.called = True
        self.received_ticket_id = ticket_id
        self.received_text = representation_text
        return _FIXED_NEIGHBOR_PREDICTION


class FakeAnalyzer:
    """Returns a fixed LLMAnalysis and records the call."""

    def __init__(self) -> None:
        self.called = False

    def analyze(self, ticket: TicketInput, neighbor_prediction: NeighborPrediction) -> LLMAnalysis:
        self.called = True
        return _FIXED_ANALYSIS


class FakeValidator:
    """Returns a fixed ValidationResult and records the call."""

    def __init__(self) -> None:
        self.called = False

    def validate(
        self,
        analysis: LLMAnalysis,
        neighbor_prediction: NeighborPrediction,
    ) -> ValidationResult:
        self.called = True
        return _FIXED_VALIDATION


class FakeRouter:
    """Returns a fixed NextAction and records the call."""

    def __init__(self) -> None:
        self.called = False

    def route(self, analysis: LLMAnalysis, validation: ValidationResult) -> NextAction:
        self.called = True
        return _FIXED_NEXT_ACTION


class FakeActionExecutor:
    """Returns a fixed ActionExecutionResult and records the call."""

    def __init__(self) -> None:
        self.called = False
        self.received_next_action = None

    def execute(
        self,
        next_action: NextAction,
        ticket_id: str,
        short_note: str | None = None,
    ) -> ActionExecutionResult:
        self.called = True
        self.received_next_action = next_action
        return _FIXED_ACTION_RESULT


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_ticket() -> TicketInput:
    """Build a minimal TicketInput for testing."""
    return TicketInput(
        ticket_id=_TICKET_ID,
        subject="Cannot log in to portal",
        body="I tried logging in but keep getting an error.",
        raw_text="Cannot log in to portal I tried logging in but keep getting an error.",
        cleaned_text="Cannot log in to portal I tried logging in but keep getting an error.",
        representation_text="Subject: Cannot log in to portal\n\nBody: I tried logging in.",
        text_snippet=_TEXT_SNIPPET,
    )


def _make_agent() -> tuple[TicketTriageAgent, FakeNeighborRetriever, FakeAnalyzer, FakeValidator, FakeRouter, FakeActionExecutor]:
    """Construct an agent backed by all fake components."""
    retriever = FakeNeighborRetriever()
    analyzer  = FakeAnalyzer()
    validator = FakeValidator()
    router    = FakeRouter()
    executor  = FakeActionExecutor()
    agent = TicketTriageAgent(
        neighbor_retriever=retriever,
        analyzer=analyzer,
        validator=validator,
        router=router,
        action_executor=executor,
    )
    return agent, retriever, analyzer, validator, router, executor


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestTicketTriageAgent:

    # ── All components are called ──────────────────────────────────────────────

    def test_all_components_are_called(self) -> None:
        """process_ticket must invoke all five components."""
        agent, retriever, analyzer, validator, router, executor = _make_agent()
        ticket = _make_ticket()

        agent.process_ticket(ticket)

        assert retriever.called,  "NeighborRetriever was not called"
        assert analyzer.called,   "Analyzer was not called"
        assert validator.called,  "Validator was not called"
        assert router.called,     "Router was not called"
        assert executor.called,   "ActionExecutor was not called"

    # ── Retriever receives ticket_id and representation_text ──────────────────

    def test_retriever_receives_correct_inputs(self) -> None:
        agent, retriever, *_ = _make_agent()
        ticket = _make_ticket()

        agent.process_ticket(ticket)

        assert retriever.received_ticket_id == ticket.ticket_id
        assert retriever.received_text      == ticket.representation_text

    # ── TriageResult fields come from the right sources ───────────────────────

    def test_triage_result_topic_from_llm_analysis(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.topic == _FIXED_ANALYSIS.topic

    def test_triage_result_urgency_from_llm_analysis(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.urgency == _FIXED_ANALYSIS.urgency

    def test_triage_result_confidence_from_llm_analysis(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.confidence == _FIXED_ANALYSIS.confidence

    def test_triage_result_short_note_from_llm_analysis(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.short_note == _FIXED_ANALYSIS.short_note

    def test_triage_result_missing_info_from_llm_analysis(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.missing_info == _FIXED_ANALYSIS.missing_info

    def test_triage_result_missing_fields_from_llm_analysis(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.missing_fields == _FIXED_ANALYSIS.missing_fields

    # ── next_action comes from the router ─────────────────────────────────────

    def test_triage_result_next_action_from_router(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.next_action == _FIXED_NEXT_ACTION

    # ── action_result comes from the executor ─────────────────────────────────

    def test_triage_result_action_result_from_executor(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.action_result == _FIXED_ACTION_RESULT

    def test_executor_receives_next_action_from_router(self) -> None:
        """The agent must pass the router's NextAction to the executor."""
        agent, _, _, _, _, executor = _make_agent()
        agent.process_ticket(_make_ticket())
        assert executor.received_next_action == _FIXED_NEXT_ACTION

    # ── requires_human_review comes from the validator ────────────────────────

    def test_requires_human_review_from_validator(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.requires_human_review == _FIXED_VALIDATION.requires_human_review

    # ── ticket_id and text_snippet come from the input ticket ─────────────────

    def test_triage_result_ticket_id_from_input(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.ticket_id == _TICKET_ID

    def test_triage_result_text_snippet_from_input(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert result.text_snippet == _TEXT_SNIPPET

    # ── Return type ───────────────────────────────────────────────────────────

    def test_process_ticket_returns_triage_result(self) -> None:
        agent, *_ = _make_agent()
        result = agent.process_ticket(_make_ticket())
        assert isinstance(result, TriageResult)

    # ── Timing dict keys ──────────────────────────────────────────────────────

    def test_timing_dict_contains_reviewer_seconds(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "reviewer_seconds" in timing

    def test_timing_dict_contains_reviewer_trace_fields(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "reviewer_used"            in timing
        assert "reviewer_model"           in timing
        assert "reviewer_changed_topic"   in timing
        assert "reviewer_changed_urgency" in timing

    def test_timing_reviewer_used_false_when_no_reviewer(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_used"] is False
        assert timing["reviewer_seconds"] == 0.0

    def test_timing_contains_first_topic(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "first_topic" in timing
        assert timing["first_topic"] == _FIXED_ANALYSIS.topic.value

    def test_timing_contains_first_urgency(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "first_urgency" in timing
        assert timing["first_urgency"] == _FIXED_ANALYSIS.urgency.value

    def test_timing_contains_first_confidence(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "first_confidence" in timing
        assert timing["first_confidence"] == _FIXED_ANALYSIS.confidence

    def test_timing_contains_reviewer_trigger_flags(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "reviewer_trigger_flags" in timing
        # When no reviewer is configured, trigger_flags defaults to empty JSON array.
        assert timing["reviewer_trigger_flags"] == "[]"


# ─── Reviewer integration tests ───────────────────────────────────────────────

# New fixed values for reviewer tests.

_REVIEWED_ANALYSIS = LLMAnalysis(
    topic=Topic.BILLING,            # changed from TECHNICAL
    urgency=Urgency.MEDIUM,         # changed from HIGH
    missing_info=False,
    missing_fields=[],
    confidence=0.80,                # above 0 — reviewer output is valid
    short_note="Reviewer revised: billing invoice issue, medium urgency.",
)

_FALLBACK_ANALYSIS = LLMAnalysis(
    topic=Topic.OTHER,
    urgency=Urgency.MEDIUM,
    missing_info=True,
    missing_fields=["reviewer_output_invalid"],
    confidence=0.0,                 # signals reviewer failure
    short_note="Reviewer LLM output could not be validated.",
)

_VALIDATION_WITH_TRIGGER_FLAGS = ValidationResult(
    is_valid=True,
    requires_human_review=True,
    flags=["low_llm_confidence", "topic_disagreement"],
    notes=[],
)

_VALIDATION_NO_TRIGGER_FLAGS = ValidationResult(
    is_valid=True,
    requires_human_review=True,
    flags=["missing_information"],   # not a trigger flag
    notes=[],
)


class FakeReviewer:
    """
    Controllable fake reviewer for agent tests.

    The agent now calls get_triggered_review_flags() to decide whether to invoke
    review().  should_review_result controls the returned trigger list:
      True  → returns ["low_llm_confidence"] (non-empty → reviewer is called)
      False → returns []                     (empty    → reviewer is skipped)

    Parameters
    ----------
    should_review_result:
        When True, get_triggered_review_flags returns a non-empty list so the
        agent calls review().  When False, returns [] so review() is skipped.
    review_result:
        Value returned by review().
    """

    def __init__(
        self,
        should_review_result: bool = True,
        review_result: LLMAnalysis | None = None,
    ) -> None:
        self.get_triggered_flags_called = False
        self.review_called              = False
        self._should_review             = should_review_result
        self._review_result             = review_result or _REVIEWED_ANALYSIS
        self.model_name                 = "fake-reviewer-model"
        # Non-empty list when should_review_result is True; empty when False.
        self._trigger_flags = ["low_llm_confidence"] if should_review_result else []

    def get_triggered_review_flags(
        self,
        validation: ValidationResult,
        first_analysis: LLMAnalysis,
    ) -> list[str]:
        self.get_triggered_flags_called = True
        return self._trigger_flags

    def should_review(self, validation: ValidationResult) -> bool:
        """Kept for backward compat — not called by the agent."""
        return self._should_review

    def review(
        self,
        ticket: TicketInput,
        neighbor_prediction: NeighborPrediction,
        first_analysis: LLMAnalysis,
        validation: ValidationResult,
    ) -> LLMAnalysis:
        self.review_called = True
        return self._review_result


def _make_agent_with_reviewer(
    reviewer: FakeReviewer,
    validation_result: ValidationResult | None = None,
) -> tuple[TicketTriageAgent, FakeNeighborRetriever, FakeAnalyzer, FakeValidator, FakeRouter, FakeActionExecutor, FakeReviewer]:
    """Construct an agent backed by fakes, with an explicit reviewer and optional validation result."""
    retriever = FakeNeighborRetriever()
    analyzer  = FakeAnalyzer()

    effective_validation = validation_result or _FIXED_VALIDATION

    class ConfigurableFakeValidator:
        def __init__(self) -> None:
            self.called = False
            self._result = effective_validation

        def validate(self, analysis: LLMAnalysis, neighbor_prediction: NeighborPrediction) -> ValidationResult:
            self.called = True
            return self._result

    validator = ConfigurableFakeValidator()
    router    = FakeRouter()
    executor  = FakeActionExecutor()
    agent = TicketTriageAgent(
        neighbor_retriever=retriever,
        analyzer=analyzer,
        validator=validator,
        router=router,
        action_executor=executor,
        reviewer=reviewer,
    )
    return agent, retriever, analyzer, validator, router, executor, reviewer


class TestTicketTriageAgentWithReviewer:

    # ── Reviewer not called when get_triggered_review_flags returns [] ───────

    def test_reviewer_not_called_when_no_trigger_flags_returned(self) -> None:
        reviewer = FakeReviewer(should_review_result=False)
        agent, *_ = _make_agent_with_reviewer(reviewer)

        agent.process_ticket(_make_ticket())

        # The agent must have consulted get_triggered_review_flags.
        assert reviewer.get_triggered_flags_called is True
        # Because the result was empty, review() must not have been called.
        assert reviewer.review_called is False

    # ── Reviewer not called when no trigger flags in validation ───────────────

    def test_reviewer_not_called_when_no_trigger_flags(self) -> None:
        reviewer = FakeReviewer(should_review_result=False)
        agent, *_ = _make_agent_with_reviewer(
            reviewer, validation_result=_VALIDATION_NO_TRIGGER_FLAGS
        )
        agent.process_ticket(_make_ticket())
        assert reviewer.review_called is False

    # ── Reviewer called when should_review returns True ───────────────────────

    def test_reviewer_called_when_should_review_true(self) -> None:
        reviewer = FakeReviewer(should_review_result=True)
        agent, *_ = _make_agent_with_reviewer(reviewer)

        agent.process_ticket(_make_ticket())

        assert reviewer.review_called is True

    # ── Reviewed analysis replaces first when confidence > 0 ─────────────────

    def test_reviewed_analysis_used_when_confidence_above_zero(self) -> None:
        reviewer = FakeReviewer(
            should_review_result=True,
            review_result=_REVIEWED_ANALYSIS,  # confidence=0.80
        )
        agent, *_ = _make_agent_with_reviewer(reviewer)
        result = agent.process_ticket(_make_ticket())

        assert result.topic      == _REVIEWED_ANALYSIS.topic
        assert result.urgency    == _REVIEWED_ANALYSIS.urgency
        assert result.confidence == _REVIEWED_ANALYSIS.confidence

    # ── First analysis kept when reviewer returns fallback (confidence=0.0) ───

    def test_first_analysis_kept_when_reviewer_returns_fallback(self) -> None:
        reviewer = FakeReviewer(
            should_review_result=True,
            review_result=_FALLBACK_ANALYSIS,  # confidence=0.0
        )
        agent, *_ = _make_agent_with_reviewer(reviewer)
        result = agent.process_ticket(_make_ticket())

        # Result should come from the first analysis, not the fallback.
        assert result.topic      == _FIXED_ANALYSIS.topic
        assert result.urgency    == _FIXED_ANALYSIS.urgency
        assert result.confidence == _FIXED_ANALYSIS.confidence

    # ── reviewer_used True when reviewer called ───────────────────────────────

    def test_reviewer_used_true_when_reviewer_called(self) -> None:
        reviewer = FakeReviewer(should_review_result=True)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_used"] is True

    def test_reviewer_used_false_when_reviewer_not_called(self) -> None:
        reviewer = FakeReviewer(should_review_result=False)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_used"] is False

    # ── reviewer_model reflects reviewer's model_name ─────────────────────────

    def test_reviewer_model_in_timing_when_reviewer_called(self) -> None:
        reviewer = FakeReviewer(should_review_result=True)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_model"] == "fake-reviewer-model"

    def test_reviewer_model_empty_when_reviewer_not_called(self) -> None:
        reviewer = FakeReviewer(should_review_result=False)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_model"] == ""

    # ── reviewer_changed_topic and reviewer_changed_urgency ───────────────────

    def test_reviewer_changed_topic_true_when_topic_differs(self) -> None:
        # _FIXED_ANALYSIS.topic = TECHNICAL; _REVIEWED_ANALYSIS.topic = BILLING
        reviewer = FakeReviewer(
            should_review_result=True,
            review_result=_REVIEWED_ANALYSIS,
        )
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_changed_topic"] is True

    def test_reviewer_changed_urgency_true_when_urgency_differs(self) -> None:
        # _FIXED_ANALYSIS.urgency = HIGH; _REVIEWED_ANALYSIS.urgency = MEDIUM
        reviewer = FakeReviewer(
            should_review_result=True,
            review_result=_REVIEWED_ANALYSIS,
        )
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_changed_urgency"] is True

    def test_reviewer_changed_fields_false_when_reviewer_fallback(self) -> None:
        # When reviewer falls back (confidence=0.0), no change should be recorded.
        reviewer = FakeReviewer(
            should_review_result=True,
            review_result=_FALLBACK_ANALYSIS,
        )
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_changed_topic"]   is False
        assert timing["reviewer_changed_urgency"] is False

    # ── reviewer_seconds present and >= 0 ─────────────────────────────────────

    def test_reviewer_seconds_present_in_timing(self) -> None:
        reviewer = FakeReviewer(should_review_result=True)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "reviewer_seconds" in timing
        assert timing["reviewer_seconds"] >= 0.0

    def test_reviewer_seconds_zero_when_not_invoked(self) -> None:
        reviewer = FakeReviewer(should_review_result=False)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_seconds"] == 0.0

    def test_first_topic_matches_first_analysis_topic(self) -> None:
        reviewer = FakeReviewer(should_review_result=True, review_result=_REVIEWED_ANALYSIS)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        # first_topic captures the FIRST analysis, not the reviewer's output.
        assert timing["first_topic"] == _FIXED_ANALYSIS.topic.value

    def test_first_urgency_matches_first_analysis_urgency(self) -> None:
        reviewer = FakeReviewer(should_review_result=True, review_result=_REVIEWED_ANALYSIS)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["first_urgency"] == _FIXED_ANALYSIS.urgency.value

    def test_first_confidence_matches_first_analysis_confidence(self) -> None:
        reviewer = FakeReviewer(should_review_result=True, review_result=_REVIEWED_ANALYSIS)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["first_confidence"] == _FIXED_ANALYSIS.confidence

    def test_reviewer_trigger_flags_not_empty_when_reviewer_called(self) -> None:
        """reviewer_trigger_flags must contain a non-empty JSON list when reviewer runs."""
        import json
        reviewer = FakeReviewer(should_review_result=True, review_result=_REVIEWED_ANALYSIS)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        flags = json.loads(timing["reviewer_trigger_flags"])
        assert isinstance(flags, list)
        # FakeReviewer returns ["low_llm_confidence"] when should_review_result=True.
        assert len(flags) > 0

    def test_reviewer_trigger_flags_empty_json_when_reviewer_not_called(self) -> None:
        reviewer = FakeReviewer(should_review_result=False)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_trigger_flags"] == "[]"

    # ── first_short_note and reviewer_note ────────────────────────────────────

    def test_first_short_note_matches_first_analysis_note(self) -> None:
        reviewer = FakeReviewer(should_review_result=True, review_result=_REVIEWED_ANALYSIS)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["first_short_note"] == _FIXED_ANALYSIS.short_note

    def test_reviewer_note_set_when_reviewer_used_and_valid(self) -> None:
        reviewer = FakeReviewer(should_review_result=True, review_result=_REVIEWED_ANALYSIS)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_note"] == _REVIEWED_ANALYSIS.short_note

    def test_reviewer_note_empty_when_reviewer_not_used(self) -> None:
        reviewer = FakeReviewer(should_review_result=False)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["reviewer_note"] == ""

    def test_reviewer_note_empty_when_reviewer_returns_fallback(self) -> None:
        reviewer = FakeReviewer(should_review_result=True, review_result=_FALLBACK_ANALYSIS)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        # fallback has confidence=0.0, so reviewer output is discarded
        assert timing["reviewer_note"] == ""

    # ── validator_flags and validator_notes ──────────────────────────────────

    def test_validator_flags_in_timing_as_json_string(self) -> None:
        import json
        reviewer = FakeReviewer(should_review_result=False)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        flags = json.loads(timing["validator_flags"])
        assert isinstance(flags, list)

    def test_validator_notes_in_timing_as_json_string(self) -> None:
        import json
        reviewer = FakeReviewer(should_review_result=False)
        agent, *_ = _make_agent_with_reviewer(reviewer)
        _, timing = agent.process_ticket_timed(_make_ticket())
        notes = json.loads(timing["validator_notes"])
        assert isinstance(notes, list)

    def test_validator_flags_contains_first_validation_flags(self) -> None:
        """validator_flags must reflect the first validation, not the reviewer's."""
        import json
        reviewer = FakeReviewer(should_review_result=False)
        agent, *_ = _make_agent_with_reviewer(reviewer, validation_result=_VALIDATION_WITH_TRIGGER_FLAGS)
        _, timing = agent.process_ticket_timed(_make_ticket())
        flags = json.loads(timing["validator_flags"])
        assert "low_llm_confidence" in flags
        assert "topic_disagreement" in flags

    def test_reviewer_trigger_flags_is_subset_of_validator_flags(self) -> None:
        """reviewer_trigger_flags must be a subset of validator_flags."""
        import json
        reviewer = FakeReviewer(should_review_result=True, review_result=_REVIEWED_ANALYSIS)
        agent, *_ = _make_agent_with_reviewer(reviewer, validation_result=_VALIDATION_WITH_TRIGGER_FLAGS)
        _, timing = agent.process_ticket_timed(_make_ticket())
        all_flags     = set(json.loads(timing["validator_flags"]))
        trigger_flags = set(json.loads(timing["reviewer_trigger_flags"]))
        assert trigger_flags.issubset(all_flags)

    # ── neighbor evidence fields ──────────────────────────────────────────────

    def test_neighbor_predicted_topic_in_timing(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "neighbor_predicted_topic" in timing

    def test_neighbor_topic_confidence_in_timing(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "neighbor_topic_confidence" in timing

    def test_neighbor_predicted_priority_in_timing(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "neighbor_predicted_priority" in timing

    def test_neighbor_priority_confidence_in_timing(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "neighbor_priority_confidence" in timing

    def test_neighbor_predicted_topic_matches_neighbor_prediction(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["neighbor_predicted_topic"] == _FIXED_NEIGHBOR_PREDICTION.predicted_proxy_topic

    def test_neighbor_predicted_priority_matches_neighbor_prediction(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["neighbor_predicted_priority"] == _FIXED_NEIGHBOR_PREDICTION.predicted_priority

    def test_neighbor_topic_confidence_matches_neighbor_prediction(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert abs(timing["neighbor_topic_confidence"] - _FIXED_NEIGHBOR_PREDICTION.proxy_topic_confidence) < 0.0001

    def test_neighbor_priority_confidence_matches_neighbor_prediction(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert abs(timing["neighbor_priority_confidence"] - _FIXED_NEIGHBOR_PREDICTION.priority_confidence) < 0.0001


# ── New timing fields without reviewer ────────────────────────────────────────

class TestNewTimingFieldsNoReviewer:
    """Verify new explainability fields are present even without a reviewer."""

    def test_first_short_note_present(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "first_short_note" in timing

    def test_reviewer_note_present_and_empty(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "reviewer_note" in timing
        assert timing["reviewer_note"] == ""

    def test_validator_flags_present(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "validator_flags" in timing

    def test_validator_notes_present(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert "validator_notes" in timing

    def test_first_short_note_equals_analysis_note(self) -> None:
        agent, *_ = _make_agent()
        _, timing = agent.process_ticket_timed(_make_ticket())
        assert timing["first_short_note"] == _FIXED_ANALYSIS.short_note
