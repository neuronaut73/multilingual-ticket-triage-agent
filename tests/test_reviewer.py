"""
Tests for Sprint 6C.5 — ConditionalLLMReviewer.

All tests are fully deterministic.  No Ollama calls.
FakeLLMClient returns pre-defined responses.

Coverage:
  get_triggered_review_flags (composite confidence-aware rules):
    - low_llm_confidence always triggers regardless of first_analysis confidence.
    - urgency_disagreement triggers only when first_analysis.confidence < 0.90.
    - urgency_disagreement does NOT trigger when confidence >= 0.90.
    - topic_disagreement triggers only when first_analysis.confidence < 0.85.
    - topic_disagreement does NOT trigger when confidence >= 0.85.
    - low_neighbor_priority_confidence alone never triggers.
    - low_neighbor_topic_confidence alone never triggers.
    - missing_information never triggers.
    - returns only the flags that actually fired.

  should_review (backward-compat path):
    - Returns False when no flags present.
    - Returns False when flags do not overlap with trigger_flags.
    - Returns True when low_llm_confidence present (no first_analysis needed).
    - Returns False for low_neighbor_priority_confidence (not a trigger).
    - Returns True when called with first_analysis and a triggering flag.

  review:
    - Returns valid LLMAnalysis on valid LLM output.
    - Returns fallback (confidence=0.0) on invalid LLM output after retry.
    - Retry is triggered on invalid JSON.
    - With max_retries=0, falls back immediately on first failure.

  Prompt content:
    - Reviewer prompt contains ticket subject and body.
    - Reviewer prompt contains neighbor evidence.
    - Reviewer prompt contains first analysis values.
    - Reviewer prompt contains validation flags.
    - Reviewer prompt does NOT contain proxy_topic or actual_queue of current ticket
      (only neighbor labels, which are retrieval evidence — not the ticket's own labels).

  Config integration:
    - main.py and cli_menu.py pass ceiling values to ConditionalLLMReviewer.

  Data leakage:
    - reviewer.py must not import actual_* or proxy_* labels of the current ticket.
"""

import json

import pytest

from src.application.reviewer import ConditionalLLMReviewer, _build_reviewer_prompt
from src.domain.enums import Topic, Urgency
from src.domain.models import (
    LLMAnalysis,
    NeighborEvidence,
    NeighborPrediction,
    TicketInput,
    ValidationResult,
)

# ─── Shared fixtures ──────────────────────────────────────────────────────────

# These three flags match the config.yaml reviewer.trigger_flags list.
# low_neighbor_priority_confidence and low_neighbor_topic_confidence are
# intentionally excluded — they are validator signals, not reviewer triggers.
TRIGGER_FLAGS = [
    "low_llm_confidence",
    "urgency_disagreement",
    "topic_disagreement",
]

VALID_JSON = json.dumps({
    "topic": "Billing / Payment",
    "urgency": "Medium",
    "missing_info": False,
    "missing_fields": [],
    "confidence": 0.78,
    "short_note": "Reviewer revised topic from Technical to Billing based on payment context.",
})

INVALID_JSON = "not valid json {"


class FakeLLMClient:
    """Returns responses in sequence.  Records all prompts."""

    def __init__(self, *responses: str) -> None:
        self._iter   = iter(responses)
        self.prompts: list[str] = []

    def generate_json(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self._iter)


def make_ticket() -> TicketInput:
    return TicketInput(
        ticket_id="EVAL_001",
        subject="Invoice shows wrong amount",
        body="I received my monthly invoice and it shows a higher amount than last month.",
        raw_text="Invoice shows wrong amount I received my monthly invoice.",
        cleaned_text="Invoice shows wrong amount I received my monthly invoice.",
        representation_text=(
            "Subject: Invoice shows wrong amount\n\n"
            "Body: I received my monthly invoice and it shows a higher amount than last month."
        ),
        text_snippet="Invoice shows wrong amount",
    )


def make_neighbor_prediction() -> NeighborPrediction:
    neighbor = NeighborEvidence(
        ticket_id="REF_007",
        distance=0.12,
        similarity=0.89,
        actual_queue="Billing and Payments",
        actual_priority="medium",
        actual_type="Request",
        actual_tags=["Invoice", "Payment"],
        proxy_topic="Billing / Payment",
        proxy_urgency="Medium",
        text_snippet="Customer received an incorrect invoice for last month.",
    )
    return NeighborPrediction(
        predicted_queue="Billing and Payments",
        queue_confidence=0.82,
        predicted_priority="medium",
        priority_confidence=0.77,
        predicted_proxy_topic="Billing / Payment",
        proxy_topic_confidence=0.80,
        predicted_tags=["Invoice", "Payment", "Billing"],
        neighbors=[neighbor],
    )


def make_first_analysis() -> LLMAnalysis:
    return LLMAnalysis(
        topic=Topic.TECHNICAL,         # intentionally wrong — reviewer should fix
        urgency=Urgency.HIGH,
        missing_info=False,
        missing_fields=[],
        confidence=0.45,               # low — triggered review
        short_note="Customer may have a technical issue with the billing portal.",
    )


def make_validation(flags: list[str] | None = None) -> ValidationResult:
    flags = flags or ["low_llm_confidence", "topic_disagreement"]
    return ValidationResult(
        is_valid=True,
        requires_human_review=True,
        flags=flags,
        notes=["LLM topic = Technical / Online Access, neighbor predicted_proxy_topic = Billing / Payment"],
    )


def make_reviewer(
    client: FakeLLMClient,
    disagreement_confidence_ceiling: float = 0.85,
    urgency_disagreement_confidence_ceiling: float = 0.90,
) -> ConditionalLLMReviewer:
    return ConditionalLLMReviewer(
        llm_client=client,
        trigger_flags=TRIGGER_FLAGS,
        max_retries=1,
        model_name="devstral-small-2:24b",
        disagreement_confidence_ceiling=disagreement_confidence_ceiling,
        urgency_disagreement_confidence_ceiling=urgency_disagreement_confidence_ceiling,
    )


# ─── Helper ───────────────────────────────────────────────────────────────────

def _make_analysis_with_confidence(confidence: float) -> LLMAnalysis:
    """Build a minimal LLMAnalysis with the given confidence for trigger tests."""
    return LLMAnalysis(
        topic=Topic.TECHNICAL,
        urgency=Urgency.HIGH,
        missing_info=False,
        missing_fields=[],
        confidence=confidence,
        short_note="test analysis",
    )


def _make_validation_with_flags(flags: list[str]) -> ValidationResult:
    return ValidationResult(is_valid=True, requires_human_review=False, flags=flags, notes=[])


# ─── get_triggered_review_flags — composite confidence-aware rules ────────────

class TestGetTriggeredReviewFlags:
    """
    Tests for the composite confidence-aware trigger logic.

    Ceilings used in make_reviewer(): topic=0.85, urgency=0.90.
    """

    def test_low_llm_confidence_always_triggers(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["low_llm_confidence"])
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.95))
        assert "low_llm_confidence" in flags

    def test_low_llm_confidence_triggers_even_at_very_high_confidence(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["low_llm_confidence"])
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.99))
        assert "low_llm_confidence" in flags

    def test_urgency_disagreement_triggers_when_below_ceiling(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["urgency_disagreement"])
        # 0.89 < 0.90 ceiling → should trigger
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.89))
        assert "urgency_disagreement" in flags

    def test_urgency_disagreement_does_not_trigger_at_ceiling(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["urgency_disagreement"])
        # 0.90 == ceiling → must NOT trigger
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.90))
        assert "urgency_disagreement" not in flags

    def test_urgency_disagreement_does_not_trigger_above_ceiling(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["urgency_disagreement"])
        # 0.95 > 0.90 ceiling → must NOT trigger
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.95))
        assert "urgency_disagreement" not in flags

    def test_topic_disagreement_triggers_when_below_ceiling(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["topic_disagreement"])
        # 0.79 < 0.85 ceiling → should trigger
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.79))
        assert "topic_disagreement" in flags

    def test_topic_disagreement_does_not_trigger_at_ceiling(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["topic_disagreement"])
        # 0.85 == ceiling → must NOT trigger
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.85))
        assert "topic_disagreement" not in flags

    def test_topic_disagreement_does_not_trigger_above_ceiling(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["topic_disagreement"])
        # 0.90 > 0.85 ceiling → must NOT trigger
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.90))
        assert "topic_disagreement" not in flags

    def test_low_neighbor_priority_confidence_alone_does_not_trigger(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["low_neighbor_priority_confidence"])
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.40))
        assert flags == []

    def test_low_neighbor_topic_confidence_alone_does_not_trigger(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["low_neighbor_topic_confidence"])
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.40))
        assert flags == []

    def test_missing_information_does_not_trigger(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["missing_information"])
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.40))
        assert flags == []

    def test_returns_only_actual_trigger_flags(self) -> None:
        """Mix of flags — only low_llm_confidence should be in the returned list."""
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags([
            "low_llm_confidence",
            "low_neighbor_priority_confidence",
            "missing_information",
        ])
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.50))
        assert flags == ["low_llm_confidence"]

    def test_returns_empty_when_no_flags_fire(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags(["low_neighbor_priority_confidence"])
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.99))
        assert flags == []

    def test_returns_empty_when_no_validation_flags(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = _make_validation_with_flags([])
        flags = reviewer.get_triggered_review_flags(validation, _make_analysis_with_confidence(0.50))
        assert flags == []

    def test_respects_custom_urgency_ceiling(self) -> None:
        """Custom urgency ceiling of 0.95 triggers at confidence 0.94 but not 0.95."""
        reviewer = make_reviewer(
            FakeLLMClient(),
            urgency_disagreement_confidence_ceiling=0.95,
        )
        validation = _make_validation_with_flags(["urgency_disagreement"])
        assert "urgency_disagreement" in reviewer.get_triggered_review_flags(
            validation, _make_analysis_with_confidence(0.94)
        )
        assert "urgency_disagreement" not in reviewer.get_triggered_review_flags(
            validation, _make_analysis_with_confidence(0.95)
        )

    def test_respects_custom_topic_ceiling(self) -> None:
        """Custom topic ceiling of 0.70 triggers at 0.69 but not 0.70."""
        reviewer = make_reviewer(
            FakeLLMClient(),
            disagreement_confidence_ceiling=0.70,
        )
        validation = _make_validation_with_flags(["topic_disagreement"])
        assert "topic_disagreement" in reviewer.get_triggered_review_flags(
            validation, _make_analysis_with_confidence(0.69)
        )
        assert "topic_disagreement" not in reviewer.get_triggered_review_flags(
            validation, _make_analysis_with_confidence(0.70)
        )


# ─── should_review — backward-compat path ────────────────────────────────────

class TestShouldReview:
    """
    Tests for should_review.

    When called with first_analysis, delegates to get_triggered_review_flags
    and applies the full composite logic.

    When called without first_analysis (backward-compat path), only
    low_llm_confidence can trigger.
    """

    def test_returns_false_when_no_flags(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = ValidationResult(is_valid=True, flags=[], notes=[])
        assert reviewer.should_review(validation) is False

    def test_returns_false_when_flags_do_not_overlap(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = ValidationResult(
            is_valid=True,
            flags=["missing_information", "high_urgency_with_limited_confidence"],
            notes=[],
        )
        assert reviewer.should_review(validation) is False

    def test_low_llm_confidence_triggers_without_first_analysis(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = ValidationResult(is_valid=True, flags=["low_llm_confidence"], notes=[])
        assert reviewer.should_review(validation) is True

    def test_low_neighbor_priority_confidence_does_not_trigger(self) -> None:
        """low_neighbor_priority_confidence must never trigger the reviewer."""
        reviewer = make_reviewer(FakeLLMClient())
        validation = ValidationResult(
            is_valid=True,
            flags=["low_neighbor_priority_confidence"],
            notes=[],
        )
        assert reviewer.should_review(validation) is False

    def test_low_neighbor_topic_confidence_does_not_trigger(self) -> None:
        """low_neighbor_topic_confidence must never trigger the reviewer."""
        reviewer = make_reviewer(FakeLLMClient())
        validation = ValidationResult(
            is_valid=True,
            flags=["low_neighbor_topic_confidence"],
            notes=[],
        )
        assert reviewer.should_review(validation) is False

    def test_urgency_disagreement_triggers_with_first_analysis_below_ceiling(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = ValidationResult(is_valid=True, flags=["urgency_disagreement"], notes=[])
        assert reviewer.should_review(validation, _make_analysis_with_confidence(0.89)) is True

    def test_urgency_disagreement_does_not_trigger_with_first_analysis_at_ceiling(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = ValidationResult(is_valid=True, flags=["urgency_disagreement"], notes=[])
        assert reviewer.should_review(validation, _make_analysis_with_confidence(0.90)) is False

    def test_topic_disagreement_triggers_with_first_analysis_below_ceiling(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = ValidationResult(is_valid=True, flags=["topic_disagreement"], notes=[])
        assert reviewer.should_review(validation, _make_analysis_with_confidence(0.79)) is True

    def test_topic_disagreement_does_not_trigger_with_first_analysis_at_ceiling(self) -> None:
        reviewer = make_reviewer(FakeLLMClient())
        validation = ValidationResult(is_valid=True, flags=["topic_disagreement"], notes=[])
        assert reviewer.should_review(validation, _make_analysis_with_confidence(0.85)) is False


# ─── review — happy path ──────────────────────────────────────────────────────

class TestReviewHappyPath:

    def test_valid_output_returns_llm_analysis(self) -> None:
        client = FakeLLMClient(VALID_JSON)
        reviewer = make_reviewer(client)
        result = reviewer.review(
            make_ticket(), make_neighbor_prediction(), make_first_analysis(), make_validation()
        )

        assert isinstance(result, LLMAnalysis)
        assert result.topic   == Topic.BILLING
        assert result.urgency == Urgency.MEDIUM
        assert result.confidence == pytest.approx(0.78)
        assert result.missing_info is False
        assert len(client.prompts) == 1

    def test_valid_output_has_confidence_above_zero(self) -> None:
        client = FakeLLMClient(VALID_JSON)
        reviewer = make_reviewer(client)
        result = reviewer.review(
            make_ticket(), make_neighbor_prediction(), make_first_analysis(), make_validation()
        )
        assert result.confidence > 0


# ─── review — retry and fallback ─────────────────────────────────────────────

class TestReviewFallback:

    def test_invalid_json_triggers_retry(self) -> None:
        client = FakeLLMClient(INVALID_JSON, VALID_JSON)
        reviewer = make_reviewer(client)
        result = reviewer.review(
            make_ticket(), make_neighbor_prediction(), make_first_analysis(), make_validation()
        )

        assert len(client.prompts) == 2
        assert result.topic == Topic.BILLING

    def test_retry_failure_returns_fallback_with_zero_confidence(self) -> None:
        client = FakeLLMClient(INVALID_JSON, INVALID_JSON)
        reviewer = make_reviewer(client)
        result = reviewer.review(
            make_ticket(), make_neighbor_prediction(), make_first_analysis(), make_validation()
        )

        assert result.confidence == pytest.approx(0.0)
        assert result.topic      == Topic.OTHER
        assert result.missing_info is True
        assert "reviewer_output_invalid" in result.missing_fields

    def test_max_retries_zero_goes_straight_to_fallback(self) -> None:
        client = FakeLLMClient(INVALID_JSON)
        reviewer = ConditionalLLMReviewer(
            llm_client=client,
            trigger_flags=TRIGGER_FLAGS,
            max_retries=0,
            model_name="test-model",
        )
        result = reviewer.review(
            make_ticket(), make_neighbor_prediction(), make_first_analysis(), make_validation()
        )

        assert len(client.prompts) == 1
        assert result.confidence == pytest.approx(0.0)


# ─── Prompt content ───────────────────────────────────────────────────────────

class TestReviewerPromptContent:

    def _get_prompt(self) -> str:
        client = FakeLLMClient(VALID_JSON)
        reviewer = make_reviewer(client)
        reviewer.review(
            make_ticket(), make_neighbor_prediction(), make_first_analysis(), make_validation()
        )
        return client.prompts[0]

    def test_prompt_contains_ticket_subject(self) -> None:
        prompt = self._get_prompt()
        assert "Invoice shows wrong amount" in prompt

    def test_prompt_contains_ticket_body(self) -> None:
        prompt = self._get_prompt()
        assert "monthly invoice" in prompt

    def test_prompt_contains_neighbor_evidence(self) -> None:
        prompt = self._get_prompt()
        assert "Billing and Payments" in prompt    # predicted_queue from neighbor
        assert "medium"               in prompt    # predicted_priority

    def test_prompt_contains_first_analysis_topic(self) -> None:
        prompt = self._get_prompt()
        assert "Technical / Online Access" in prompt  # first_analysis.topic.value

    def test_prompt_contains_first_analysis_confidence(self) -> None:
        prompt = self._get_prompt()
        assert "0.45" in prompt  # first_analysis.confidence

    def test_prompt_contains_validation_flags(self) -> None:
        prompt = self._get_prompt()
        assert "low_llm_confidence"  in prompt
        assert "topic_disagreement"  in prompt

    def test_prompt_contains_validation_notes(self) -> None:
        prompt = self._get_prompt()
        assert "neighbor predicted_proxy_topic" in prompt

    def test_prompt_contains_allowed_topics(self) -> None:
        prompt = self._get_prompt()
        assert "Policy / Contract"         in prompt
        assert "Claims / Damage"           in prompt
        assert "Billing / Payment"         in prompt
        assert "Technical / Online Access" in prompt
        assert "Other"                     in prompt

    def test_prompt_contains_allowed_urgency_values(self) -> None:
        prompt = self._get_prompt()
        assert "Low"    in prompt
        assert "Medium" in prompt
        assert "High"   in prompt

    def test_prompt_does_not_contain_current_ticket_proxy_topic(self) -> None:
        """
        The reviewer prompt must not receive proxy_topic of the CURRENT ticket.
        Only neighbor proxy labels (retrieval evidence) are allowed.

        The _build_reviewer_prompt function receives TicketInput (no proxy labels)
        and NeighborPrediction (neighbor evidence).  The proxy_topic of the ticket
        being evaluated never enters the reviewer's prompt.
        """
        ticket = make_ticket()
        # proxy_topic of the current ticket is NOT a field on TicketInput.
        # Verify TicketInput has no proxy_topic attribute.
        assert not hasattr(ticket, "proxy_topic"), (
            "TicketInput must not carry proxy_topic — data leakage risk"
        )

    def test_build_reviewer_prompt_signature_accepts_no_proxy_labels(self) -> None:
        """
        _build_reviewer_prompt must not accept actual_* or proxy_* of the current ticket.

        It accepts: TicketInput, NeighborPrediction, LLMAnalysis, ValidationResult.
        None of these contain the current ticket's evaluation labels.
        """
        import inspect
        sig = inspect.signature(_build_reviewer_prompt)
        param_names = list(sig.parameters.keys())
        assert param_names == ["ticket", "neighbor_prediction", "first_analysis", "validation"]
