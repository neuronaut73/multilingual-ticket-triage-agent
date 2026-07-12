"""
Tests for Sprint 5C — TriageRouter.

All tests are fully deterministic.  No LLM, embeddings, or LanceDB calls.

Test coverage:
  - Rule 1: missing_info  → ASK_MORE_INFO
  - Rule 2: requires_human_review → ESCALATE
  - Rule 3: high urgency  → ESCALATE
  - Rule 4: topic defaults (Claims, Billing, Technical, Policy, Other)
  - Precedence: missing_info beats human_review
                human_review beats high urgency
                high urgency beats topic default
"""

import pytest

from src.application.router import TriageRouter
from src.domain.enums import NextAction, Topic, Urgency
from src.domain.models import LLMAnalysis, ValidationResult


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _analysis(
    topic: Topic = Topic.TECHNICAL,
    urgency: Urgency = Urgency.MEDIUM,
    missing_info: bool = False,
    confidence: float = 0.85,
) -> LLMAnalysis:
    """Build a minimal LLMAnalysis for testing."""
    return LLMAnalysis(
        topic=topic,
        urgency=urgency,
        missing_info=missing_info,
        missing_fields=[],
        confidence=confidence,
        short_note="test",
    )


def _validation(
    requires_human_review: bool = False,
    flags: list[str] | None = None,
) -> ValidationResult:
    """Build a minimal ValidationResult for testing."""
    return ValidationResult(
        is_valid=True,
        requires_human_review=requires_human_review,
        flags=flags or [],
        notes=[],
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestTriageRouter:
    def setup_method(self) -> None:
        self.router = TriageRouter()

    # ── Rule 1: missing information ───────────────────────────────────────────

    def test_missing_info_routes_to_ask_more_info(self) -> None:
        analysis   = _analysis(missing_info=True)
        validation = _validation(requires_human_review=False)
        result = self.router.route(analysis, validation)
        assert result == NextAction.ASK_MORE_INFO

    # ── Rule 2: human review ──────────────────────────────────────────────────

    def test_requires_human_review_routes_to_escalate(self) -> None:
        analysis   = _analysis(urgency=Urgency.MEDIUM, missing_info=False)
        validation = _validation(requires_human_review=True)
        result = self.router.route(analysis, validation)
        assert result == NextAction.ESCALATE

    # ── Rule 3: high urgency ──────────────────────────────────────────────────

    def test_high_urgency_routes_to_escalate(self) -> None:
        analysis   = _analysis(urgency=Urgency.HIGH, missing_info=False)
        validation = _validation(requires_human_review=False)
        result = self.router.route(analysis, validation)
        assert result == NextAction.ESCALATE

    def test_technical_topic_high_urgency_routes_to_escalate(self) -> None:
        """High urgency beats the Technical topic default action."""
        analysis   = _analysis(topic=Topic.TECHNICAL, urgency=Urgency.HIGH)
        validation = _validation(requires_human_review=False)
        result = self.router.route(analysis, validation)
        assert result == NextAction.ESCALATE

    # ── Rule 4: topic defaults ────────────────────────────────────────────────

    def test_claims_topic_routes_to_create_claim(self) -> None:
        analysis   = _analysis(topic=Topic.CLAIMS, urgency=Urgency.MEDIUM)
        validation = _validation(requires_human_review=False)
        result = self.router.route(analysis, validation)
        assert result == NextAction.CREATE_CLAIM

    def test_billing_topic_routes_to_forward_billing(self) -> None:
        analysis   = _analysis(topic=Topic.BILLING, urgency=Urgency.MEDIUM)
        validation = _validation(requires_human_review=False)
        result = self.router.route(analysis, validation)
        assert result == NextAction.FORWARD_BILLING

    def test_technical_topic_routes_to_forward_technical(self) -> None:
        analysis   = _analysis(topic=Topic.TECHNICAL, urgency=Urgency.MEDIUM)
        validation = _validation(requires_human_review=False)
        result = self.router.route(analysis, validation)
        assert result == NextAction.FORWARD_TECHNICAL

    def test_policy_topic_routes_to_send_faq(self) -> None:
        analysis   = _analysis(topic=Topic.POLICY, urgency=Urgency.LOW)
        validation = _validation(requires_human_review=False)
        result = self.router.route(analysis, validation)
        assert result == NextAction.SEND_FAQ

    def test_other_topic_routes_to_send_faq(self) -> None:
        analysis   = _analysis(topic=Topic.OTHER, urgency=Urgency.LOW)
        validation = _validation(requires_human_review=False)
        result = self.router.route(analysis, validation)
        assert result == NextAction.SEND_FAQ

    # ── Precedence ────────────────────────────────────────────────────────────

    def test_missing_info_beats_human_review(self) -> None:
        """Rule 1 fires before Rule 2 even when both conditions are true."""
        analysis   = _analysis(missing_info=True)
        validation = _validation(requires_human_review=True)
        result = self.router.route(analysis, validation)
        assert result == NextAction.ASK_MORE_INFO

    def test_human_review_beats_high_urgency(self) -> None:
        """Rule 2 fires before Rule 3 even when urgency is HIGH."""
        analysis   = _analysis(urgency=Urgency.HIGH, missing_info=False)
        validation = _validation(requires_human_review=True)
        result = self.router.route(analysis, validation)
        assert result == NextAction.ESCALATE

    def test_high_urgency_beats_topic_default(self) -> None:
        """Rule 3 fires before Rule 4: Claims topic with HIGH urgency escalates."""
        analysis   = _analysis(topic=Topic.CLAIMS, urgency=Urgency.HIGH, missing_info=False)
        validation = _validation(requires_human_review=False)
        result = self.router.route(analysis, validation)
        assert result == NextAction.ESCALATE

    # ── Return type ───────────────────────────────────────────────────────────

    def test_route_always_returns_next_action(self) -> None:
        """Verify the return type is always a NextAction enum member."""
        for topic in Topic:
            for urgency in Urgency:
                for missing in (True, False):
                    for review in (True, False):
                        analysis   = _analysis(topic=topic, urgency=urgency, missing_info=missing)
                        validation = _validation(requires_human_review=review)
                        result = self.router.route(analysis, validation)
                        assert isinstance(result, NextAction)
