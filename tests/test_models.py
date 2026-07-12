"""
Tests for domain enums and Pydantic models.

Verifies that:
- Topic enum uses HDI assignment values, not Kaggle queue names.
- LLMAnalysis rejects invalid topic strings.
- TriageResult can be constructed with valid assignment labels.
"""
import pytest
from pydantic import ValidationError

from src.domain.enums import NextAction, Topic, Urgency
from src.domain.models import (
    ActionExecutionResult,
    LLMAnalysis,
    TicketInput,
    TriageResult,
)


# ─── Topic enum ───────────────────────────────────────────────────────────────

def test_topic_enum_has_assignment_values():
    values = {t.value for t in Topic}
    assert "Policy / Contract"        in values
    assert "Claims / Damage"          in values
    assert "Billing / Payment"        in values
    assert "Technical / Online Access" in values
    assert "Other"                    in values


def test_topic_enum_has_no_kaggle_queue_names():
    values = {t.value for t in Topic}
    assert "Technical Support"            not in values
    assert "IT Support"                   not in values
    assert "Billing and Payments"         not in values
    assert "Product Support"              not in values
    assert "Service Outages and Maintenance" not in values


# ─── NextAction enum ──────────────────────────────────────────────────────────

def test_next_action_enum_has_assignment_values():
    values = {a.value for a in NextAction}
    assert "send_standard_faq_or_self_service_link" in values
    assert "create_or_update_claim"                 in values
    assert "forward_to_billing_team"                in values
    assert "forward_to_technical_support"           in values
    assert "escalate_to_human_supervisor"           in values
    assert "ask_for_more_information"               in values


# ─── LLMAnalysis ─────────────────────────────────────────────────────────────

def test_llm_analysis_rejects_invalid_topic():
    with pytest.raises(ValidationError):
        LLMAnalysis(
            topic="Technical Support",   # Kaggle queue — not a valid Topic
            urgency=Urgency.HIGH,
            missing_info=False,
            confidence=0.9,
            short_note="test",
        )


def test_llm_analysis_accepts_valid_topic():
    analysis = LLMAnalysis(
        topic=Topic.TECHNICAL,
        urgency=Urgency.HIGH,
        missing_info=False,
        confidence=0.9,
        short_note="Access issue reported.",
    )
    assert analysis.topic == Topic.TECHNICAL
    assert analysis.urgency == Urgency.HIGH


def test_llm_analysis_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        LLMAnalysis(
            topic=Topic.BILLING,
            urgency=Urgency.MEDIUM,
            missing_info=False,
            confidence=1.5,    # > 1.0 — invalid
            short_note="test",
        )


# ─── TriageResult ─────────────────────────────────────────────────────────────

def test_triage_result_with_valid_assignment_labels():
    result = TriageResult(
        ticket_id="T001",
        text_snippet="Cannot log in to the customer portal.",
        topic=Topic.TECHNICAL,
        urgency=Urgency.HIGH,
        next_action=NextAction.FORWARD_TECHNICAL,
        confidence=0.85,
        missing_info=False,
        missing_fields=[],
        requires_human_review=False,
        short_note="Login issue on portal.",
        action_result=None,
    )
    assert result.topic.value          == "Technical / Online Access"
    assert result.urgency.value        == "High"
    assert result.next_action.value    == "forward_to_technical_support"
    assert result.action_result is None


def test_triage_result_with_action_execution_result():
    action = ActionExecutionResult(
        selected_action=NextAction.ESCALATE,
        action_status="dispatched",
        action_note="Routed to supervisor queue.",
        target="supervisor-queue",
    )
    result = TriageResult(
        ticket_id="T002",
        text_snippet="Urgent claim for fire damage.",
        topic=Topic.CLAIMS,
        urgency=Urgency.HIGH,
        next_action=NextAction.ESCALATE,
        confidence=0.92,
        missing_info=False,
        missing_fields=[],
        requires_human_review=True,
        short_note="High-urgency claim escalated.",
        action_result=action,
    )
    assert result.action_result.selected_action == NextAction.ESCALATE
    assert result.requires_human_review is True
