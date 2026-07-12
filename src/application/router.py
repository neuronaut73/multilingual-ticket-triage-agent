"""
Sprint 5C — Deterministic Router.

TriageRouter selects the final next_action from the LLM analysis and the
validation result.  It is fully deterministic, calls no LLM, and executes
no actions — it only returns one NextAction value.

Routing rules (applied in strict priority order):

  Rule 1  missing information    → ask_for_more_information
  Rule 2  human review required  → escalate_to_human_supervisor
  Rule 3  high urgency           → escalate_to_human_supervisor
  Rule 4  topic-based default    → topic-specific action
  Rule 5  safe fallback          → escalate_to_human_supervisor
"""

from src.domain.enums import NextAction, Topic, Urgency
from src.domain.models import LLMAnalysis, ValidationResult

# Maps each topic to its default action when no escalation condition applies.
_TOPIC_DEFAULT_ACTION: dict[Topic, NextAction] = {
    Topic.CLAIMS:    NextAction.CREATE_CLAIM,
    Topic.BILLING:   NextAction.FORWARD_BILLING,
    Topic.TECHNICAL: NextAction.FORWARD_TECHNICAL,
    Topic.POLICY:    NextAction.SEND_FAQ,
    Topic.OTHER:     NextAction.SEND_FAQ,
}


class TriageRouter:
    """
    Selects the next_action for a triaged ticket.

    Stateless — __init__ takes no parameters because the routing logic is
    expressed entirely as ordered rules, not configurable thresholds.
    """

    def __init__(self) -> None:
        pass

    def route(
        self,
        analysis: LLMAnalysis,
        validation: ValidationResult,
    ) -> NextAction:
        """
        Apply routing rules in priority order and return exactly one NextAction.

        Parameters
        ----------
        analysis:
            Structured LLM output (topic, urgency, missing_info, confidence).
        validation:
            Deterministic validation result (requires_human_review, flags).

        Returns
        -------
        NextAction
            The single selected action for this ticket.
        """
        # Rule 1 — missing information takes highest priority.
        # The customer must provide more details before any routing can proceed.
        if analysis.missing_info:
            return NextAction.ASK_MORE_INFO

        # Rule 2 — human review required.
        # Triggered by low confidence, urgency disagreements, or other flags.
        if validation.requires_human_review:
            return NextAction.ESCALATE

        # Rule 3 — high urgency.
        # Even when validation passes, high-urgency tickets are escalated.
        if analysis.urgency == Urgency.HIGH:
            return NextAction.ESCALATE

        # Rule 4 — topic-based default action.
        topic_action = _TOPIC_DEFAULT_ACTION.get(analysis.topic)
        if topic_action is not None:
            return topic_action

        # Rule 5 — safe fallback.
        # Should not be reached with valid Topic values, but included for safety.
        return NextAction.ESCALATE
