"""
Sprint 5D — ActionExecutor / Simulated Tools.

ActionExecutor dispatches a NextAction to a named simulated tool method.
It does not call external APIs, send real emails, or write to real systems.
It returns an ActionExecutionResult describing what would happen in production.

Each private method corresponds to one simulated downstream tool.
The dispatch table maps NextAction members to those methods.

Interview explanation:
  In a production system each method would be replaced by a real API call
  (e.g. POST /claims, PUT /tickets/escalate).  In this prototype the methods
  return a structured record describing the intended action so the full
  agentic pipeline can be demonstrated and audited locally.
"""

from src.domain.enums import NextAction
from src.domain.models import ActionExecutionResult


class ActionExecutor:
    """
    Simulates downstream tool execution for each supported NextAction.

    Stateless — no configuration is needed because all targets and notes
    are fixed strings defined per action.
    """

    def __init__(self) -> None:
        # Build the dispatch table once at construction time.
        self._dispatch = {
            NextAction.SEND_FAQ:          self._send_faq_or_self_service_link,
            NextAction.CREATE_CLAIM:      self._create_or_update_claim,
            NextAction.FORWARD_BILLING:   self._forward_to_billing_team,
            NextAction.FORWARD_TECHNICAL: self._forward_to_technical_support,
            NextAction.ESCALATE:          self._escalate_to_human_supervisor,
            NextAction.ASK_MORE_INFO:     self._ask_for_more_information,
        }

    def execute(
        self,
        next_action: NextAction,
        ticket_id: str,
        short_note: str | None = None,
    ) -> ActionExecutionResult:
        """
        Dispatch next_action to the matching simulated tool method.

        Parameters
        ----------
        next_action:
            The routing decision produced by TriageRouter.
        ticket_id:
            The ticket being processed (included in the result for tracing).
        short_note:
            Optional context from the LLM analysis to carry into the note.

        Returns
        -------
        ActionExecutionResult
            A structured record of what the simulated action did.
        """
        tool = self._dispatch.get(next_action)

        if tool is None:
            # Defensive fallback — should not be reached with valid NextAction values.
            return ActionExecutionResult(
                selected_action=NextAction.ESCALATE,
                action_status="simulated_fallback",
                action_note="Unsupported action received; simulated escalation to human supervisor.",
                target="human_supervisor_queue",
            )

        return tool(ticket_id, short_note)

    # ── Simulated tool methods ─────────────────────────────────────────────────

    def _send_faq_or_self_service_link(
        self,
        ticket_id: str,
        short_note: str | None,
    ) -> ActionExecutionResult:
        return ActionExecutionResult(
            selected_action=NextAction.SEND_FAQ,
            action_status="simulated_success",
            action_note="Simulated action: send standard FAQ or self-service link to the customer.",
            target="self_service_faq",
        )

    def _create_or_update_claim(
        self,
        ticket_id: str,
        short_note: str | None,
    ) -> ActionExecutionResult:
        return ActionExecutionResult(
            selected_action=NextAction.CREATE_CLAIM,
            action_status="simulated_success",
            action_note="Simulated action: create or update a claim record for this ticket.",
            target="claims_queue",
        )

    def _forward_to_billing_team(
        self,
        ticket_id: str,
        short_note: str | None,
    ) -> ActionExecutionResult:
        return ActionExecutionResult(
            selected_action=NextAction.FORWARD_BILLING,
            action_status="simulated_success",
            action_note="Simulated action: forward ticket to the billing team queue.",
            target="billing_team_queue",
        )

    def _forward_to_technical_support(
        self,
        ticket_id: str,
        short_note: str | None,
    ) -> ActionExecutionResult:
        return ActionExecutionResult(
            selected_action=NextAction.FORWARD_TECHNICAL,
            action_status="simulated_success",
            action_note="Simulated action: forward ticket to the technical support queue.",
            target="technical_support_queue",
        )

    def _escalate_to_human_supervisor(
        self,
        ticket_id: str,
        short_note: str | None,
    ) -> ActionExecutionResult:
        return ActionExecutionResult(
            selected_action=NextAction.ESCALATE,
            action_status="simulated_success",
            action_note="Simulated action: escalate ticket to a human supervisor.",
            target="human_supervisor_queue",
        )

    def _ask_for_more_information(
        self,
        ticket_id: str,
        short_note: str | None,
    ) -> ActionExecutionResult:
        return ActionExecutionResult(
            selected_action=NextAction.ASK_MORE_INFO,
            action_status="simulated_success",
            action_note="Simulated action: ask customer for additional information before routing.",
            target="customer",
        )
