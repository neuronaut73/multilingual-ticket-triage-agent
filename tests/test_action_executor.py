"""
Tests for Sprint 5D — ActionExecutor.

All tests are fully deterministic.  No LLM, embeddings, or LanceDB calls.

Test coverage:
  - Each supported NextAction routes to the correct target.
  - Every supported action returns action_status = "simulated_success".
  - Every result preserves selected_action matching the input.
  - The defensive fallback returns action_status = "simulated_fallback".
"""

import pytest

from src.application.action_executor import ActionExecutor
from src.domain.enums import NextAction
from src.domain.models import ActionExecutionResult


TICKET_ID = "test-ticket-001"
SHORT_NOTE = "Test note"


class TestActionExecutor:
    def setup_method(self) -> None:
        self.executor = ActionExecutor()

    # ── Target assertions ──────────────────────────────────────────────────────

    def test_send_faq_returns_self_service_faq_target(self) -> None:
        result = self.executor.execute(NextAction.SEND_FAQ, TICKET_ID, SHORT_NOTE)
        assert result.target == "self_service_faq"

    def test_create_claim_returns_claims_queue_target(self) -> None:
        result = self.executor.execute(NextAction.CREATE_CLAIM, TICKET_ID, SHORT_NOTE)
        assert result.target == "claims_queue"

    def test_forward_billing_returns_billing_team_queue_target(self) -> None:
        result = self.executor.execute(NextAction.FORWARD_BILLING, TICKET_ID, SHORT_NOTE)
        assert result.target == "billing_team_queue"

    def test_forward_technical_returns_technical_support_queue_target(self) -> None:
        result = self.executor.execute(NextAction.FORWARD_TECHNICAL, TICKET_ID, SHORT_NOTE)
        assert result.target == "technical_support_queue"

    def test_escalate_returns_human_supervisor_queue_target(self) -> None:
        result = self.executor.execute(NextAction.ESCALATE, TICKET_ID, SHORT_NOTE)
        assert result.target == "human_supervisor_queue"

    def test_ask_more_info_returns_customer_target(self) -> None:
        result = self.executor.execute(NextAction.ASK_MORE_INFO, TICKET_ID, SHORT_NOTE)
        assert result.target == "customer"

    # ── Status assertions ──────────────────────────────────────────────────────

    def test_every_supported_action_returns_simulated_success(self) -> None:
        supported = [
            NextAction.SEND_FAQ,
            NextAction.CREATE_CLAIM,
            NextAction.FORWARD_BILLING,
            NextAction.FORWARD_TECHNICAL,
            NextAction.ESCALATE,
            NextAction.ASK_MORE_INFO,
        ]
        for action in supported:
            result = self.executor.execute(action, TICKET_ID)
            assert result.action_status == "simulated_success", (
                f"Expected simulated_success for {action}, got {result.action_status}"
            )

    # ── selected_action preservation ──────────────────────────────────────────

    def test_every_result_preserves_selected_action(self) -> None:
        supported = [
            NextAction.SEND_FAQ,
            NextAction.CREATE_CLAIM,
            NextAction.FORWARD_BILLING,
            NextAction.FORWARD_TECHNICAL,
            NextAction.ESCALATE,
            NextAction.ASK_MORE_INFO,
        ]
        for action in supported:
            result = self.executor.execute(action, TICKET_ID)
            assert result.selected_action == action, (
                f"selected_action mismatch for {action}: got {result.selected_action}"
            )

    # ── Return type ───────────────────────────────────────────────────────────

    def test_execute_returns_action_execution_result(self) -> None:
        result = self.executor.execute(NextAction.ESCALATE, TICKET_ID)
        assert isinstance(result, ActionExecutionResult)

    # ── short_note is optional ────────────────────────────────────────────────

    def test_execute_works_without_short_note(self) -> None:
        result = self.executor.execute(NextAction.FORWARD_BILLING, TICKET_ID)
        assert result.action_status == "simulated_success"

    # ── Defensive fallback ────────────────────────────────────────────────────

    def test_fallback_branch_returns_simulated_fallback(self) -> None:
        """
        Force the fallback branch by bypassing the dispatch table.

        The dispatch table covers all valid NextAction members so the fallback
        cannot be triggered via the public API with a valid enum.  We simulate
        the condition by temporarily clearing the dispatch table.
        """
        executor = ActionExecutor()
        executor._dispatch = {}  # remove all entries to trigger fallback
        result = executor.execute(NextAction.ESCALATE, TICKET_ID)
        assert result.action_status == "simulated_fallback"
        assert result.selected_action == NextAction.ESCALATE
        assert result.target == "human_supervisor_queue"
