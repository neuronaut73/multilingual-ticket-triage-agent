"""
Sprint 5E — End-to-End Agent Orchestration.
Sprint 6B (timing) — Per-ticket step timing via process_ticket_timed.
Sprint 6C.5 (reviewer) — Optional conditional reviewer LLM loop.

TicketTriageAgent is a thin orchestrator.  It calls five existing components
in sequence and assembles a TriageResult.  It contains no business logic of
its own — routing rules, validation checks, and action dispatch all live in
their respective modules.

The workflow (reviewer disabled or no trigger flags):

  1. neighbor_retriever.retrieve_and_predict   → NeighborPrediction
  2. analyzer.analyze                          → LLMAnalysis  (first)
  3. validator.validate                        → ValidationResult
  4. router.route                              → NextAction
  5. action_executor.execute                   → ActionExecutionResult
  6. pack into TriageResult and return

With reviewer enabled and trigger flags raised:

  1. neighbor_retriever.retrieve_and_predict   → NeighborPrediction
  2. analyzer.analyze                          → LLMAnalysis  (first)
  3. validator.validate                        → ValidationResult  (first)
  4. reviewer.review                           → LLMAnalysis  (reviewed)
     validator.validate                        → ValidationResult  (reviewed)
     use reviewed analysis if confidence > 0, else keep first
  5. router.route                              → NextAction
  6. action_executor.execute                   → ActionExecutionResult
  7. pack into TriageResult and return

All components are injected at construction time so the class is easy
to test with fakes and easy to wire with real components in main.py.

Timing:
  process_ticket_timed returns (TriageResult, timing_dict) where timing_dict
  contains per-step wall-clock seconds measured with time.perf_counter().
  process_ticket delegates to process_ticket_timed so its interface is unchanged.

Reviewer trace fields in the timing dict:
  reviewer_used            — True when reviewer.review() was called
  reviewer_model           — model name string (empty if reviewer not called)
  reviewer_changed_topic   — True when reviewer changed the topic
  reviewer_changed_urgency — True when reviewer changed the urgency
  reviewer_seconds         — wall-clock seconds for the reviewer step (0.0 if skipped)

LangGraph note:
  The node boundaries here are the same as the future LangGraph nodes
  described in PROJECT_ARCHITECTURE.md §14.1.  Adding LangGraph later means
  wrapping each component in a node function — no restructuring needed.
"""

import json
import time

from src.domain.models import (
    ActionExecutionResult,
    LLMAnalysis,
    NeighborPrediction,
    TicketInput,
    TriageResult,
    ValidationResult,
)
from src.domain.enums import NextAction


class TicketTriageAgent:
    """
    Orchestrate the end-to-end triage workflow for a single ticket.

    Parameters
    ----------
    neighbor_retriever:
        Retrieves top-k similar historical tickets and produces a weighted
        NeighborPrediction.  Interface: retrieve_and_predict(ticket_id, text).
    analyzer:
        Calls the local LLM and returns a validated LLMAnalysis.
        Interface: analyze(ticket, neighbor_prediction).
    validator:
        Runs deterministic agreement checks and returns a ValidationResult.
        Interface: validate(analysis, neighbor_prediction).
    router:
        Selects the final NextAction from the analysis and validation result.
        Interface: route(analysis, validation).
    action_executor:
        Simulates tool execution for the selected action.
        Interface: execute(next_action, ticket_id, short_note).
    reviewer:
        Optional ConditionalLLMReviewer.  When provided, it is invoked after
        the first validation if composite trigger flags fire.
        Interface: get_triggered_review_flags(validation, first_analysis) -> list[str],
                   review(ticket, neighbor_prediction, first_analysis, validation) -> LLMAnalysis.
        Defaults to None (reviewer loop disabled).
    """

    def __init__(
        self,
        neighbor_retriever,
        analyzer,
        validator,
        router,
        action_executor,
        reviewer=None,
    ) -> None:
        self.neighbor_retriever = neighbor_retriever
        self.analyzer           = analyzer
        self.validator          = validator
        self.router             = router
        self.action_executor    = action_executor
        self.reviewer           = reviewer

    def process_ticket(self, ticket: TicketInput) -> TriageResult:
        """
        Run one ticket through the full triage pipeline.

        Delegates to process_ticket_timed and returns only the TriageResult.
        Existing callers and tests are unaffected.
        """
        result, _ = self.process_ticket_timed(ticket)
        return result

    def process_ticket_timed(
        self, ticket: TicketInput
    ) -> tuple[TriageResult, dict]:
        """
        Run one ticket through the full triage pipeline and return step timings.

        Steps
        -----
        1. Retrieve similar historical tickets and aggregate into a
           weighted NeighborPrediction.
        2. Analyze the ticket with the local LLM; get a structured
           LLMAnalysis (first analysis).
        3. Validate the LLM output against the neighbor evidence; get
           a ValidationResult with flags and requires_human_review.
        4. If reviewer is configured and validation has trigger flags:
             a. Run the reviewer LLM to get a potentially revised LLMAnalysis.
             b. Re-validate the reviewed analysis.
             c. If reviewer confidence > 0, use the reviewed analysis.
             d. Otherwise keep the first analysis.
        5. Route to the final NextAction using deterministic rules.
        6. Execute the action via the simulated ActionExecutor.
        7. Pack all outputs into a TriageResult and return alongside
           a timing dict with per-step wall-clock seconds.

        Parameters
        ----------
        ticket:
            A TicketInput built from subject + body only.  No labels.

        Returns
        -------
        (TriageResult, dict)
            The triage decision and a dict with keys:
              retrieval_seconds, llm_seconds, validation_seconds,
              reviewer_seconds, routing_seconds, action_execution_seconds,
              total_ticket_seconds,
              reviewer_used, reviewer_model,
              reviewer_changed_topic, reviewer_changed_urgency,
              reviewer_trigger_flags,
              first_topic, first_urgency, first_confidence.
        """
        t_total = time.perf_counter()

        # Step 1 — neighbor retrieval and weighted voting.
        t0 = time.perf_counter()
        neighbor_prediction: NeighborPrediction = (
            self.neighbor_retriever.retrieve_and_predict(
                ticket.ticket_id,
                ticket.representation_text,
            )
        )
        retrieval_seconds = time.perf_counter() - t0

        # Step 2 — local LLM structured analysis (first pass).
        t0 = time.perf_counter()
        first_analysis: LLMAnalysis = self.analyzer.analyze(ticket, neighbor_prediction)
        llm_seconds = time.perf_counter() - t0

        # Step 3 — deterministic validation of the first analysis.
        t0 = time.perf_counter()
        first_validation: ValidationResult = self.validator.validate(
            first_analysis, neighbor_prediction
        )
        validation_seconds = time.perf_counter() - t0

        # Capture first analysis values before the reviewer can change them.
        first_topic      = first_analysis.topic.value
        first_urgency    = first_analysis.urgency.value
        first_confidence = first_analysis.confidence
        first_short_note = first_analysis.short_note

        # Capture all deterministic validator output from the first validation pass.
        # validator_flags and validator_notes represent everything the validator detected.
        # reviewer_trigger_flags (below) is a subset of validator_flags.
        validator_flags = json.dumps(first_validation.flags)
        validator_notes = json.dumps(first_validation.notes)

        # Neighbor retrieval evidence — safe to log because it comes from historical
        # reference tickets, not from the current eval ticket's evaluation labels.
        neighbor_predicted_topic    = neighbor_prediction.predicted_proxy_topic or ""
        neighbor_topic_confidence   = neighbor_prediction.proxy_topic_confidence
        neighbor_predicted_priority = neighbor_prediction.predicted_priority or ""
        neighbor_priority_confidence = neighbor_prediction.priority_confidence

        # Step 4 — optional reviewer loop.
        reviewer_seconds         = 0.0
        reviewer_used            = False
        reviewer_model           = ""
        reviewer_changed_topic   = False
        reviewer_changed_urgency = False
        reviewer_trigger_flags   = "[]"
        reviewer_note            = ""

        final_analysis   = first_analysis
        final_validation = first_validation

        if self.reviewer is not None:
            triggered = self.reviewer.get_triggered_review_flags(first_validation, first_analysis)
        else:
            triggered = []

        if triggered:
            reviewer_trigger_flags = json.dumps(triggered)

            t0 = time.perf_counter()
            reviewed_analysis: LLMAnalysis = self.reviewer.review(
                ticket, neighbor_prediction, first_analysis, first_validation
            )
            reviewer_seconds = time.perf_counter() - t0
            reviewer_used    = True
            reviewer_model   = getattr(self.reviewer, "model_name", "")

            if reviewed_analysis.confidence > 0:
                # Reviewer produced a valid, non-fallback output — re-validate and use it.
                reviewed_validation = self.validator.validate(
                    reviewed_analysis, neighbor_prediction
                )
                reviewer_changed_topic   = reviewed_analysis.topic   != first_analysis.topic
                reviewer_changed_urgency = reviewed_analysis.urgency != first_analysis.urgency
                reviewer_note            = reviewed_analysis.short_note
                final_analysis   = reviewed_analysis
                final_validation = reviewed_validation
            # else: reviewer returned the fallback (confidence=0.0) — keep first analysis.

        # Step 5 — deterministic routing.
        t0 = time.perf_counter()
        next_action: NextAction = self.router.route(final_analysis, final_validation)
        routing_seconds = time.perf_counter() - t0

        # Step 6 — simulated action execution.
        t0 = time.perf_counter()
        action_result: ActionExecutionResult = self.action_executor.execute(
            next_action=next_action,
            ticket_id=ticket.ticket_id,
            short_note=final_analysis.short_note,
        )
        action_execution_seconds = time.perf_counter() - t0

        total_ticket_seconds = time.perf_counter() - t_total

        # Step 7 — assemble TriageResult.
        result = TriageResult(
            ticket_id=ticket.ticket_id,
            text_snippet=ticket.text_snippet,
            topic=final_analysis.topic,
            urgency=final_analysis.urgency,
            next_action=next_action,
            confidence=final_analysis.confidence,
            missing_info=final_analysis.missing_info,
            missing_fields=final_analysis.missing_fields,
            requires_human_review=final_validation.requires_human_review,
            short_note=final_analysis.short_note,
            action_result=action_result,
        )

        timing = {
            "retrieval_seconds":          round(retrieval_seconds, 4),
            "llm_seconds":                round(llm_seconds, 4),
            "validation_seconds":         round(validation_seconds, 4),
            "reviewer_seconds":           round(reviewer_seconds, 4),
            "routing_seconds":            round(routing_seconds, 4),
            "action_execution_seconds":   round(action_execution_seconds, 4),
            "total_ticket_seconds":       round(total_ticket_seconds, 4),
            # Reviewer trace fields.
            "reviewer_used":              reviewer_used,
            "reviewer_model":             reviewer_model,
            "reviewer_changed_topic":     reviewer_changed_topic,
            "reviewer_changed_urgency":   reviewer_changed_urgency,
            "reviewer_trigger_flags":     reviewer_trigger_flags,
            # Pre-reviewer LLM outputs (prediction-side fields, no labels).
            "first_topic":                first_topic,
            "first_urgency":              first_urgency,
            "first_confidence":           first_confidence,
            # Explainability fields — notes and validator output.
            "first_short_note":           first_short_note,
            "reviewer_note":              reviewer_note,
            "validator_flags":            validator_flags,
            "validator_notes":            validator_notes,
            # Neighbor retrieval evidence (historical, not current-ticket labels).
            "neighbor_predicted_topic":    neighbor_predicted_topic,
            "neighbor_topic_confidence":   round(neighbor_topic_confidence, 4),
            "neighbor_predicted_priority": neighbor_predicted_priority,
            "neighbor_priority_confidence": round(neighbor_priority_confidence, 4),
        }
        return result, timing
