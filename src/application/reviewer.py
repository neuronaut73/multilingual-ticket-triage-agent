"""
Sprint 6C.5 — Conditional Reviewer LLM Loop.

ConditionalLLMReviewer is invoked after the first LLM analysis when the
TriageValidator raises one or more configured trigger flags.  It uses the same
Pydantic output model (LLMAnalysis) as the primary analyzer so the rest of the
pipeline is unchanged.

Inputs the reviewer is allowed to see:
  - TicketInput  (subject + body only — no labels)
  - NeighborPrediction  (retrieval evidence from historical neighbors)
  - LLMAnalysis  (the first analysis — no evaluation labels)
  - ValidationResult  (flags and notes — no evaluation labels)

Inputs the reviewer must NOT see:
  - actual_queue / actual_priority / actual_type of the current ticket
  - proxy_topic / proxy_urgency / proxy_next_action of the current ticket

Reviewer output:
  - A revised LLMAnalysis using the same allowed enum values.
  - If the LLM output cannot be validated after retries, _make_fallback()
    is returned with confidence=0.0.  The agent checks confidence > 0 to decide
    whether to use the reviewed analysis or fall back to the first analysis.

Data leakage note:
  The NeighborPrediction contains historical neighbor labels (actual_queue,
  actual_priority, etc.) used as retrieval evidence.  These are the same
  signals shown to the primary analyzer and are NOT labels of the current
  ticket being triaged.  This is safe.
"""

from __future__ import annotations

from pydantic import ValidationError

from src.domain.enums import Topic, Urgency
from src.domain.models import LLMAnalysis, NeighborPrediction, TicketInput, ValidationResult

# Reuse the same topic descriptions as the primary analyzer for consistency.
_TOPIC_DESCRIPTIONS: dict[Topic, str] = {
    Topic.POLICY:    "Policy documents, coverage, contract changes, cancellation, subscriptions, warranty.",
    Topic.CLAIMS:    "Insurance claim, accident, damage, theft, loss, repair, reimbursement.",
    Topic.BILLING:   "Invoice, premium, payment, refund, duplicate charge, direct debit, billing issue.",
    Topic.TECHNICAL: "Login, portal, app, password, authentication, online account, access, outage.",
    Topic.OTHER:     "Unclear or insufficiently specific request.",
}

_SCHEMA_HINT = """\
{
  "topic": "<exactly one of: Policy / Contract | Claims / Damage | Billing / Payment | Technical / Online Access | Other>",
  "urgency": "<exactly one of: Low | Medium | High>",
  "missing_info": <true or false>,
  "missing_fields": ["<field name if info is missing — use empty list if nothing is missing>"],
  "confidence": <float between 0.0 and 1.0>,
  "short_note": "<1-2 sentence plain-text summary of the ticket and triage decision>"
}"""


def _make_fallback() -> LLMAnalysis:
    """Return a sentinel LLMAnalysis with confidence=0.0 when reviewer fails.

    The agent checks confidence > 0 to decide whether the reviewed analysis
    should replace the first analysis.  A confidence of 0.0 means the reviewer
    could not produce a valid response, so the agent falls back to the first
    analysis.
    """
    return LLMAnalysis(
        topic=Topic.OTHER,
        urgency=Urgency.MEDIUM,
        missing_info=True,
        missing_fields=["reviewer_output_invalid"],
        confidence=0.0,
        short_note="Reviewer LLM output could not be validated.",
    )


def _build_reviewer_prompt(
    ticket: TicketInput,
    neighbor_prediction: NeighborPrediction,
    first_analysis: LLMAnalysis,
    validation: ValidationResult,
) -> str:
    """
    Build the reviewer prompt.

    The reviewer sees:
      - Ticket subject and body (inference input — no labels).
      - Neighbor retrieval evidence (historical, not the current ticket's labels).
      - The first analysis produced by the primary LLM.
      - Validation flags and notes that triggered this review.

    The reviewer does NOT see:
      - actual_queue, actual_priority, actual_type, proxy_* of the current ticket.
    """
    topic_lines = "\n".join(
        f"  - {topic.value}: {desc}"
        for topic, desc in _TOPIC_DESCRIPTIONS.items()
    )
    urgency_values = " | ".join(u.value for u in Urgency)

    nb = neighbor_prediction
    evidence_lines = [
        f"  predicted_queue    : {nb.predicted_queue or 'unknown'} (confidence: {nb.queue_confidence:.2f})",
        f"  predicted_priority : {nb.predicted_priority or 'unknown'} (confidence: {nb.priority_confidence:.2f})",
        f"  predicted_topic    : {nb.predicted_proxy_topic or 'unknown'} (confidence: {nb.proxy_topic_confidence:.2f})",
        f"  predicted_tags     : {', '.join(nb.predicted_tags) if nb.predicted_tags else 'none'}",
    ]

    neighbor_summaries = []
    for rank, nbe in enumerate(nb.neighbors[:5], start=1):
        snippet = (nbe.text_snippet or "")[:80].replace("\n", " ")
        neighbor_summaries.append(
            f"  [{rank}] queue={nbe.actual_queue or '?'}"
            f"  priority={nbe.actual_priority or '?'}"
            f"  proxy_topic={nbe.proxy_topic or '?'}"
            f'  snippet="{snippet}"'
        )
    if not neighbor_summaries:
        neighbor_summaries = ["  (no neighbor summaries available)"]

    first_analysis_lines = [
        f"  topic        : {first_analysis.topic.value}",
        f"  urgency      : {first_analysis.urgency.value}",
        f"  confidence   : {first_analysis.confidence}",
        f"  missing_info : {first_analysis.missing_info}",
        f"  missing_fields: {first_analysis.missing_fields}",
        f"  short_note   : {first_analysis.short_note}",
    ]

    flags_text = ", ".join(validation.flags) if validation.flags else "none"
    notes_lines = [f"  - {note}" for note in validation.notes] if validation.notes else ["  (none)"]

    lines = [
        "ROLE: You are a second-pass triage reviewer for an insurance support ticket system.",
        "TASK: Review the first analysis below and decide whether to keep or revise it.",
        "      The first analysis was flagged for review due to uncertainty or disagreement.",
        "",
        "ALLOWED TOPICS (output exactly one of these values for the topic field):",
        topic_lines,
        "",
        f"ALLOWED URGENCY VALUES: {urgency_values}",
        "",
        "INSTRUCTIONS:",
        "  - Use only the ticket text and neighbor evidence below.",
        "  - Do not invent customer facts.",
        "  - Consider the validation flags that triggered this review.",
        "  - If the first analysis is correct, keep the same values.",
        "  - If the first analysis is incorrect, revise topic, urgency, or both.",
        "  - Adjust confidence to reflect your certainty after review.",
        "  - The topic field must be exactly one of the allowed topics above.",
        "  - Output JSON only. No markdown. No explanation outside JSON.",
        "",
        "TICKET:",
        f"Subject: {ticket.subject}",
        f"Body: {ticket.body[:1500]}",
        "",
        "NEIGHBOR EVIDENCE (weighted vote from similar historical tickets):",
        *evidence_lines,
        "",
        "TOP NEIGHBOR SUMMARIES:",
        *neighbor_summaries,
        "",
        "FIRST ANALYSIS (produced by primary LLM — review and correct if needed):",
        *first_analysis_lines,
        "",
        "VALIDATION FLAGS THAT TRIGGERED THIS REVIEW:",
        f"  {flags_text}",
        "",
        "VALIDATION NOTES:",
        *notes_lines,
        "",
        "OUTPUT (JSON only, no markdown, no other text):",
        _SCHEMA_HINT,
    ]
    return "\n".join(lines)


def _build_correction_prompt(raw_response: str, error: Exception) -> str:
    """Correction prompt shown to the reviewer on validation failure."""
    error_summary = str(error)[:500]
    lines = [
        "Your previous response was invalid or did not match the required schema.",
        "",
        "YOUR PREVIOUS RESPONSE:",
        raw_response,
        "",
        "VALIDATION ERROR:",
        error_summary,
        "",
        "REQUIRED JSON SCHEMA (output exactly this shape, no markdown, no other text):",
        _SCHEMA_HINT,
        "",
        "Output ONLY valid JSON matching the schema above.",
    ]
    return "\n".join(lines)


class ConditionalLLMReviewer:
    """
    A second-pass LLM reviewer invoked only when the validator raises trigger flags.

    The reviewer uses the same LLMAnalysis Pydantic model as the primary analyzer.
    If the reviewer produces a valid response with confidence > 0, the agent uses it
    in place of the first analysis.  If the reviewer fails or returns confidence=0.0,
    the agent keeps the first analysis unchanged.

    Parameters
    ----------
    llm_client:
        Any object with generate_json(prompt: str) -> str.
        In production this is a second OllamaClient (potentially a different model).
    trigger_flags:
        List of ValidationResult flag strings that may trigger the reviewer.
        Composite confidence-aware rules are applied on top of this list.
    max_retries:
        Number of correction retries on Pydantic validation failure.
    model_name:
        Model name string stored in reviewer trace fields (for observability).
    disagreement_confidence_ceiling:
        topic_disagreement only triggers the reviewer when first_analysis.confidence
        is strictly below this value.  Default: 0.85.
    urgency_disagreement_confidence_ceiling:
        urgency_disagreement only triggers the reviewer when first_analysis.confidence
        is strictly below this value.  Default: 0.90.
    """

    def __init__(
        self,
        llm_client,
        trigger_flags: list[str],
        max_retries: int = 1,
        model_name: str = "",
        disagreement_confidence_ceiling: float = 0.85,
        urgency_disagreement_confidence_ceiling: float = 0.90,
    ) -> None:
        self.llm_client    = llm_client
        self.trigger_flags = set(trigger_flags)
        self.max_retries   = max_retries
        self.model_name    = model_name
        self.disagreement_confidence_ceiling          = disagreement_confidence_ceiling
        self.urgency_disagreement_confidence_ceiling  = urgency_disagreement_confidence_ceiling

    def get_triggered_review_flags(
        self,
        validation: ValidationResult,
        first_analysis: LLMAnalysis,
    ) -> list[str]:
        """
        Return the subset of validation flags that actually trigger the reviewer.

        Composite confidence-aware rules (evaluated in order):
          - low_llm_confidence         → always triggers; no confidence gate.
          - urgency_disagreement        → triggers only when
                first_analysis.confidence < urgency_disagreement_confidence_ceiling.
          - topic_disagreement          → triggers only when
                first_analysis.confidence < disagreement_confidence_ceiling.
          - low_neighbor_priority_confidence → never triggers the reviewer.
          - low_neighbor_topic_confidence    → never triggers the reviewer.
          - Any missing-information flag     → never triggers the reviewer.
          - Any flag not in trigger_flags    → ignored.

        Returns only the flags that actually fired, in the order they appear
        in validation.flags.
        """
        triggered = []
        for flag in validation.flags:
            if flag not in self.trigger_flags:
                continue
            if flag == "low_llm_confidence":
                triggered.append(flag)
            elif flag == "urgency_disagreement":
                if first_analysis.confidence < self.urgency_disagreement_confidence_ceiling:
                    triggered.append(flag)
            elif flag == "topic_disagreement":
                if first_analysis.confidence < self.disagreement_confidence_ceiling:
                    triggered.append(flag)
            # low_neighbor_priority_confidence, low_neighbor_topic_confidence,
            # missing_information, and all other flags: never trigger the reviewer.
        return triggered

    def should_review(
        self,
        validation: ValidationResult,
        first_analysis: LLMAnalysis | None = None,
    ) -> bool:
        """
        Return True if the composite trigger logic fires for this validation result.

        When first_analysis is provided the full confidence-aware rules apply
        (delegates to get_triggered_review_flags).

        When first_analysis is None (backward-compatibility path), only
        low_llm_confidence can trigger; disagreement flags do not fire without
        the confidence context they require.
        """
        if first_analysis is not None:
            return bool(self.get_triggered_review_flags(validation, first_analysis))
        # Backward-compat: only low_llm_confidence can fire without an analysis object.
        return (
            "low_llm_confidence" in self.trigger_flags
            and "low_llm_confidence" in validation.flags
        )

    def review(
        self,
        ticket: TicketInput,
        neighbor_prediction: NeighborPrediction,
        first_analysis: LLMAnalysis,
        validation: ValidationResult,
    ) -> LLMAnalysis:
        """
        Run the reviewer LLM and return a (possibly revised) LLMAnalysis.

        Steps:
          1. Build reviewer prompt from ticket, evidence, first analysis, flags.
          2. Call llm_client.generate_json(prompt).
          3. Validate with LLMAnalysis.model_validate_json.
          4. On failure: send one correction prompt and retry.
          5. On repeated failure: return _make_fallback() (confidence=0.0).

        The agent checks confidence > 0 to decide whether the reviewed analysis
        should replace the first analysis.

        No evaluation labels are read or passed here.
        """
        prompt = _build_reviewer_prompt(
            ticket, neighbor_prediction, first_analysis, validation
        )
        raw = self.llm_client.generate_json(prompt)

        try:
            return LLMAnalysis.model_validate_json(raw)
        except (ValidationError, ValueError) as first_error:
            if self.max_retries < 1:
                return _make_fallback()

            correction_prompt = _build_correction_prompt(raw, first_error)
            raw_retry = self.llm_client.generate_json(correction_prompt)

            try:
                return LLMAnalysis.model_validate_json(raw_retry)
            except (ValidationError, ValueError):
                return _make_fallback()
