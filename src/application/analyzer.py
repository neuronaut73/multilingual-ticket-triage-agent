"""
Sprint 5A — Local LLM Analyzer with Pydantic feedback loop.

LocalLLMAnalyzer:
  1. Builds a structured prompt from ticket text and neighbor evidence.
  2. Calls the local LLM via llm_client.generate_json().
  3. Validates the response with Pydantic (LLMAnalysis.model_validate_json).
  4. On validation failure, sends a correction prompt and retries once.
  5. On repeated failure, returns a safe fallback LLMAnalysis.

The feedback loop is agentic in the sense that the system reacts to its own
output, but it requires only one LLM — no multiple agents needed.

Allowed LLM output schema:
  {
    "topic":         one of the five assignment Topic values,
    "urgency":       Low | Medium | High,
    "missing_info":  true | false,
    "missing_fields": [list of missing field names],
    "confidence":    float 0.0 to 1.0,
    "short_note":    1-2 sentence summary
  }
"""

from __future__ import annotations

from pydantic import ValidationError

from src.domain.enums import Topic, Urgency
from src.domain.models import LLMAnalysis, NeighborPrediction, TicketInput

# Short descriptions for each topic — included in every triage prompt.
_TOPIC_DESCRIPTIONS: dict[Topic, str] = {
    Topic.POLICY:    "Policy documents, coverage, contract changes, cancellation, subscriptions, warranty.",
    Topic.CLAIMS:    "Insurance claim, accident, damage, theft, loss, repair, reimbursement.",
    Topic.BILLING:   "Invoice, premium, payment, refund, duplicate charge, direct debit, billing issue.",
    Topic.TECHNICAL: "Login, portal, app, password, authentication, online account, access, outage.",
    Topic.OTHER:     "Unclear or insufficiently specific request.",
}

# Exact JSON shape shown to the LLM in both the initial and correction prompts.
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
    """Return a safe default LLMAnalysis when LLM output cannot be validated."""
    return LLMAnalysis(
        topic=Topic.OTHER,
        urgency=Urgency.MEDIUM,
        missing_info=True,
        missing_fields=["valid_structured_llm_output"],
        confidence=0.0,
        short_note="LLM output could not be validated after retry.",
    )


def _build_prompt(ticket: TicketInput, neighbor_prediction: NeighborPrediction) -> str:
    """
    Build the initial triage prompt sent to the LLM.

    Includes:
      - Role and task description.
      - Allowed topic values with short descriptions.
      - Allowed urgency values.
      - Ticket subject and body (body truncated to 1500 chars).
      - Weighted neighbor prediction: predicted queue, priority, topic, tags.
      - Top-5 neighbor snippet summaries.
      - Required JSON output schema.
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

    lines = [
        "ROLE: You are an insurance support ticket triage assistant.",
        "TASK: Analyze the ticket below and output a structured JSON triage decision.",
        "",
        "ALLOWED TOPICS (output exactly one of these values for the topic field):",
        topic_lines,
        "",
        f"ALLOWED URGENCY VALUES: {urgency_values}",
        "",
        "INSTRUCTIONS:",
        "  - Use only the ticket text and provided neighbor evidence.",
        "  - Do not invent customer facts.",
        "  - Historical queue/priority are evidence, not final labels.",
        "  - The topic field must be one of the allowed topics listed above.",
        "  - If the ticket is too vague to classify, set missing_info = true.",
        "  - If neighbor evidence strongly indicates high priority, consider urgency High.",
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
        "OUTPUT (JSON only, no markdown, no other text):",
        _SCHEMA_HINT,
    ]
    return "\n".join(lines)


def _build_correction_prompt(raw_response: str, error: Exception) -> str:
    """
    Build a correction prompt that shows the LLM its invalid output and the error.

    The LLM sees:
      - What it previously returned (so it can identify the mistake).
      - The Pydantic validation error (so it knows what was wrong).
      - The exact required JSON schema (so it can fix the output).
    """
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


class LocalLLMAnalyzer:
    """
    Analyze a ticket with a local LLM and validate the output with Pydantic.

    If the first LLM response fails validation, one correction prompt is sent
    (agentic feedback loop — no second LLM agent required).

    If the retry also fails, a safe fallback LLMAnalysis is returned.

    Parameters
    ----------
    llm_client:
        Any object with a generate_json(prompt: str) -> str method.
        In production this is OllamaClient. In tests a simple fake is used.
    max_retries:
        Number of correction retries on validation failure. Sprint 5A uses 1.
    debug:
        If True, print the raw LLM response before validation.
    """

    def __init__(
        self,
        llm_client,
        max_retries: int = 1,
        debug: bool = False,
    ) -> None:
        self.llm_client  = llm_client
        self.max_retries = max_retries
        self.debug       = debug

    def analyze(
        self,
        ticket: TicketInput,
        neighbor_prediction: NeighborPrediction,
    ) -> LLMAnalysis:
        """
        Run one ticket through the LLM and return a validated LLMAnalysis.

        Steps:
          1. Build a structured prompt from ticket text and neighbor evidence.
          2. Call llm_client.generate_json(prompt).
          3. Parse and validate with LLMAnalysis.model_validate_json(raw).
          4. On failure: build a correction prompt, retry once.
          5. On repeated failure: return safe fallback.
        """
        prompt = _build_prompt(ticket, neighbor_prediction)
        raw    = self.llm_client.generate_json(prompt)

        if self.debug:
            print(f"\n  [debug] Raw LLM response:\n{raw}\n")

        try:
            return LLMAnalysis.model_validate_json(raw)
        except (ValidationError, ValueError) as first_error:
            if self.max_retries < 1:
                return _make_fallback()

            correction_prompt = _build_correction_prompt(raw, first_error)
            raw_retry         = self.llm_client.generate_json(correction_prompt)

            if self.debug:
                print(f"\n  [debug] Raw LLM retry response:\n{raw_retry}\n")

            try:
                return LLMAnalysis.model_validate_json(raw_retry)
            except (ValidationError, ValueError):
                return _make_fallback()
