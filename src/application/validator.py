"""
Sprint 5B — Deterministic Validator.

TriageValidator checks the LLM analysis against the neighbor prediction and
returns a ValidationResult with flags, notes, and a requires_human_review signal.

Design principles:
  - Fully deterministic — no LLM calls.
  - Auditable — every flag has a named reason.
  - Does not choose next_action.
  - Does not execute actions.
  - is_valid is always True because Pydantic structural validation already
    happened in Sprint 5A. Flags and requires_human_review carry the signal.

Checks performed (A–G):
  A. LLM confidence below minimum.
  B. Ticket has missing information.
  C. Neighbor priority confidence below minimum.
  D. Neighbor topic confidence below minimum (only when topic prediction exists).
  E. Urgency disagreement between LLM and neighbor priority.
  F. Topic disagreement between LLM and neighbor proxy topic.
  G. High urgency declared with limited LLM confidence.
"""

from __future__ import annotations

from src.domain.enums import Urgency
from src.domain.models import LLMAnalysis, NeighborPrediction, ValidationResult

# Maps raw Kaggle priority strings (lower-cased) to assignment Urgency values.
_PRIORITY_TO_URGENCY: dict[str, Urgency] = {
    "high":   Urgency.HIGH,
    "medium": Urgency.MEDIUM,
    "low":    Urgency.LOW,
}


def _map_priority_to_urgency(priority: str | None) -> Urgency | None:
    """
    Convert a raw kNN-predicted priority string to an Urgency enum value.

    Returns None if the priority string is absent or unrecognised.
    """
    if priority is None:
        return None
    return _PRIORITY_TO_URGENCY.get(priority.strip().lower())


class TriageValidator:
    """
    Deterministic agreement checks between LLM analysis and neighbor evidence.

    Parameters
    ----------
    min_llm_confidence:
        LLM confidence below this value triggers low_llm_confidence review.
    min_neighbor_confidence:
        Neighbor confidence below this value triggers low_neighbor_* review.
    high_confidence_threshold:
        When neighbor confidence reaches this level, disagreements with the LLM
        are treated as definitive enough to require human review.
    """

    def __init__(
        self,
        min_llm_confidence: float = 0.60,
        min_neighbor_confidence: float = 0.50,
        high_confidence_threshold: float = 0.80,
    ) -> None:
        self.min_llm_confidence      = min_llm_confidence
        self.min_neighbor_confidence = min_neighbor_confidence
        self.high_confidence_threshold = high_confidence_threshold

    def validate(
        self,
        analysis: LLMAnalysis,
        neighbor_prediction: NeighborPrediction,
    ) -> ValidationResult:
        """
        Run all checks and return a ValidationResult.

        is_valid is always True — structural validation already passed.
        flags and requires_human_review carry the actionable signal.
        """
        flags:                list[str] = []
        notes:                list[str] = []
        requires_human_review: bool     = False

        # ── Check A: LLM confidence too low ──────────────────────────────────
        if analysis.confidence < self.min_llm_confidence:
            flags.append("low_llm_confidence")
            requires_human_review = True

        # ── Check B: Missing information ──────────────────────────────────────
        if analysis.missing_info:
            flags.append("missing_information")
            requires_human_review = True
            if not analysis.missing_fields:
                flags.append("missing_fields_not_specified")

        # ── Check C: Low neighbor priority confidence ──────────────────────────
        if neighbor_prediction.priority_confidence < self.min_neighbor_confidence:
            flags.append("low_neighbor_priority_confidence")
            requires_human_review = True

        # ── Check D: Low neighbor topic confidence ─────────────────────────────
        # Only meaningful when a proxy topic prediction actually exists.
        if (
            neighbor_prediction.predicted_proxy_topic is not None
            and neighbor_prediction.proxy_topic_confidence < self.min_neighbor_confidence
        ):
            flags.append("low_neighbor_topic_confidence")
            requires_human_review = True

        # ── Check E: Urgency disagreement ──────────────────────────────────────
        mapped_neighbor_urgency = _map_priority_to_urgency(
            neighbor_prediction.predicted_priority
        )
        if (
            mapped_neighbor_urgency is not None
            and mapped_neighbor_urgency != analysis.urgency
        ):
            flags.append("urgency_disagreement")
            notes.append(
                f"LLM urgency = {analysis.urgency.value}, "
                f"neighbor predicted_priority = {neighbor_prediction.predicted_priority} "
                f"(confidence: {neighbor_prediction.priority_confidence:.2f})"
            )
            if neighbor_prediction.priority_confidence >= self.high_confidence_threshold:
                requires_human_review = True

        # ── Check F: Topic disagreement ────────────────────────────────────────
        nb_topic = neighbor_prediction.predicted_proxy_topic
        if nb_topic is not None and nb_topic != analysis.topic.value:
            flags.append("topic_disagreement")
            notes.append(
                f"LLM topic = {analysis.topic.value}, "
                f"neighbor predicted_proxy_topic = {nb_topic} "
                f"(confidence: {neighbor_prediction.proxy_topic_confidence:.2f})"
            )
            if neighbor_prediction.proxy_topic_confidence >= self.high_confidence_threshold:
                requires_human_review = True

        # ── Check G: High urgency with limited LLM confidence ─────────────────
        if (
            analysis.urgency == Urgency.HIGH
            and analysis.confidence < self.high_confidence_threshold
        ):
            flags.append("high_urgency_with_limited_confidence")
            requires_human_review = True

        return ValidationResult(
            is_valid=True,
            requires_human_review=requires_human_review,
            flags=flags,
            notes=notes,
        )
