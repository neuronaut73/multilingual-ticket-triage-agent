"""
Unit tests for src/application/validator.py (TriageValidator).

All tests use plain LLMAnalysis and NeighborPrediction objects.
No Ollama, no embeddings, no LanceDB is called.

Each test targets one or two specific validation checks so that
failures are easy to diagnose. The helper functions make_analysis()
and make_prediction() provide sane defaults; individual tests override
only the fields relevant to the check being exercised.
"""

import pytest

from src.application.validator import TriageValidator, _map_priority_to_urgency
from src.domain.enums import Topic, Urgency
from src.domain.models import LLMAnalysis, NeighborPrediction


# ─── Shared helpers ───────────────────────────────────────────────────────────

def make_analysis(**overrides) -> LLMAnalysis:
    """
    Return an LLMAnalysis with sane defaults.
    Overrides are applied on top of the defaults.
    """
    defaults = dict(
        topic=Topic.TECHNICAL,
        urgency=Urgency.HIGH,
        missing_info=False,
        missing_fields=[],
        confidence=0.85,
        short_note="Customer reports a login issue.",
    )
    defaults.update(overrides)
    return LLMAnalysis(**defaults)


def make_prediction(**overrides) -> NeighborPrediction:
    """
    Return a NeighborPrediction with sane defaults.
    Overrides are applied on top of the defaults.
    """
    defaults = dict(
        predicted_queue="Technical Support",
        queue_confidence=0.80,
        predicted_priority="high",
        priority_confidence=0.80,
        predicted_proxy_topic="Technical / Online Access",
        proxy_topic_confidence=0.80,
        predicted_tags=[],
        neighbors=[],
    )
    defaults.update(overrides)
    return NeighborPrediction(**defaults)


def default_validator() -> TriageValidator:
    return TriageValidator(
        min_llm_confidence=0.60,
        min_neighbor_confidence=0.50,
        high_confidence_threshold=0.80,
    )


# ─── Priority → Urgency helper ────────────────────────────────────────────────

def test_map_priority_to_urgency_high():
    assert _map_priority_to_urgency("high")   == Urgency.HIGH

def test_map_priority_to_urgency_medium():
    assert _map_priority_to_urgency("medium") == Urgency.MEDIUM

def test_map_priority_to_urgency_low():
    assert _map_priority_to_urgency("low")    == Urgency.LOW

def test_map_priority_to_urgency_case_insensitive():
    assert _map_priority_to_urgency("HIGH")   == Urgency.HIGH
    assert _map_priority_to_urgency("Medium") == Urgency.MEDIUM

def test_map_priority_to_urgency_none_returns_none():
    assert _map_priority_to_urgency(None) is None

def test_map_priority_to_urgency_unknown_returns_none():
    assert _map_priority_to_urgency("critical") is None


# ─── Check A: LLM confidence ──────────────────────────────────────────────────

def test_high_confidence_aligned_analysis_no_flags():
    """Happy path: high confidence, all signals aligned, no flags expected."""
    analysis   = make_analysis(confidence=0.85)
    prediction = make_prediction()

    result = default_validator().validate(analysis, prediction)

    assert result.is_valid is True
    assert result.requires_human_review is False
    assert result.flags == []
    assert result.notes == []


def test_low_llm_confidence_sets_flag_and_review():
    analysis   = make_analysis(confidence=0.45)
    prediction = make_prediction()

    result = default_validator().validate(analysis, prediction)

    assert "low_llm_confidence"   in result.flags
    assert result.requires_human_review is True


def test_confidence_at_threshold_is_not_flagged():
    """Confidence exactly at min_llm_confidence should not be flagged."""
    analysis   = make_analysis(confidence=0.60)
    prediction = make_prediction()

    result = default_validator().validate(analysis, prediction)

    assert "low_llm_confidence" not in result.flags


# ─── Check B: Missing information ─────────────────────────────────────────────

def test_missing_info_true_sets_flag_and_review():
    analysis   = make_analysis(missing_info=True, missing_fields=["policy_number"])
    prediction = make_prediction()

    result = default_validator().validate(analysis, prediction)

    assert "missing_information"  in result.flags
    assert result.requires_human_review is True


def test_missing_info_true_with_empty_missing_fields_adds_extra_flag():
    analysis   = make_analysis(missing_info=True, missing_fields=[])
    prediction = make_prediction()

    result = default_validator().validate(analysis, prediction)

    assert "missing_information"         in result.flags
    assert "missing_fields_not_specified" in result.flags
    assert result.requires_human_review  is True


def test_missing_info_false_no_flag():
    analysis   = make_analysis(missing_info=False)
    prediction = make_prediction()

    result = default_validator().validate(analysis, prediction)

    assert "missing_information"         not in result.flags
    assert "missing_fields_not_specified" not in result.flags


# ─── Check C: Neighbor priority confidence ────────────────────────────────────

def test_low_neighbor_priority_confidence_sets_flag():
    analysis   = make_analysis()
    prediction = make_prediction(priority_confidence=0.30)

    result = default_validator().validate(analysis, prediction)

    assert "low_neighbor_priority_confidence" in result.flags
    assert result.requires_human_review is True


def test_priority_confidence_at_threshold_not_flagged():
    analysis   = make_analysis()
    prediction = make_prediction(priority_confidence=0.50)

    result = default_validator().validate(analysis, prediction)

    assert "low_neighbor_priority_confidence" not in result.flags


# ─── Check D: Neighbor topic confidence ───────────────────────────────────────

def test_low_neighbor_topic_confidence_when_topic_exists():
    analysis   = make_analysis()
    prediction = make_prediction(
        predicted_proxy_topic="Technical / Online Access",
        proxy_topic_confidence=0.30,
    )

    result = default_validator().validate(analysis, prediction)

    assert "low_neighbor_topic_confidence" in result.flags
    assert result.requires_human_review is True


def test_low_topic_confidence_skipped_when_no_topic_prediction():
    """Check D must not fire when predicted_proxy_topic is None."""
    analysis   = make_analysis()
    prediction = make_prediction(
        predicted_proxy_topic=None,
        proxy_topic_confidence=0.0,
    )

    result = default_validator().validate(analysis, prediction)

    assert "low_neighbor_topic_confidence" not in result.flags


# ─── Check E: Urgency disagreement ───────────────────────────────────────────

def test_urgency_disagreement_high_neighbor_confidence_requires_review():
    """
    LLM says Medium, neighbor says high with confidence 0.90.
    Confidence >= high_confidence_threshold → requires_human_review.
    """
    analysis   = make_analysis(urgency=Urgency.MEDIUM, confidence=0.85)
    prediction = make_prediction(predicted_priority="high", priority_confidence=0.90)

    result = default_validator().validate(analysis, prediction)

    assert "urgency_disagreement"     in result.flags
    assert result.requires_human_review is True


def test_urgency_disagreement_mid_neighbor_confidence_does_not_require_review():
    """
    LLM says Medium, neighbor says high with confidence 0.70.
    0.50 <= confidence < 0.80 → flag raised but no human review from this check.
    (No other checks should fire with these values.)
    """
    analysis   = make_analysis(urgency=Urgency.MEDIUM, confidence=0.85)
    prediction = make_prediction(predicted_priority="high", priority_confidence=0.70)

    result = default_validator().validate(analysis, prediction)

    assert "urgency_disagreement" in result.flags
    assert result.requires_human_review is False


def test_urgency_agreement_no_flag():
    analysis   = make_analysis(urgency=Urgency.HIGH)
    prediction = make_prediction(predicted_priority="high", priority_confidence=0.90)

    result = default_validator().validate(analysis, prediction)

    assert "urgency_disagreement" not in result.flags


def test_urgency_disagreement_note_contains_both_values():
    analysis   = make_analysis(urgency=Urgency.MEDIUM, confidence=0.85)
    prediction = make_prediction(predicted_priority="high", priority_confidence=0.90)

    result = default_validator().validate(analysis, prediction)

    assert any("Medium" in note and "high" in note for note in result.notes)


def test_urgency_disagreement_unknown_neighbor_priority_no_flag():
    """If the neighbor priority is unrecognised, no urgency disagreement should fire."""
    analysis   = make_analysis(urgency=Urgency.HIGH)
    prediction = make_prediction(predicted_priority="critical", priority_confidence=0.90)

    result = default_validator().validate(analysis, prediction)

    assert "urgency_disagreement" not in result.flags


def test_urgency_disagreement_none_neighbor_priority_no_flag():
    analysis   = make_analysis()
    prediction = make_prediction(predicted_priority=None, priority_confidence=0.0)

    result = default_validator().validate(analysis, prediction)

    assert "urgency_disagreement" not in result.flags


# ─── Check F: Topic disagreement ─────────────────────────────────────────────

def test_topic_disagreement_high_neighbor_confidence_requires_review():
    """
    LLM says Other, neighbor says Technical with confidence 0.90.
    Confidence >= high_confidence_threshold → requires_human_review.
    """
    analysis   = make_analysis(topic=Topic.OTHER, urgency=Urgency.MEDIUM, confidence=0.85)
    prediction = make_prediction(
        predicted_proxy_topic="Technical / Online Access",
        proxy_topic_confidence=0.90,
    )

    result = default_validator().validate(analysis, prediction)

    assert "topic_disagreement"       in result.flags
    assert result.requires_human_review is True


def test_topic_disagreement_mid_neighbor_confidence_does_not_require_review():
    """
    LLM says Other, neighbor says Technical with confidence 0.65.
    0.50 <= confidence < 0.80 → flag raised but no human review from this check.
    (Use urgency=Low so Check G does not also fire.)
    """
    analysis   = make_analysis(topic=Topic.OTHER, urgency=Urgency.LOW, confidence=0.85)
    prediction = make_prediction(
        predicted_proxy_topic="Technical / Online Access",
        proxy_topic_confidence=0.65,
        predicted_priority="low",
        priority_confidence=0.65,
    )

    result = default_validator().validate(analysis, prediction)

    assert "topic_disagreement"  in result.flags
    assert result.requires_human_review is False


def test_topic_agreement_no_flag():
    analysis   = make_analysis(topic=Topic.TECHNICAL)
    prediction = make_prediction(predicted_proxy_topic="Technical / Online Access")

    result = default_validator().validate(analysis, prediction)

    assert "topic_disagreement" not in result.flags


def test_topic_disagreement_note_contains_both_values():
    analysis   = make_analysis(topic=Topic.OTHER, urgency=Urgency.MEDIUM, confidence=0.85)
    prediction = make_prediction(
        predicted_proxy_topic="Technical / Online Access",
        proxy_topic_confidence=0.90,
        predicted_priority="medium",
        priority_confidence=0.90,
    )

    result = default_validator().validate(analysis, prediction)

    assert any("Other" in note and "Technical" in note for note in result.notes)


def test_topic_disagreement_none_neighbor_topic_no_flag():
    analysis   = make_analysis()
    prediction = make_prediction(predicted_proxy_topic=None)

    result = default_validator().validate(analysis, prediction)

    assert "topic_disagreement" not in result.flags


# ─── Check G: High urgency with limited confidence ────────────────────────────

def test_high_urgency_with_low_confidence_sets_flag():
    analysis   = make_analysis(urgency=Urgency.HIGH, confidence=0.55)
    prediction = make_prediction(
        predicted_priority="high",
        priority_confidence=0.55,
    )

    result = default_validator().validate(analysis, prediction)

    assert "high_urgency_with_limited_confidence" in result.flags
    assert result.requires_human_review is True


def test_high_urgency_with_sufficient_confidence_no_check_g_flag():
    """confidence >= high_confidence_threshold → Check G must not fire."""
    analysis   = make_analysis(urgency=Urgency.HIGH, confidence=0.85)
    prediction = make_prediction(predicted_priority="high", priority_confidence=0.85)

    result = default_validator().validate(analysis, prediction)

    assert "high_urgency_with_limited_confidence" not in result.flags


def test_medium_urgency_does_not_trigger_check_g():
    analysis   = make_analysis(urgency=Urgency.MEDIUM, confidence=0.55)
    prediction = make_prediction(
        predicted_priority="medium",
        priority_confidence=0.55,
    )

    result = default_validator().validate(analysis, prediction)

    assert "high_urgency_with_limited_confidence" not in result.flags


# ─── Edge cases ───────────────────────────────────────────────────────────────

def test_empty_neighbor_prediction_does_not_crash():
    """
    An entirely empty NeighborPrediction (all None / 0.0) must not raise.
    Check C will fire (priority_confidence=0.0 < 0.50), which is acceptable.
    """
    analysis   = make_analysis()
    prediction = NeighborPrediction()

    result = default_validator().validate(analysis, prediction)

    assert isinstance(result.flags, list)
    assert "low_neighbor_priority_confidence" in result.flags


def test_is_valid_always_true():
    """is_valid is always True — structural validation already passed in Sprint 5A."""
    analysis   = make_analysis(confidence=0.10, missing_info=True, missing_fields=[])
    prediction = make_prediction(priority_confidence=0.10, proxy_topic_confidence=0.10)

    result = default_validator().validate(analysis, prediction)

    assert result.is_valid is True


def test_multiple_flags_accumulate():
    """Multiple checks can fire simultaneously."""
    analysis = make_analysis(
        confidence=0.40,        # triggers Check A
        missing_info=True,      # triggers Check B
        missing_fields=[],      # triggers B extra flag
        urgency=Urgency.MEDIUM,
    )
    prediction = make_prediction(
        priority_confidence=0.30,       # triggers Check C (and Check E low → no review)
        proxy_topic_confidence=0.30,    # triggers Check D
        predicted_priority="high",      # triggers Check E (disagrees with Medium)
    )

    result = default_validator().validate(analysis, prediction)

    assert "low_llm_confidence"              in result.flags
    assert "missing_information"             in result.flags
    assert "missing_fields_not_specified"    in result.flags
    assert "low_neighbor_priority_confidence" in result.flags
    assert result.requires_human_review is True
