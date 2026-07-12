"""
Tests for Sprint 6B — metrics.py.

All tests are fully deterministic.
No Ollama, embeddings, LanceDB, or DuckDB calls.

Test coverage:
  - accuracy computes correct fraction
  - accuracy skips rows where actual is empty
  - accuracy returns 0.0 when all actuals are empty
  - macro_precision_recall_f1 handles a simple 2-label example
  - macro_precision_recall_f1 returns zeros for all-empty actuals
  - confusion_counts returns expected (actual, predicted, count) dicts
  - confusion_counts skips rows where actual is empty
  - compute_evaluation_metrics includes all required KPI names
  - compute_evaluation_metrics returns 0.0 for empty rows
  - human_review_rate and missing_info_rate are computed correctly
  - average_confidence is computed correctly
  - operational rates (escalation, faq, billing, technical, claim)
  compute_timing_metrics:
  - returns empty dict for empty input
  - avg is arithmetic mean
  - min and max are correct
  - p50 is median
  - p95 uses nearest-rank method
  - single-element list returns identical values for all stats
"""

import pytest

from src.application.metrics import (
    accuracy,
    compute_timing_metrics,
    compute_reviewer_metrics,
    confusion_counts,
    compute_evaluation_metrics,
    macro_precision_recall_f1,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _row(
    urgency="High",
    proxy_urgency="High",
    topic="Technical / Online Access",
    proxy_topic="Technical / Online Access",
    next_action="forward_to_technical_support",
    proxy_next_action="forward_to_technical_support",
    confidence=0.80,
    requires_human_review=False,
    missing_info=False,
) -> dict:
    return {
        "urgency":               urgency,
        "proxy_urgency":         proxy_urgency,
        "topic":                 topic,
        "proxy_topic":           proxy_topic,
        "next_action":           next_action,
        "proxy_next_action":     proxy_next_action,
        "confidence":            confidence,
        "requires_human_review": requires_human_review,
        "missing_info":          missing_info,
    }


# ─── accuracy ─────────────────────────────────────────────────────────────────

class TestAccuracy:

    def test_all_correct(self) -> None:
        rows = [
            {"pred": "High", "actual": "High"},
            {"pred": "Low",  "actual": "Low"},
        ]
        assert accuracy(rows, "pred", "actual") == 1.0

    def test_partial_correct(self) -> None:
        rows = [
            {"pred": "High",   "actual": "High"},
            {"pred": "High",   "actual": "Low"},
            {"pred": "Medium", "actual": "Medium"},
            {"pred": "Low",    "actual": "High"},
        ]
        # 2 correct out of 4
        assert accuracy(rows, "pred", "actual") == 0.5

    def test_none_correct(self) -> None:
        rows = [
            {"pred": "Low",  "actual": "High"},
            {"pred": "High", "actual": "Low"},
        ]
        assert accuracy(rows, "pred", "actual") == 0.0

    def test_skips_empty_actual(self) -> None:
        rows = [
            {"pred": "High", "actual": "High"},
            {"pred": "High", "actual": ""},        # skipped
            {"pred": "High", "actual": None},      # skipped
        ]
        # Only 1 valid row, and it's correct
        assert accuracy(rows, "pred", "actual") == 1.0

    def test_all_empty_actual_returns_zero(self) -> None:
        rows = [{"pred": "High", "actual": ""}]
        assert accuracy(rows, "pred", "actual") == 0.0

    def test_empty_rows_returns_zero(self) -> None:
        assert accuracy([], "pred", "actual") == 0.0

    def test_strips_whitespace(self) -> None:
        rows = [{"pred": " High ", "actual": "High"}]
        assert accuracy(rows, "pred", "actual") == 1.0


# ─── macro_precision_recall_f1 ────────────────────────────────────────────────

class TestMacroPrecisionRecallF1:

    def test_perfect_predictions(self) -> None:
        rows = [
            {"pred": "High", "actual": "High"},
            {"pred": "Low",  "actual": "Low"},
            {"pred": "High", "actual": "High"},
        ]
        result = macro_precision_recall_f1(rows, "pred", "actual")
        assert result["macro_precision"] == 1.0
        assert result["macro_recall"]    == 1.0
        assert result["macro_f1"]        == 1.0

    def test_symmetric_two_label_example(self) -> None:
        """
        Symmetric confusion: each class predicted wrong once and right once.

        Class A: tp=1, fp=1, fn=1  →  P=0.5, R=0.5, F1=0.5
        Class B: tp=1, fp=1, fn=1  →  P=0.5, R=0.5, F1=0.5
        macro: P=0.5, R=0.5, F1=0.5
        """
        rows = [
            {"pred": "A", "actual": "A"},  # A correct
            {"pred": "B", "actual": "B"},  # B correct
            {"pred": "B", "actual": "A"},  # A predicted as B: FP for B, FN for A
            {"pred": "A", "actual": "B"},  # B predicted as A: FP for A, FN for B
        ]
        result = macro_precision_recall_f1(rows, "pred", "actual")
        assert result["macro_precision"] == 0.5
        assert result["macro_recall"]    == 0.5
        assert result["macro_f1"]        == 0.5

    def test_all_empty_actual_returns_zeros(self) -> None:
        rows = [{"pred": "High", "actual": ""}]
        result = macro_precision_recall_f1(rows, "pred", "actual")
        assert result["macro_precision"] == 0.0
        assert result["macro_recall"]    == 0.0
        assert result["macro_f1"]        == 0.0

    def test_empty_rows_returns_zeros(self) -> None:
        result = macro_precision_recall_f1([], "pred", "actual")
        assert result == {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    def test_skips_empty_actual(self) -> None:
        rows = [
            {"pred": "High", "actual": "High"},
            {"pred": "Low",  "actual": ""},      # skipped
        ]
        result = macro_precision_recall_f1(rows, "pred", "actual")
        assert result["macro_precision"] == 1.0


# ─── confusion_counts ─────────────────────────────────────────────────────────

class TestConfusionCounts:

    def test_perfect_classifier(self) -> None:
        rows = [
            {"pred": "High", "actual": "High"},
            {"pred": "High", "actual": "High"},
            {"pred": "Low",  "actual": "Low"},
        ]
        result = confusion_counts(rows, "pred", "actual")
        # Each (actual, predicted) pair becomes one row
        by_pair = {(r["actual_label"], r["predicted_label"]): r["count"] for r in result}
        assert by_pair[("High", "High")] == 2
        assert by_pair[("Low", "Low")]   == 1

    def test_off_diagonal_entries(self) -> None:
        rows = [
            {"pred": "Low",  "actual": "High"},
            {"pred": "High", "actual": "Low"},
        ]
        result = confusion_counts(rows, "pred", "actual")
        by_pair = {(r["actual_label"], r["predicted_label"]): r["count"] for r in result}
        assert by_pair[("High", "Low")] == 1
        assert by_pair[("Low", "High")] == 1

    def test_skips_empty_actual(self) -> None:
        rows = [
            {"pred": "High", "actual": "High"},
            {"pred": "High", "actual": ""},  # skipped
        ]
        result = confusion_counts(rows, "pred", "actual")
        assert len(result) == 1
        assert result[0]["count"] == 1

    def test_empty_rows_returns_empty_list(self) -> None:
        assert confusion_counts([], "pred", "actual") == []

    def test_result_is_sorted(self) -> None:
        rows = [
            {"pred": "Low",  "actual": "Medium"},
            {"pred": "High", "actual": "High"},
        ]
        result = confusion_counts(rows, "pred", "actual")
        keys = [(r["actual_label"], r["predicted_label"]) for r in result]
        assert keys == sorted(keys)


# ─── compute_evaluation_metrics ───────────────────────────────────────────────

REQUIRED_KPIS = {
    "urgency_accuracy",
    "urgency_macro_precision",
    "urgency_macro_recall",
    "urgency_macro_f1",
    "topic_proxy_accuracy",
    "topic_macro_precision",
    "topic_macro_recall",
    "topic_macro_f1",
    "next_action_proxy_agreement",
    "human_review_rate",
    "missing_info_rate",
    "average_confidence",
    "escalation_rate",
    "faq_rate",
    "billing_forward_rate",
    "technical_forward_rate",
    "claim_action_rate",
}


class TestComputeEvaluationMetrics:

    def test_includes_all_required_kpis(self) -> None:
        rows = [_row()]
        result = compute_evaluation_metrics(rows)
        for kpi in REQUIRED_KPIS:
            assert kpi in result, f"Missing KPI: {kpi}"

    def test_empty_rows_returns_empty_dict(self) -> None:
        assert compute_evaluation_metrics([]) == {}

    def test_urgency_accuracy_correct_value(self) -> None:
        rows = [
            _row(urgency="High",   proxy_urgency="High"),
            _row(urgency="High",   proxy_urgency="Low"),   # wrong
            _row(urgency="Medium", proxy_urgency="Medium"),
            _row(urgency="Low",    proxy_urgency="High"),  # wrong
        ]
        result = compute_evaluation_metrics(rows)
        assert result["urgency_accuracy"] == 0.5

    def test_human_review_rate(self) -> None:
        rows = [
            _row(requires_human_review=True),
            _row(requires_human_review=False),
            _row(requires_human_review=False),
            _row(requires_human_review=False),
        ]
        result = compute_evaluation_metrics(rows)
        assert result["human_review_rate"] == 0.25

    def test_missing_info_rate(self) -> None:
        rows = [
            _row(missing_info=True),
            _row(missing_info=True),
            _row(missing_info=False),
        ]
        result = compute_evaluation_metrics(rows)
        assert abs(result["missing_info_rate"] - 2/3) < 0.0001

    def test_average_confidence(self) -> None:
        rows = [
            _row(confidence=0.8),
            _row(confidence=0.6),
        ]
        result = compute_evaluation_metrics(rows)
        assert result["average_confidence"] == 0.7

    def test_escalation_rate(self) -> None:
        rows = [
            _row(next_action="escalate_to_human_supervisor"),
            _row(next_action="escalate_to_human_supervisor"),
            _row(next_action="forward_to_billing_team"),
        ]
        result = compute_evaluation_metrics(rows)
        assert abs(result["escalation_rate"] - 2/3) < 0.0001

    def test_billing_forward_rate(self) -> None:
        rows = [
            _row(next_action="forward_to_billing_team"),
            _row(next_action="forward_to_technical_support"),
        ]
        result = compute_evaluation_metrics(rows)
        assert result["billing_forward_rate"] == 0.5

    def test_next_action_proxy_agreement_named_correctly(self) -> None:
        rows = [_row()]
        result = compute_evaluation_metrics(rows)
        assert "next_action_proxy_agreement" in result
        assert "next_action_proxy_accuracy" not in result

    def test_topic_proxy_accuracy_named_correctly(self) -> None:
        rows = [_row()]
        result = compute_evaluation_metrics(rows)
        assert "topic_proxy_accuracy" in result
        assert "topic_accuracy" not in result

    def test_all_values_are_floats(self) -> None:
        rows = [_row()]
        result = compute_evaluation_metrics(rows)
        for name, value in result.items():
            assert isinstance(value, float), f"{name} is not float: {type(value)}"


# ─── compute_timing_metrics ───────────────────────────────────────────────────

class TestComputeTimingMetrics:

    def test_empty_input_returns_empty_dict(self) -> None:
        assert compute_timing_metrics([]) == {}

    def test_returns_required_keys(self) -> None:
        result = compute_timing_metrics([1.0, 2.0, 3.0])
        expected_keys = {
            "avg_seconds_per_ticket",
            "p50_seconds_per_ticket",
            "p95_seconds_per_ticket",
            "min_seconds_per_ticket",
            "max_seconds_per_ticket",
        }
        assert expected_keys == set(result.keys())

    def test_avg_is_arithmetic_mean(self) -> None:
        result = compute_timing_metrics([1.0, 2.0, 3.0])
        assert result["avg_seconds_per_ticket"] == 2.0

    def test_min_is_smallest_value(self) -> None:
        result = compute_timing_metrics([5.0, 1.0, 3.0])
        assert result["min_seconds_per_ticket"] == 1.0

    def test_max_is_largest_value(self) -> None:
        result = compute_timing_metrics([5.0, 1.0, 3.0])
        assert result["max_seconds_per_ticket"] == 5.0

    def test_p50_is_median_of_odd_list(self) -> None:
        # median of [1, 2, 3] = 2
        result = compute_timing_metrics([3.0, 1.0, 2.0])
        assert result["p50_seconds_per_ticket"] == 2.0

    def test_p50_is_median_of_even_list(self) -> None:
        # median of [1, 2, 3, 4] = 2.5
        result = compute_timing_metrics([1.0, 2.0, 3.0, 4.0])
        assert result["p50_seconds_per_ticket"] == 2.5

    def test_p95_is_within_range(self) -> None:
        # p95 must be >= p50 and <= max
        values = [float(i) for i in range(1, 21)]  # 20 values: 1..20
        result = compute_timing_metrics(values)
        assert result["p95_seconds_per_ticket"] >= result["p50_seconds_per_ticket"]
        assert result["p95_seconds_per_ticket"] <= result["max_seconds_per_ticket"]

    def test_p95_of_twenty_values(self) -> None:
        # 20 values 1..20 sorted; nearest-rank idx = ceil(0.95*20)-1 = 19-1=18 → value 19
        values = [float(i) for i in range(1, 21)]
        result = compute_timing_metrics(values)
        assert result["p95_seconds_per_ticket"] == 19.0

    def test_single_element_all_equal(self) -> None:
        result = compute_timing_metrics([7.5])
        assert result["avg_seconds_per_ticket"] == 7.5
        assert result["p50_seconds_per_ticket"] == 7.5
        assert result["p95_seconds_per_ticket"] == 7.5
        assert result["min_seconds_per_ticket"] == 7.5
        assert result["max_seconds_per_ticket"] == 7.5

    def test_all_values_are_floats(self) -> None:
        result = compute_timing_metrics([1.0, 2.0, 3.0])
        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be float"


# ─── compute_reviewer_metrics ─────────────────────────────────────────────────

def _reviewer_row(
    reviewer_used: bool = False,
    reviewer_changed_topic: bool = False,
    reviewer_changed_urgency: bool = False,
    reviewer_seconds: float = 0.0,
    requires_human_review: bool = False,
) -> dict:
    """Build a minimal result row for reviewer metric tests."""
    return {
        "reviewer_used":            reviewer_used,
        "reviewer_changed_topic":   reviewer_changed_topic,
        "reviewer_changed_urgency": reviewer_changed_urgency,
        "reviewer_seconds":         reviewer_seconds,
        "requires_human_review":    requires_human_review,
    }


class TestComputeReviewerMetrics:

    def test_empty_rows_returns_zeros(self) -> None:
        result = compute_reviewer_metrics([])
        assert result["reviewer_invocation_rate"]    == 0.0
        assert result["reviewer_topic_change_rate"]  == 0.0
        assert result["reviewer_urgency_change_rate"] == 0.0
        assert result["avg_reviewer_seconds"]        == 0.0

    def test_invocation_rate_uses_reviewer_used_not_requires_human_review(self) -> None:
        """
        reviewer_invocation_rate must count reviewer_used=True rows,
        NOT requires_human_review=True rows.
        """
        rows = [
            # reviewer_used=False but requires_human_review=True — must NOT count
            _reviewer_row(reviewer_used=False, requires_human_review=True),
            # reviewer_used=True — must count
            _reviewer_row(reviewer_used=True,  requires_human_review=False),
            _reviewer_row(reviewer_used=False, requires_human_review=False),
            _reviewer_row(reviewer_used=False, requires_human_review=False),
        ]
        result = compute_reviewer_metrics(rows)
        # Only 1 out of 4 rows had reviewer_used=True.
        assert result["reviewer_invocation_rate"] == pytest.approx(0.25)

    def test_invocation_rate_zero_when_reviewer_never_used(self) -> None:
        rows = [
            _reviewer_row(reviewer_used=False, requires_human_review=True),
            _reviewer_row(reviewer_used=False, requires_human_review=True),
        ]
        result = compute_reviewer_metrics(rows)
        assert result["reviewer_invocation_rate"] == 0.0

    def test_invocation_rate_one_when_all_reviewed(self) -> None:
        rows = [
            _reviewer_row(reviewer_used=True),
            _reviewer_row(reviewer_used=True),
        ]
        result = compute_reviewer_metrics(rows)
        assert result["reviewer_invocation_rate"] == 1.0

    def test_topic_change_rate_only_over_reviewed_rows(self) -> None:
        rows = [
            _reviewer_row(reviewer_used=True,  reviewer_changed_topic=True),
            _reviewer_row(reviewer_used=True,  reviewer_changed_topic=False),
            _reviewer_row(reviewer_used=False, reviewer_changed_topic=True),  # not counted
        ]
        result = compute_reviewer_metrics(rows)
        # 1 out of 2 reviewer_used=True rows changed topic.
        assert result["reviewer_topic_change_rate"] == pytest.approx(0.5)

    def test_urgency_change_rate_only_over_reviewed_rows(self) -> None:
        rows = [
            _reviewer_row(reviewer_used=True,  reviewer_changed_urgency=True),
            _reviewer_row(reviewer_used=True,  reviewer_changed_urgency=True),
            _reviewer_row(reviewer_used=True,  reviewer_changed_urgency=False),
            _reviewer_row(reviewer_used=False, reviewer_changed_urgency=True),  # not counted
        ]
        result = compute_reviewer_metrics(rows)
        # 2 out of 3 reviewer_used=True rows changed urgency; rounded to 4 dp = 0.6667
        assert abs(result["reviewer_urgency_change_rate"] - 2 / 3) < 0.001

    def test_avg_reviewer_seconds_over_all_rows(self) -> None:
        rows = [
            _reviewer_row(reviewer_used=True,  reviewer_seconds=3.0),
            _reviewer_row(reviewer_used=False, reviewer_seconds=0.0),
            _reviewer_row(reviewer_used=False, reviewer_seconds=0.0),
            _reviewer_row(reviewer_used=False, reviewer_seconds=0.0),
        ]
        result = compute_reviewer_metrics(rows)
        # Mean over all 4 rows: (3.0 + 0.0 + 0.0 + 0.0) / 4 = 0.75
        assert result["avg_reviewer_seconds"] == pytest.approx(0.75)

    def test_reviewer_metrics_included_in_compute_evaluation_metrics(self) -> None:
        """compute_evaluation_metrics must include all four reviewer KPIs."""
        rows = [_row()]
        result = compute_evaluation_metrics(rows)
        assert "reviewer_invocation_rate"    in result
        assert "reviewer_topic_change_rate"  in result
        assert "reviewer_urgency_change_rate" in result
        assert "avg_reviewer_seconds"        in result
