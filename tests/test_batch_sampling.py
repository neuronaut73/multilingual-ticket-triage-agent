"""
Mini Sprint — Balanced Evaluation Sampling.

Tests for _sample_tickets() and _balanced_sample() in main.py.

All tests are fully in-memory: no DuckDB, LanceDB, embeddings, or Ollama.

Data leakage note tested here:
  proxy_* labels appear in the fetched rows as evaluation metadata.
  The sampling functions use proxy_* only to select rows.
  BatchRunner (tested separately in test_batch_runner.py) ensures that
  TicketInput is built exclusively from text fields (subject, body, etc.).

Coverage:
  - natural: returns first limit rows, preserves ORDER BY ticket_id order
  - random:  deterministic with same seed, differs with different seed
  - balanced_proxy_topic:  at most limit_per_label rows per proxy_topic class
  - balanced_proxy_urgency: at most limit_per_label rows per proxy_urgency class
  - underrepresented class: takes all rows, does not crash
  - empty class list: no crash, no synthetic rows
  - unknown strategy: raises ValueError
  - result rows include proxy_* and actual_* for post-prediction evaluation
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.abspath("."))

from main import _sample_tickets, _balanced_sample


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

def _make_rows(
    n: int,
    proxy_topic: str = "Technical / Online Access",
    proxy_urgency: str = "High",
) -> list[dict]:
    """Build n minimal ticket rows with required fields."""
    return [
        {
            "ticket_id":           f"ticket_{i:04d}",
            "subject":             f"Subject {i}",
            "body":                f"Body {i}",
            "raw_text":            f"Subject {i} Body {i}",
            "cleaned_text":        f"Subject {i} Body {i}",
            "representation_text": f"Subject: Subject {i}\n\nBody: Body {i}",
            "text_snippet":        f"Subject {i} Body {i}"[:60],
            "actual_queue":        "Technical Support",
            "actual_priority":     "high",
            "actual_type":         "Incident",
            "actual_tags_json":    "[]",
            "proxy_topic":         proxy_topic,
            "proxy_urgency":       proxy_urgency,
            "proxy_next_action":   "forward_to_technical_support",
            "proxy_topic_source":  "queue_mapping",
        }
        for i in range(n)
    ]


def _mixed_rows() -> list[dict]:
    """
    30 rows spanning 3 proxy_topic classes and 3 proxy_urgency classes.
    10 rows per topic class, 10 rows per urgency class.
    """
    topics    = ["Technical / Online Access", "Billing / Payment", "Other"]
    urgencies = ["High", "Medium", "Low"]
    rows = []
    idx = 0
    for t, u in zip(topics, urgencies):
        rows.extend(_make_rows(10, proxy_topic=t, proxy_urgency=u))
        idx += 10
    return rows


# ── natural strategy ───────────────────────────────────────────────────────────

class TestNaturalStrategy:

    def test_returns_exactly_limit_rows(self) -> None:
        rows = _make_rows(20)
        result = _sample_tickets(rows, "natural", limit=10, random_seed=42, limit_per_label=5)
        assert len(result) == 10

    def test_returns_all_when_fewer_than_limit(self) -> None:
        rows = _make_rows(5)
        result = _sample_tickets(rows, "natural", limit=10, random_seed=42, limit_per_label=5)
        assert len(result) == 5

    def test_preserves_input_order(self) -> None:
        """natural strategy must not shuffle — ORDER BY ticket_id order is preserved."""
        rows = _make_rows(10)
        result = _sample_tickets(rows, "natural", limit=5, random_seed=42, limit_per_label=5)
        assert [r["ticket_id"] for r in result] == [rows[i]["ticket_id"] for i in range(5)]

    def test_empty_input_returns_empty(self) -> None:
        result = _sample_tickets([], "natural", limit=10, random_seed=42, limit_per_label=5)
        assert result == []


# ── random strategy ────────────────────────────────────────────────────────────

class TestRandomStrategy:

    def test_returns_exactly_limit_rows(self) -> None:
        rows = _make_rows(50)
        result = _sample_tickets(rows, "random", limit=20, random_seed=42, limit_per_label=5)
        assert len(result) == 20

    def test_same_seed_is_deterministic(self) -> None:
        rows = _make_rows(50)
        r1 = _sample_tickets(rows, "random", limit=20, random_seed=7, limit_per_label=5)
        r2 = _sample_tickets(rows, "random", limit=20, random_seed=7, limit_per_label=5)
        assert [r["ticket_id"] for r in r1] == [r["ticket_id"] for r in r2]

    def test_different_seeds_produce_different_order(self) -> None:
        rows = _make_rows(50)
        r1 = _sample_tickets(rows, "random", limit=20, random_seed=1, limit_per_label=5)
        r2 = _sample_tickets(rows, "random", limit=20, random_seed=2, limit_per_label=5)
        # With 50 rows and limit 20 the probability of identical order is negligible.
        assert [r["ticket_id"] for r in r1] != [r["ticket_id"] for r in r2]

    def test_returns_all_when_fewer_than_limit(self) -> None:
        rows = _make_rows(5)
        result = _sample_tickets(rows, "random", limit=10, random_seed=42, limit_per_label=5)
        assert len(result) == 5

    def test_empty_input_returns_empty(self) -> None:
        result = _sample_tickets([], "random", limit=10, random_seed=42, limit_per_label=5)
        assert result == []


# ── balanced_proxy_topic strategy ─────────────────────────────────────────────

class TestBalancedProxyTopic:

    def test_at_most_limit_per_label_per_class(self) -> None:
        rows = _mixed_rows()  # 10 rows per class, 3 classes
        result = _sample_tickets(
            rows, "balanced_proxy_topic", limit=999, random_seed=42, limit_per_label=5
        )
        from collections import Counter
        counts = Counter(r["proxy_topic"] for r in result)
        for count in counts.values():
            assert count <= 5

    def test_total_is_classes_times_limit_per_label_when_enough_rows(self) -> None:
        rows = _mixed_rows()  # 10 rows per class, 3 classes, limit_per_label=5
        result = _sample_tickets(
            rows, "balanced_proxy_topic", limit=999, random_seed=42, limit_per_label=5
        )
        assert len(result) == 15  # 3 classes × 5

    def test_all_classes_represented(self) -> None:
        rows = _mixed_rows()
        result = _sample_tickets(
            rows, "balanced_proxy_topic", limit=999, random_seed=42, limit_per_label=5
        )
        topics = {r["proxy_topic"] for r in result}
        assert "Technical / Online Access" in topics
        assert "Billing / Payment"         in topics
        assert "Other"                     in topics

    def test_same_seed_is_deterministic(self) -> None:
        rows = _mixed_rows()
        r1 = _sample_tickets(
            rows, "balanced_proxy_topic", limit=999, random_seed=99, limit_per_label=5
        )
        r2 = _sample_tickets(
            rows, "balanced_proxy_topic", limit=999, random_seed=99, limit_per_label=5
        )
        assert [r["ticket_id"] for r in r1] == [r["ticket_id"] for r in r2]

    def test_underrepresented_class_takes_all_and_does_not_crash(self) -> None:
        """A class with 2 rows and limit_per_label=10 must take all 2 rows."""
        rows = _make_rows(10, proxy_topic="Technical / Online Access")
        rows += _make_rows(2, proxy_topic="Other")
        result = _sample_tickets(
            rows, "balanced_proxy_topic", limit=999, random_seed=42, limit_per_label=10
        )
        from collections import Counter
        counts = Counter(r["proxy_topic"] for r in result)
        assert counts["Other"] == 2
        assert counts["Technical / Online Access"] == 10

    def test_no_synthetic_rows_created(self) -> None:
        """Total rows out must never exceed total rows in."""
        rows = _make_rows(3, proxy_topic="Technical / Online Access")
        result = _sample_tickets(
            rows, "balanced_proxy_topic", limit=999, random_seed=42, limit_per_label=10
        )
        assert len(result) <= len(rows)

    def test_empty_input_returns_empty(self) -> None:
        result = _sample_tickets(
            [], "balanced_proxy_topic", limit=999, random_seed=42, limit_per_label=5
        )
        assert result == []


# ── balanced_proxy_urgency strategy ───────────────────────────────────────────

class TestBalancedProxyUrgency:

    def test_at_most_limit_per_label_per_class(self) -> None:
        rows = _mixed_rows()  # 10 rows per urgency class, 3 classes
        result = _sample_tickets(
            rows, "balanced_proxy_urgency", limit=999, random_seed=42, limit_per_label=4
        )
        from collections import Counter
        counts = Counter(r["proxy_urgency"] for r in result)
        for count in counts.values():
            assert count <= 4

    def test_all_urgency_classes_represented(self) -> None:
        rows = _mixed_rows()
        result = _sample_tickets(
            rows, "balanced_proxy_urgency", limit=999, random_seed=42, limit_per_label=4
        )
        urgencies = {r["proxy_urgency"] for r in result}
        assert "High"   in urgencies
        assert "Medium" in urgencies
        assert "Low"    in urgencies

    def test_same_seed_is_deterministic(self) -> None:
        rows = _mixed_rows()
        r1 = _sample_tickets(
            rows, "balanced_proxy_urgency", limit=999, random_seed=7, limit_per_label=4
        )
        r2 = _sample_tickets(
            rows, "balanced_proxy_urgency", limit=999, random_seed=7, limit_per_label=4
        )
        assert [r["ticket_id"] for r in r1] == [r["ticket_id"] for r in r2]

    def test_underrepresented_class_does_not_crash(self) -> None:
        rows = _make_rows(10, proxy_urgency="High")
        rows += _make_rows(1, proxy_urgency="Low")
        result = _sample_tickets(
            rows, "balanced_proxy_urgency", limit=999, random_seed=42, limit_per_label=10
        )
        from collections import Counter
        counts = Counter(r["proxy_urgency"] for r in result)
        assert counts["Low"] == 1


# ── Unknown strategy ───────────────────────────────────────────────────────────

def test_unknown_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown sample_strategy"):
        _sample_tickets(
            _make_rows(5), "fancy_oversampling", limit=5, random_seed=42, limit_per_label=5
        )


# ── Result rows retain evaluation metadata ────────────────────────────────────

class TestResultRowMetadata:
    """
    Sampled rows must include proxy_* and actual_* fields so BatchRunner
    can write them to the output CSV for post-prediction evaluation.
    These fields must never enter TicketInput (covered by test_batch_runner.py).
    """

    def _sampled_row(self) -> dict:
        rows = _mixed_rows()
        result = _sample_tickets(rows, "natural", limit=1, random_seed=42, limit_per_label=5)
        return result[0]

    def test_row_has_proxy_topic(self) -> None:
        assert "proxy_topic" in self._sampled_row()

    def test_row_has_proxy_urgency(self) -> None:
        assert "proxy_urgency" in self._sampled_row()

    def test_row_has_proxy_next_action(self) -> None:
        assert "proxy_next_action" in self._sampled_row()

    def test_row_has_actual_queue(self) -> None:
        assert "actual_queue" in self._sampled_row()

    def test_row_has_actual_priority(self) -> None:
        assert "actual_priority" in self._sampled_row()

    def test_row_has_text_fields_for_ticket_input(self) -> None:
        row = self._sampled_row()
        for field in ("ticket_id", "subject", "body", "representation_text", "text_snippet"):
            assert field in row
