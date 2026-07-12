"""
Tests for Sprint 4: weighted vote helpers and NeighborRetriever.

Strategy:
  - All tests use fake neighbors or fake embedding/store objects.
  - No real EmbeddingModel or LanceDB instance is needed.
  - Tests are fast and do not require the multilingual-e5-large model.

Covered:
  - distance_to_similarity formula
  - weighted_vote: correct label selection, confidence, edge cases
  - aggregate_tags: top-N selection, malformed JSON, deduplication
  - parse_tags_json: safe parsing
  - NeighborRetriever.predict_from_neighbors: full aggregation from neighbors
  - NeighborRetriever.retrieve_and_predict: integration via fake model + store
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.application.neighbor_retriever import NeighborRetriever, _row_to_evidence
from src.application.weighted_vote import (
    aggregate_tags,
    distance_to_similarity,
    parse_tags_json,
    weighted_vote,
)
from src.domain.models import NeighborEvidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_neighbor(
    ticket_id: str = "t001",
    similarity: float = 0.9,
    actual_queue: str | None = "Technical Support",
    actual_priority: str | None = "high",
    proxy_topic: str | None = "Technical / Online Access",
    actual_tags: list[str] | None = None,
) -> NeighborEvidence:
    """Build a minimal NeighborEvidence for testing."""
    return NeighborEvidence(
        ticket_id=ticket_id,
        distance=0.0,
        similarity=similarity,
        actual_queue=actual_queue,
        actual_priority=actual_priority,
        proxy_topic=proxy_topic,
        actual_tags=actual_tags or [],
        text_snippet="Sample snippet.",
    )


# ---------------------------------------------------------------------------
# distance_to_similarity
# ---------------------------------------------------------------------------

def test_distance_to_similarity_zero():
    """Distance 0 -> similarity 1.0."""
    assert distance_to_similarity(0.0) == pytest.approx(1.0)


def test_distance_to_similarity_positive():
    """1 / (1 + 1.0) = 0.5."""
    assert distance_to_similarity(1.0) == pytest.approx(0.5)


def test_distance_to_similarity_large():
    """Large distance -> similarity close to 0."""
    sim = distance_to_similarity(999.0)
    assert 0.0 < sim < 0.01


def test_distance_to_similarity_monotone():
    """Higher distance -> lower similarity."""
    assert distance_to_similarity(0.1) > distance_to_similarity(0.5)
    assert distance_to_similarity(0.5) > distance_to_similarity(2.0)


# ---------------------------------------------------------------------------
# weighted_vote
# ---------------------------------------------------------------------------

def test_weighted_vote_selects_highest_weight():
    pairs = [
        ("Technical Support", 0.9),
        ("Billing",           0.4),
        ("Technical Support", 0.5),
    ]
    label, conf = weighted_vote(pairs)
    assert label == "Technical Support"


def test_weighted_vote_confidence_formula():
    """confidence = winning_weight / total_weight."""
    pairs = [
        ("A", 0.8),
        ("B", 0.2),
    ]
    label, conf = weighted_vote(pairs)
    assert label == "A"
    assert conf == pytest.approx(0.8 / 1.0)


def test_weighted_vote_equal_weights():
    """When weights are equal, one label wins (deterministic via max)."""
    pairs = [("A", 0.5), ("B", 0.5)]
    label, conf = weighted_vote(pairs)
    assert label in ("A", "B")
    assert conf == pytest.approx(0.5)


def test_weighted_vote_empty_returns_none():
    label, conf = weighted_vote([])
    assert label is None
    assert conf == pytest.approx(0.0)


def test_weighted_vote_all_none_labels():
    pairs = [(None, 0.9), (None, 0.7)]
    label, conf = weighted_vote(pairs)
    assert label is None
    assert conf == pytest.approx(0.0)


def test_weighted_vote_mixed_none_and_valid():
    """None labels are ignored; valid labels still compete."""
    pairs = [(None, 0.9), ("Billing", 0.5)]
    label, conf = weighted_vote(pairs)
    assert label == "Billing"
    assert conf == pytest.approx(1.0)


def test_weighted_vote_priority():
    pairs = [
        ("high",   0.8),
        ("medium", 0.6),
        ("high",   0.3),
    ]
    label, conf = weighted_vote(pairs)
    assert label == "high"
    assert conf == pytest.approx(1.1 / 1.7)


def test_weighted_vote_proxy_topic():
    pairs = [
        ("Technical / Online Access", 0.9),
        ("Billing / Payment",         0.4),
        ("Technical / Online Access", 0.2),
    ]
    label, conf = weighted_vote(pairs)
    assert label == "Technical / Online Access"


# ---------------------------------------------------------------------------
# aggregate_tags
# ---------------------------------------------------------------------------

def test_aggregate_tags_returns_top_n():
    pairs = [
        (["network", "login", "vpn"], 0.9),
        (["network", "firewall"],     0.7),
        (["billing"],                 0.5),
    ]
    tags = aggregate_tags(pairs, top_n=2)
    assert len(tags) == 2
    assert tags[0] == "network"   # highest combined weight (0.9 + 0.7)


def test_aggregate_tags_empty_input():
    assert aggregate_tags([], top_n=5) == []


def test_aggregate_tags_all_empty_tag_lists():
    pairs = [([], 0.9), ([], 0.5)]
    assert aggregate_tags(pairs, top_n=5) == []


def test_aggregate_tags_deduplication_per_neighbor():
    """Same tag in one neighbor counted only once for that neighbor."""
    pairs = [
        (["vpn", "vpn", "vpn"], 1.0),  # repeated tags in same neighbor
        (["vpn"],               0.5),
    ]
    tags = aggregate_tags(pairs, top_n=5)
    assert tags == ["vpn"]
    # vpn gets 1.0 + 0.5 = 1.5, not 3.0 + 0.5


def test_aggregate_tags_top_n_respected():
    pairs = [(["a", "b", "c", "d", "e", "f"], 1.0)]
    tags = aggregate_tags(pairs, top_n=3)
    assert len(tags) == 3


# ---------------------------------------------------------------------------
# parse_tags_json
# ---------------------------------------------------------------------------

def test_parse_tags_json_valid():
    assert parse_tags_json('["network", "login"]') == ["network", "login"]


def test_parse_tags_json_empty_array():
    assert parse_tags_json("[]") == []


def test_parse_tags_json_none():
    assert parse_tags_json(None) == []


def test_parse_tags_json_blank_string():
    assert parse_tags_json("") == []


def test_parse_tags_json_malformed():
    """Malformed JSON must not crash — return empty list."""
    assert parse_tags_json("{not valid json") == []


def test_parse_tags_json_non_list():
    """If JSON is valid but not a list, return empty list."""
    assert parse_tags_json('"just a string"') == []


# ---------------------------------------------------------------------------
# _row_to_evidence
# ---------------------------------------------------------------------------

def test_row_to_evidence_basic():
    row = {
        "_distance":       0.5,
        "ticket_id":       "t001",
        "actual_queue":    "Technical Support",
        "actual_priority": "high",
        "actual_type":     "Incident",
        "actual_tags_json": '["vpn", "login"]',
        "proxy_topic":     "Technical / Online Access",
        "proxy_urgency":   "high",
        "proxy_next_action": "forward_to_technical_support",
        "text_snippet":    "User cannot log in.",
    }
    ev = _row_to_evidence(row)
    assert ev.ticket_id == "t001"
    assert ev.distance == pytest.approx(0.5)
    assert ev.similarity == pytest.approx(1.0 / 1.5)
    assert ev.actual_queue == "Technical Support"
    assert ev.actual_tags == ["vpn", "login"]
    assert ev.proxy_topic == "Technical / Online Access"


def test_row_to_evidence_missing_distance_defaults_to_zero():
    row = {"ticket_id": "t002", "text_snippet": "x", "actual_tags_json": "[]"}
    ev = _row_to_evidence(row)
    assert ev.distance == pytest.approx(0.0)
    assert ev.similarity == pytest.approx(1.0)


def test_row_to_evidence_malformed_tags_gives_empty_list():
    row = {
        "_distance":       0.0,
        "ticket_id":       "t003",
        "text_snippet":    "x",
        "actual_tags_json": "NOT_JSON",
    }
    ev = _row_to_evidence(row)
    assert ev.actual_tags == []


# ---------------------------------------------------------------------------
# NeighborRetriever.predict_from_neighbors
# ---------------------------------------------------------------------------

def test_predict_empty_neighbors_returns_safe_defaults():
    retriever = NeighborRetriever(
        embedding_model=None,
        ticket_store=None,
        top_k=5,
    )
    pred = retriever.predict_from_neighbors([])
    assert pred.predicted_queue is None
    assert pred.queue_confidence == pytest.approx(0.0)
    assert pred.predicted_priority is None
    assert pred.predicted_proxy_topic is None
    assert pred.predicted_tags == []


def test_predict_single_neighbor():
    retriever = NeighborRetriever(None, None, top_k=5)
    nb = _make_neighbor(
        actual_queue="Billing and Payments",
        actual_priority="medium",
        proxy_topic="Billing / Payment",
        actual_tags=["invoice", "refund"],
        similarity=0.8,
    )
    pred = retriever.predict_from_neighbors([nb])
    assert pred.predicted_queue == "Billing and Payments"
    assert pred.queue_confidence == pytest.approx(1.0)
    assert pred.predicted_priority == "medium"
    assert pred.predicted_proxy_topic == "Billing / Payment"
    assert "invoice" in pred.predicted_tags or "refund" in pred.predicted_tags


def test_predict_majority_wins():
    """Three neighbors: two Technical Support, one Billing. TS should win."""
    retriever = NeighborRetriever(None, None, top_k=5)
    neighbors = [
        _make_neighbor("t1", 0.9, "Technical Support", "high", "Technical / Online Access"),
        _make_neighbor("t2", 0.7, "Technical Support", "high", "Technical / Online Access"),
        _make_neighbor("t3", 0.5, "Billing",           "low",  "Billing / Payment"),
    ]
    pred = retriever.predict_from_neighbors(neighbors)
    assert pred.predicted_queue == "Technical Support"
    assert pred.predicted_priority == "high"
    assert pred.predicted_proxy_topic == "Technical / Online Access"


def test_predict_confidence_is_fraction_of_total_weight():
    retriever = NeighborRetriever(None, None, top_k=5)
    neighbors = [
        _make_neighbor("t1", 0.8, actual_queue="A"),
        _make_neighbor("t2", 0.2, actual_queue="B"),
    ]
    pred = retriever.predict_from_neighbors(neighbors)
    assert pred.predicted_queue == "A"
    assert pred.queue_confidence == pytest.approx(0.8 / 1.0)


def test_predict_neighbors_attached_to_prediction():
    retriever = NeighborRetriever(None, None, top_k=5)
    nb = _make_neighbor()
    pred = retriever.predict_from_neighbors([nb])
    assert len(pred.neighbors) == 1
    assert pred.neighbors[0].ticket_id == nb.ticket_id


# ---------------------------------------------------------------------------
# NeighborRetriever.retrieve_and_predict — fake embedding model + store
# ---------------------------------------------------------------------------

class _FakeEmbeddingModel:
    """Returns a fixed 4-dim zero vector for any input."""
    def encode_queries(self, texts: list[str], **kwargs) -> np.ndarray:
        return np.zeros((len(texts), 4), dtype=np.float32)


class _FakeTicketStore:
    """Returns a fixed list of fake LanceDB rows when searched."""
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def search(self, query_vector, top_k: int = 5) -> list[dict]:
        return self._rows[:top_k]


def _make_fake_lancedb_rows(n: int) -> list[dict]:
    rows = []
    queues = ["Technical Support", "Billing and Payments"]
    for i in range(n):
        rows.append({
            "_distance":        float(i) * 0.1,
            "ticket_id":        f"ref_{i:04d}",
            "actual_queue":     queues[i % 2],
            "actual_priority":  "high" if i % 2 == 0 else "medium",
            "actual_type":      "Incident",
            "actual_tags_json": json.dumps(["tag_a", "tag_b"]),
            "proxy_topic":      "Technical / Online Access" if i % 2 == 0 else "Billing / Payment",
            "proxy_urgency":    "high" if i % 2 == 0 else "medium",
            "proxy_next_action": "forward_to_technical_support",
            "text_snippet":     f"Ticket {i} snippet.",
        })
    return rows


def test_retrieve_and_predict_returns_prediction():
    model = _FakeEmbeddingModel()
    store = _FakeTicketStore(_make_fake_lancedb_rows(6))
    retriever = NeighborRetriever(model, store, top_k=5)

    pred = retriever.retrieve_and_predict("eval_001", "Subject: test\n\nBody: test body")

    assert pred.predicted_queue is not None
    assert 0.0 <= pred.queue_confidence <= 1.0
    assert len(pred.neighbors) == 5


def test_retrieve_and_predict_empty_store():
    model = _FakeEmbeddingModel()
    store = _FakeTicketStore([])
    retriever = NeighborRetriever(model, store, top_k=5)

    pred = retriever.retrieve_and_predict("eval_001", "some text")

    assert pred.predicted_queue is None
    assert pred.queue_confidence == pytest.approx(0.0)
    assert pred.predicted_tags == []
