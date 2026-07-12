"""
Tests for src/infrastructure/lancedb_ticket_store.py

Strategy:
  - Use a temporary directory for each test so tests are isolated.
  - Use tiny 4-dimensional fake vectors — no real model needed.
  - All tests are fast.
"""

import tempfile

import numpy as np
import pytest

from src.infrastructure.lancedb_ticket_store import LanceDBTicketStore


def _make_fake_rows(n: int, dim: int = 4) -> list[dict]:
    """Build n fake ticket rows with random float32 vectors of dimension dim."""
    rows = []
    rng = np.random.default_rng(seed=0)
    for i in range(n):
        rows.append({
            "vector":              rng.random(dim, dtype=np.float32).tolist(),
            "ticket_id":           f"tid_{i:04d}",
            "split_name":          "reference",
            "representation_text": f"Subject: Ticket {i}\n\nBody: This is ticket number {i}.",
            "text_snippet":        f"Ticket {i}",
            "actual_queue":        "Technical Support" if i % 2 == 0 else "Billing and Payments",
            "actual_priority":     "high" if i % 3 == 0 else "medium",
            "actual_type":         "Incident",
            "actual_tags_json":    "[]",
            "proxy_topic":         "Technical / Online Access" if i % 2 == 0 else "Billing / Payment",
            "proxy_urgency":       "high" if i % 3 == 0 else "medium",
            "proxy_next_action":   "forward_to_technical_support",
            "proxy_topic_source":  "queue_mapping",
        })
    return rows


# ---------------------------------------------------------------------------
# recreate_table + count
# ---------------------------------------------------------------------------

def test_recreate_table_and_count():
    with tempfile.TemporaryDirectory() as tmp:
        store = LanceDBTicketStore(path=tmp, table_name="tickets")
        rows = _make_fake_rows(10)
        store.recreate_table(rows)
        assert store.count() == 10


def test_recreate_table_is_idempotent():
    """Calling recreate_table twice should result in the latest row count."""
    with tempfile.TemporaryDirectory() as tmp:
        store = LanceDBTicketStore(path=tmp, table_name="tickets")
        store.recreate_table(_make_fake_rows(5))
        store.recreate_table(_make_fake_rows(8))
        assert store.count() == 8


def test_recreate_table_raises_on_empty_rows():
    with tempfile.TemporaryDirectory() as tmp:
        store = LanceDBTicketStore(path=tmp, table_name="tickets")
        with pytest.raises(ValueError, match="empty"):
            store.recreate_table([])


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_returns_top_k():
    with tempfile.TemporaryDirectory() as tmp:
        store = LanceDBTicketStore(path=tmp, table_name="tickets")
        rows = _make_fake_rows(20)
        store.recreate_table(rows)

        query = [0.5, 0.5, 0.5, 0.5]
        results = store.search(query, top_k=5)
        assert len(results) == 5


def test_search_returns_fewer_than_top_k_when_table_is_small():
    with tempfile.TemporaryDirectory() as tmp:
        store = LanceDBTicketStore(path=tmp, table_name="tickets")
        rows = _make_fake_rows(3)
        store.recreate_table(rows)

        query = [0.1, 0.2, 0.3, 0.4]
        results = store.search(query, top_k=10)
        assert len(results) == 3


def test_search_result_has_expected_keys():
    with tempfile.TemporaryDirectory() as tmp:
        store = LanceDBTicketStore(path=tmp, table_name="tickets")
        store.recreate_table(_make_fake_rows(5))

        results = store.search([0.1, 0.2, 0.3, 0.4], top_k=1)
        assert len(results) == 1

        row = results[0]
        assert "ticket_id"    in row
        assert "actual_queue" in row
        assert "proxy_topic"  in row
        assert "_distance"    in row


def test_search_before_table_creation_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        store = LanceDBTicketStore(path=tmp, table_name="tickets")
        results = store.search([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert results == []


def test_count_before_table_creation_returns_zero():
    with tempfile.TemporaryDirectory() as tmp:
        store = LanceDBTicketStore(path=tmp, table_name="tickets")
        assert store.count() == 0


# ---------------------------------------------------------------------------
# persistence — open the same path again
# ---------------------------------------------------------------------------

def test_store_persists_across_instances():
    """A new LanceDBTicketStore instance pointing at the same path sees the data."""
    with tempfile.TemporaryDirectory() as tmp:
        store1 = LanceDBTicketStore(path=tmp, table_name="tickets")
        store1.recreate_table(_make_fake_rows(7))

        store2 = LanceDBTicketStore(path=tmp, table_name="tickets")
        assert store2.count() == 7
