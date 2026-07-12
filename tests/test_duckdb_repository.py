"""
Tests for Sprint 2 additions:
  - src/domain/mapping.py  (proxy label mapping logic)
  - src/infrastructure/duckdb_repository.py  (storage of actual_* and proxy_* fields)

Verifies that:
- answer is excluded from source_row_json.
- proxy_urgency maps high / medium / low correctly.
- strong tag signal overrides queue mapping.
- queue mapping is used when no strong tag signal exists.
- fallback_other is used when no mapping exists.
- inserted rows contain actual_* and proxy_* fields with correct values.
- deterministic split is reproducible across calls.
"""
import json

import pytest

from src.domain.mapping import (
    map_proxy_next_action,
    map_proxy_topic,
    map_proxy_urgency,
)
from src.infrastructure.duckdb_repository import (
    DuckDBRepository,
    _assign_split,
    _deterministic_ticket_id,
)


# ─── map_proxy_urgency ────────────────────────────────────────────────────────

def test_proxy_urgency_maps_high():
    assert map_proxy_urgency("high") == "High"


def test_proxy_urgency_maps_medium():
    assert map_proxy_urgency("medium") == "Medium"


def test_proxy_urgency_maps_low():
    assert map_proxy_urgency("low") == "Low"


def test_proxy_urgency_returns_none_for_unknown():
    assert map_proxy_urgency("critical") is None


def test_proxy_urgency_returns_none_for_empty():
    assert map_proxy_urgency("") is None


def test_proxy_urgency_is_case_insensitive():
    assert map_proxy_urgency("HIGH") == "High"
    assert map_proxy_urgency("Medium") == "Medium"


# ─── map_proxy_topic — strong tag signal ─────────────────────────────────────

def test_strong_tag_signal_billing_overrides_queue():
    # Queue would map to Technical, but billing tag wins.
    topic, source = map_proxy_topic("IT Support", ["Billing", "Invoice"])
    assert topic == "Billing / Payment"
    assert source == "strong_tag_signal"


def test_strong_tag_signal_technical():
    topic, source = map_proxy_topic("Customer Service", ["Login", "Password"])
    assert topic == "Technical / Online Access"
    assert source == "strong_tag_signal"


def test_strong_tag_signal_policy():
    topic, source = map_proxy_topic("General Inquiry", ["Policy", "Contract"])
    assert topic == "Policy / Contract"
    assert source == "strong_tag_signal"


def test_strong_tag_signal_claims():
    topic, source = map_proxy_topic("General Inquiry", ["Claim", "Damage"])
    assert topic == "Claims / Damage"
    assert source == "strong_tag_signal"


# ─── map_proxy_topic — queue mapping fallback ─────────────────────────────────

def test_queue_mapping_used_when_no_strong_tag():
    topic, source = map_proxy_topic("Billing and Payments", ["Feedback"])
    assert topic == "Billing / Payment"
    assert source == "queue_mapping"


def test_queue_mapping_technical_support():
    topic, source = map_proxy_topic("Technical Support", [])
    assert topic == "Technical / Online Access"
    assert source == "queue_mapping"


# ─── map_proxy_topic — fallback_other ────────────────────────────────────────

def test_fallback_other_when_queue_unknown_and_no_tags():
    topic, source = map_proxy_topic("Unknown Queue XYZ", [])
    assert topic == "Other"
    assert source == "fallback_other"


def test_fallback_other_when_empty_queue_and_no_tags():
    topic, source = map_proxy_topic("", [])
    assert topic == "Other"
    assert source == "fallback_other"


# ─── map_proxy_next_action ────────────────────────────────────────────────────

def test_next_action_for_billing():
    assert map_proxy_next_action("Billing / Payment") == "forward_to_billing_team"


def test_next_action_for_technical():
    assert map_proxy_next_action("Technical / Online Access") == "forward_to_technical_support"


def test_next_action_for_other_falls_back():
    result = map_proxy_next_action("Nonexistent Topic")
    assert result == "ask_for_more_information"


# ─── Deterministic split ──────────────────────────────────────────────────────

def test_split_is_reproducible():
    tid = _deterministic_ticket_id("subject", "body", 0)
    split_a = _assign_split(tid, 0.2, 42)
    split_b = _assign_split(tid, 0.2, 42)
    assert split_a == split_b


def test_split_values_are_valid():
    for i in range(20):
        tid = _deterministic_ticket_id(f"subject {i}", f"body {i}", i)
        split = _assign_split(tid, 0.2, 42)
        assert split in {"reference", "eval"}


def test_split_fraction_is_approximately_correct():
    evals = sum(
        1
        for i in range(1000)
        if _assign_split(
            _deterministic_ticket_id(f"s{i}", f"b{i}", i), 0.2, 42
        ) == "eval"
    )
    # Expect roughly 200 ± 50 eval rows out of 1000
    assert 150 <= evals <= 250


# ─── DuckDB storage of actual_* and proxy_* fields ───────────────────────────

@pytest.fixture
def in_memory_repo():
    repo = DuckDBRepository(":memory:")
    yield repo
    repo.close()


def _make_row(
    row_index=0,
    subject="Login fails",
    body="Cannot access portal",
    queue="IT Support",
    priority="high",
    tags=None,
) -> dict:
    if tags is None:
        tags = ["Bug", "Access"]

    from src.application.preprocessing import (
        build_raw_text,
        build_representation_text,
        make_text_snippet,
        normalize_text,
    )
    from src.domain.mapping import (
        map_proxy_next_action,
        map_proxy_topic,
        map_proxy_urgency,
    )

    raw_text = build_raw_text(subject, body)
    proxy_topic, proxy_topic_source = map_proxy_topic(queue, tags)
    source_row = {"subject": subject, "body": body, "queue": queue, "priority": priority}

    return {
        "_row_index":          row_index,
        "subject":             subject,
        "body":                body,
        "raw_text":            raw_text,
        "cleaned_text":        normalize_text(raw_text),
        "representation_text": build_representation_text(subject, body),
        "text_snippet":        make_text_snippet(raw_text),
        "actual_queue":        queue,
        "actual_priority":     priority,
        "actual_type":         "Incident",
        "actual_tags_json":    json.dumps(tags),
        "language":            "en",
        "proxy_topic":         proxy_topic,
        "proxy_urgency":       map_proxy_urgency(priority),
        "proxy_next_action":   map_proxy_next_action(proxy_topic),
        "proxy_topic_source":  proxy_topic_source,
        "source_row_json":     json.dumps(source_row),
    }


def test_inserted_row_has_actual_fields(in_memory_repo):
    in_memory_repo.insert_tickets([_make_row()])
    rows = in_memory_repo.conn.execute(
        "SELECT actual_queue, actual_priority, actual_type, actual_tags_json"
        " FROM historical_tickets"
    ).fetchall()
    assert len(rows) == 1
    actual_queue, actual_priority, actual_type, actual_tags_json = rows[0]
    assert actual_queue    == "IT Support"
    assert actual_priority == "high"
    assert actual_type     == "Incident"
    assert json.loads(actual_tags_json) == ["Bug", "Access"]


def test_inserted_row_has_proxy_fields(in_memory_repo):
    in_memory_repo.insert_tickets([_make_row()])
    rows = in_memory_repo.conn.execute(
        "SELECT proxy_topic, proxy_urgency, proxy_next_action, proxy_topic_source"
        " FROM historical_tickets"
    ).fetchall()
    assert len(rows) == 1
    proxy_topic, proxy_urgency, proxy_next_action, proxy_topic_source = rows[0]
    assert proxy_topic         == "Technical / Online Access"
    assert proxy_urgency       == "High"
    assert proxy_next_action   == "forward_to_technical_support"
    assert proxy_topic_source  == "strong_tag_signal"


def test_source_row_json_excludes_answer(in_memory_repo):
    row = _make_row()
    # Confirm answer is not present even if someone accidentally passes it
    src = json.loads(row["source_row_json"])
    assert "answer" not in src


def test_inserted_row_source_row_json_does_not_contain_answer(in_memory_repo):
    in_memory_repo.insert_tickets([_make_row()])
    stored = in_memory_repo.conn.execute(
        "SELECT source_row_json FROM historical_tickets"
    ).fetchone()[0]
    data = json.loads(stored)
    assert "answer" not in data


def test_count_by_split_sums_to_total(in_memory_repo):
    rows = [_make_row(row_index=i, subject=f"s{i}", body=f"b{i}") for i in range(20)]
    in_memory_repo.insert_tickets(rows)
    counts = in_memory_repo.count_by_split()
    assert sum(counts.values()) == 20
    assert set(counts.keys()).issubset({"reference", "eval"})


def test_count_by_proxy_topic_returns_dict(in_memory_repo):
    in_memory_repo.insert_tickets([_make_row()])
    result = in_memory_repo.count_by_proxy_topic()
    assert isinstance(result, dict)
    assert len(result) >= 1
