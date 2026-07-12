"""
Tests for src/domain/mapping.py — proxy label mapping.

Covers the Claims / Damage proxy mapping added in the sprint:
  - Returns and Exchanges queue maps to Claims / Damage when no stronger signal overrides.
  - Return / Exchange / Replacement tags map to Claims / Damage.
  - Billing tags still override and map to Billing / Payment.
  - Technical tags still override and map to Technical / Online Access.
  - Service Outages and Technical Support with Incident tag do not map to Claims / Damage.
  - Unknown queue / empty tags fall back to Other.
"""
import pytest

from src.domain.mapping import map_proxy_topic


# ─── Claims / Damage via queue mapping ────────────────────────────────────────

def test_returns_and_exchanges_no_tags_maps_to_claims():
    """Returns and Exchanges with no tags falls back to the queue mapping -> Claims / Damage."""
    topic, source = map_proxy_topic("Returns and Exchanges", [])
    assert topic == "Claims / Damage"
    assert source == "queue_mapping"


# ─── Claims / Damage via strong tag signals ───────────────────────────────────

def test_return_tag_maps_to_claims():
    topic, source = map_proxy_topic("Returns and Exchanges", ["Return"])
    assert topic == "Claims / Damage"
    assert source == "strong_tag_signal"


def test_exchange_tag_maps_to_claims():
    topic, source = map_proxy_topic("Returns and Exchanges", ["Exchange"])
    assert topic == "Claims / Damage"
    assert source == "strong_tag_signal"


def test_replacement_tag_maps_to_claims():
    topic, source = map_proxy_topic("Customer Service", ["Replacement"])
    assert topic == "Claims / Damage"
    assert source == "strong_tag_signal"


# ─── Billing tags override Returns and Exchanges ───────────────────────────────

def test_billing_tag_overrides_returns_queue():
    """A Billing tag on a Returns/Exchanges ticket overrides to Billing / Payment."""
    topic, source = map_proxy_topic("Returns and Exchanges", ["Refund"])
    assert topic == "Billing / Payment"
    assert source == "strong_tag_signal"


# ─── Technical tags still map correctly ───────────────────────────────────────

def test_technical_tags_map_to_technical():
    """Login tag maps to Technical / Online Access regardless of queue."""
    topic, source = map_proxy_topic("Returns and Exchanges", ["Login"])
    assert topic == "Technical / Online Access"
    assert source == "strong_tag_signal"


# ─── Incident tag does NOT map to Claims / Damage ─────────────────────────────

def test_service_outages_with_incident_tag_maps_to_technical():
    """
    Incident is only a weak signal for Claims / Damage — it does not appear
    in strong_signals, so Service Outages and Maintenance falls back to its
    queue mapping: Technical / Online Access.
    """
    topic, source = map_proxy_topic("Service Outages and Maintenance", ["Incident"])
    assert topic == "Technical / Online Access"
    assert source == "queue_mapping"


def test_technical_support_with_incident_tag_maps_to_technical():
    """Technical Support + Incident tag -> Technical / Online Access via queue mapping."""
    topic, source = map_proxy_topic("Technical Support", ["Incident"])
    assert topic == "Technical / Online Access"
    assert source == "queue_mapping"


# ─── Fallback to Other ────────────────────────────────────────────────────────

def test_unknown_queue_no_tags_maps_to_other():
    topic, source = map_proxy_topic("Unknown Queue", [])
    assert topic == "Other"
    assert source == "fallback_other"
