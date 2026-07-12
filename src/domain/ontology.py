"""
Loads ontology/ticket_ontology.yaml and exposes the assignment triage schema,
deterministic proxy mappings, and historical tag hints.

The YAML is loaded once at module import time. All functions return plain
Python dicts or lists — no custom classes needed.
"""
from pathlib import Path

import yaml

_ONTOLOGY_PATH = Path(__file__).parent.parent.parent / "ontology" / "ticket_ontology.yaml"


def _load_ontology() -> dict:
    with open(_ONTOLOGY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_ontology: dict = _load_ontology()


# ─── Assignment triage schema ─────────────────────────────────────────────────

def get_allowed_topics() -> list[str]:
    """Return the list of valid assignment topic values."""
    return _ontology["assignment_triage_fields"]["topic"]["allowed_values"]


def get_allowed_urgencies() -> list[str]:
    """Return the list of valid assignment urgency values."""
    return _ontology["assignment_triage_fields"]["urgency"]["allowed_values"]


def get_allowed_next_actions() -> list[str]:
    """Return the list of valid assignment next_action values."""
    return _ontology["assignment_triage_fields"]["next_action"]["allowed_values"]


# ─── Deterministic proxy mappings ─────────────────────────────────────────────

def get_queue_to_topic_mapping() -> dict[str, str]:
    """Return the Kaggle queue -> assignment topic mapping."""
    return _ontology["mapping"]["queue_to_topic"]


def get_priority_to_urgency_mapping() -> dict[str, str]:
    """Return the Kaggle priority -> assignment urgency mapping."""
    return _ontology["mapping"]["priority_to_urgency"]


def get_topic_to_default_next_action_mapping() -> dict[str, str]:
    """Return the topic -> default next_action mapping."""
    return _ontology["mapping"]["topic_to_default_next_action"]


# ─── Hint structures ──────────────────────────────────────────────────────────

def get_assignment_topic_hints() -> dict:
    """
    Return assignment topic descriptions and example signals.
    Used for LLM prompt context.
    """
    return _ontology["assignment_topic_hints"]


def get_historical_tag_hints() -> dict:
    """
    Return strong and weak tag signals per assignment topic.
    Mined from the Kaggle historical label space.
    """
    return _ontology["historical_tag_to_assignment_topic_hints"]
