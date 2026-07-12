"""
Deterministic proxy label mapping for historical Kaggle ticket metadata.

Converts historical Kaggle labels (queue, priority, tags) into the assignment
triage schema (proxy_topic, proxy_urgency, proxy_next_action).

These proxy labels are used for evaluation only — never for routing new tickets.

The strong-signal frozensets are built once at import time from the curated
ontology YAML so the data lives in one place.
"""
from src.domain.ontology import (
    get_historical_tag_hints,
    get_priority_to_urgency_mapping,
    get_queue_to_topic_mapping,
    get_topic_to_default_next_action_mapping,
)

# Priority order for strong-tag-signal checks.
# When a ticket's tags match multiple topics, the first match wins.
_TOPIC_CHECK_ORDER = [
    "Billing / Payment",
    "Technical / Online Access",
    "Policy / Contract",
    "Claims / Damage",
]

# Build lowercase frozensets from the ontology YAML at import time.
# Only strong_signals are used here; weak signals are informational.
_hints = get_historical_tag_hints()
_STRONG_SIGNALS: dict[str, frozenset[str]] = {
    topic: frozenset(s.lower().strip() for s in data.get("strong_signals", []))
    for topic, data in _hints.items()
}


def map_proxy_urgency(priority: str | None) -> str | None:
    """
    Map a Kaggle priority string to the assignment urgency scale.

    Returns None when the input is missing or not in the mapping.
    The mapping is direct: low -> Low, medium -> Medium, high -> High.
    """
    if not priority:
        return None
    mapping = get_priority_to_urgency_mapping()
    return mapping.get(priority.lower().strip())


def map_proxy_topic(queue: str, tags: list[str]) -> tuple[str, str]:
    """
    Map Kaggle queue + tags to an assignment topic and record the source.

    Logic in priority order:
    1. Check strong tag signals (case-insensitive).  → source: strong_tag_signal
    2. Fall back to queue_to_topic mapping.           → source: queue_mapping
    3. Return "Other" if no mapping found.            → source: fallback_other

    Returns (topic, source) where source is one of:
      strong_tag_signal | queue_mapping | fallback_other
    """
    lower_tags = {t.lower().strip() for t in tags if isinstance(t, str) and t.strip()}

    for topic in _TOPIC_CHECK_ORDER:
        strong = _STRONG_SIGNALS.get(topic, frozenset())
        if lower_tags & strong:
            return topic, "strong_tag_signal"

    queue_map = get_queue_to_topic_mapping()
    mapped_topic = queue_map.get(queue, "")
    if mapped_topic:
        return mapped_topic, "queue_mapping"

    return "Other", "fallback_other"


def map_proxy_next_action(proxy_topic: str) -> str:
    """
    Map an assignment topic to its default next_action.

    Falls back to ask_for_more_information if the topic is not found.
    """
    mapping = get_topic_to_default_next_action_mapping()
    return mapping.get(proxy_topic, "ask_for_more_information")
