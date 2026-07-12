"""
Sprint 4 — Weighted voting helpers for neighbor-based prediction.

Three small pure functions:
  distance_to_similarity  — converts LanceDB _distance to a [0, 1] similarity score
  weighted_vote           — picks the label with the highest summed similarity weight
  aggregate_tags          — returns the top-N tags ranked by total similarity weight
"""

from __future__ import annotations

import json


def distance_to_similarity(distance: float) -> float:
    """
    Convert a LanceDB L2 distance to a similarity score in (0, 1].

    Formula: similarity = 1 / (1 + distance)

    At distance = 0 (identical vector) -> similarity = 1.0.
    As distance grows -> similarity approaches 0.
    This is a monotone proxy: closer neighbors get higher weight.
    """
    return 1.0 / (1.0 + distance)


def weighted_vote(
    label_weight_pairs: list[tuple[str | None, float]],
) -> tuple[str | None, float]:
    """
    Select the label with the highest total similarity weight.

    Parameters
    ----------
    label_weight_pairs:
        List of (label, weight) tuples. label may be None or empty string.

    Returns
    -------
    (best_label, confidence)
        best_label   — the label with the highest summed weight, or None if no
                       valid labels exist.
        confidence   — winning_weight / total_weight, or 0.0 on edge cases.

    Safe defaults:
        - Empty input          -> (None, 0.0)
        - All None/empty labels -> (None, 0.0)
        - Zero total weight    -> (None, 0.0)
    """
    scores: dict[str, float] = {}

    for label, weight in label_weight_pairs:
        if label is None or str(label).strip() == "":
            continue
        scores[label] = scores.get(label, 0.0) + weight

    if not scores:
        return None, 0.0

    total_weight = sum(scores.values())
    if total_weight <= 0.0:
        return None, 0.0

    best_label = max(scores, key=lambda lbl: scores[lbl])
    confidence = scores[best_label] / total_weight
    return best_label, confidence


def aggregate_tags(
    actual_tags_with_weights: list[tuple[list[str], float]],
    top_n: int = 5,
) -> list[str]:
    """
    Aggregate tags from multiple neighbors weighted by similarity.

    Parameters
    ----------
    actual_tags_with_weights:
        List of (tag_list, weight) pairs. tag_list is the parsed tag list for
        one neighbor. weight is that neighbor's similarity score.
    top_n:
        Number of top tags to return.

    Returns
    -------
    List of tag strings, ordered by descending total weight. Empty if no tags.
    """
    tag_scores: dict[str, float] = {}

    for tags, weight in actual_tags_with_weights:
        seen_in_this_neighbor: set[str] = set()
        for tag in tags:
            tag = tag.strip()
            if not tag:
                continue
            # Count each distinct tag at most once per neighbor to avoid
            # double-counting if the same tag appears twice in one row.
            if tag in seen_in_this_neighbor:
                continue
            seen_in_this_neighbor.add(tag)
            tag_scores[tag] = tag_scores.get(tag, 0.0) + weight

    sorted_tags = sorted(tag_scores, key=lambda t: tag_scores[t], reverse=True)
    return sorted_tags[:top_n]


def parse_tags_json(actual_tags_json: str) -> list[str]:
    """
    Safely parse a JSON-encoded tag list.

    Returns an empty list if the value is None, blank, or malformed JSON.
    """
    if not actual_tags_json:
        return []
    try:
        parsed = json.loads(actual_tags_json)
        if isinstance(parsed, list):
            return [str(t) for t in parsed if t]
        return []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
