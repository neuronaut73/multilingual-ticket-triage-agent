"""
Sprint 6B — Evaluation Metrics and Run Tracking.
Sprint 6C.5 (reviewer) — Reviewer invocation and change-rate metrics.

All functions operate on list[dict] result rows.
Rows come from BatchRunner.process_tickets output.

Prediction-time columns compared against proxy/actual references:
  urgency         vs  proxy_urgency      (proxy-strong: same 3-level scale)
  topic           vs  proxy_topic        (proxy: via queue→topic mapping)
  next_action     vs  proxy_next_action  (proxy agreement, no ground truth)

Timing:
  compute_timing_metrics aggregates per-ticket total_ticket_seconds
  collected by BatchRunner into run-level statistics.

Reviewer metrics (compute_reviewer_metrics):
  reviewer_invocation_rate   — fraction of tickets where reviewer was called
  reviewer_topic_change_rate — fraction of reviewer calls that changed the topic
                               (denominator: reviewer-invoked tickets only)
  reviewer_urgency_change_rate — fraction of reviewer calls that changed urgency
  avg_reviewer_seconds       — mean reviewer wall-clock time across ALL tickets

Data leakage note:
  proxy_* and actual_* values are never used as prediction input.
  They appear here only as evaluation references.
  reviewer_changed_topic / reviewer_changed_urgency compare the reviewer's
  output against the first analysis — no evaluation labels involved.
"""

import statistics
from collections import defaultdict


def accuracy(rows: list[dict], predicted_col: str, actual_col: str) -> float:
    """
    Fraction of rows where predicted == actual (exact match, stripped).
    Rows where actual is empty or None are skipped.
    Returns 0.0 if no valid rows exist.
    """
    correct = 0
    total = 0
    for row in rows:
        raw_actual = row.get(actual_col)
        if raw_actual is None or str(raw_actual).strip() == "":
            continue
        actual    = str(raw_actual).strip()
        predicted = str(row.get(predicted_col, "")).strip()
        if predicted == actual:
            correct += 1
        total += 1
    return round(correct / total, 4) if total > 0 else 0.0


def macro_precision_recall_f1(
    rows: list[dict],
    predicted_col: str,
    actual_col: str,
) -> dict[str, float]:
    """
    Macro-averaged precision, recall, and F1 across all labels present in actual_col.

    Macro-average: compute per-label metrics, then take the unweighted mean.
    Rows where actual is empty are skipped.
    Returns zeros if no valid rows exist.
    """
    valid = [
        r for r in rows
        if r.get(actual_col) is not None and str(r.get(actual_col, "")).strip()
    ]
    if not valid:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    labels = sorted({str(r[actual_col]).strip() for r in valid})

    per_class: list[tuple[float, float, float]] = []
    for label in labels:
        tp = sum(
            1 for r in valid
            if str(r.get(predicted_col, "")).strip() == label
            and str(r[actual_col]).strip() == label
        )
        fp = sum(
            1 for r in valid
            if str(r.get(predicted_col, "")).strip() == label
            and str(r[actual_col]).strip() != label
        )
        fn = sum(
            1 for r in valid
            if str(r.get(predicted_col, "")).strip() != label
            and str(r[actual_col]).strip() == label
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class.append((precision, recall, f1))

    n = len(per_class)
    macro_p = round(sum(p for p, _, _ in per_class) / n, 4)
    macro_r = round(sum(r for _, r, _ in per_class) / n, 4)
    macro_f = round(sum(f for _, _, f in per_class) / n, 4)

    return {"macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f}


def confusion_counts(
    rows: list[dict],
    predicted_col: str,
    actual_col: str,
) -> list[dict]:
    """
    Return confusion matrix as a list of dicts:
      { "actual_label": str, "predicted_label": str, "count": int }

    Rows where actual is empty are skipped.
    Sorted by (actual_label, predicted_label) for deterministic output.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        raw_actual = row.get(actual_col)
        if raw_actual is None or str(raw_actual).strip() == "":
            continue
        actual = str(raw_actual).strip()
        predicted = str(row.get(predicted_col, "")).strip()
        counts[(actual, predicted)] += 1

    return [
        {"actual_label": actual, "predicted_label": predicted, "count": count}
        for (actual, predicted), count in sorted(counts.items())
    ]


def _rate(rows: list[dict], col: str, value: str) -> float:
    """Fraction of rows where col == value."""
    total = len(rows)
    if total == 0:
        return 0.0
    count = sum(1 for r in rows if str(r.get(col, "")).strip() == value)
    return round(count / total, 4)


def compute_evaluation_metrics(rows: list[dict]) -> dict[str, float]:
    """
    Compute all evaluation KPIs from batch result rows.

    Returns a flat dict of metric_name -> float.

    Urgency:    predicted urgency vs proxy_urgency (proxy-strong signal).
    Topic:      predicted topic vs proxy_topic (proxy via queue mapping).
    Next action: agreement with proxy_next_action (named 'agreement', not 'accuracy').
    Operational: rates computed from predicted output columns only.
    """
    total = len(rows)
    if total == 0:
        return {}

    urgency_prf = macro_precision_recall_f1(rows, "urgency", "proxy_urgency")
    topic_prf   = macro_precision_recall_f1(rows, "topic",   "proxy_topic")

    human_review_count = sum(1 for r in rows if r.get("requires_human_review"))
    missing_info_count = sum(1 for r in rows if r.get("missing_info"))

    confidences = [
        float(r["confidence"])
        for r in rows
        if "confidence" in r and r["confidence"] is not None
    ]
    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    reviewer = compute_reviewer_metrics(rows)

    return {
        # Urgency
        "urgency_accuracy":            accuracy(rows, "urgency", "proxy_urgency"),
        "urgency_macro_precision":     urgency_prf["macro_precision"],
        "urgency_macro_recall":        urgency_prf["macro_recall"],
        "urgency_macro_f1":            urgency_prf["macro_f1"],
        # Topic (proxy-based — queue → topic mapping)
        "topic_proxy_accuracy":        accuracy(rows, "topic", "proxy_topic"),
        "topic_macro_precision":       topic_prf["macro_precision"],
        "topic_macro_recall":          topic_prf["macro_recall"],
        "topic_macro_f1":              topic_prf["macro_f1"],
        # Next action (proxy agreement — no ground truth, use 'agreement' not 'accuracy')
        "next_action_proxy_agreement": accuracy(rows, "next_action", "proxy_next_action"),
        # Operational rates
        "human_review_rate":           round(human_review_count / total, 4),
        "missing_info_rate":           round(missing_info_count  / total, 4),
        "average_confidence":          avg_confidence,
        "escalation_rate":             _rate(rows, "next_action", "escalate_to_human_supervisor"),
        "faq_rate":                    _rate(rows, "next_action", "send_standard_faq_or_self_service_link"),
        "billing_forward_rate":        _rate(rows, "next_action", "forward_to_billing_team"),
        "technical_forward_rate":      _rate(rows, "next_action", "forward_to_technical_support"),
        "claim_action_rate":           _rate(rows, "next_action", "create_or_update_claim"),
        # Reviewer operational metrics
        **reviewer,
    }


def compute_reviewer_metrics(rows: list[dict]) -> dict[str, float]:
    """
    Compute reviewer operational metrics from batch result rows.

    reviewer_invocation_rate:
        Fraction of all tickets where the reviewer LLM was called.

    reviewer_topic_change_rate:
        Among reviewer-invoked tickets, fraction where the topic was changed.
        0.0 when reviewer was never invoked.

    reviewer_urgency_change_rate:
        Among reviewer-invoked tickets, fraction where the urgency was changed.
        0.0 when reviewer was never invoked.

    avg_reviewer_seconds:
        Mean reviewer wall-clock seconds across ALL tickets (including 0.0 for
        tickets where the reviewer was not invoked).  Shows pipeline overhead.

    Data leakage note:
        reviewer_changed_topic and reviewer_changed_urgency compare the
        reviewer's output against the first analysis.  No evaluation labels
        are used here.
    """
    total = len(rows)
    if total == 0:
        return {
            "reviewer_invocation_rate":    0.0,
            "reviewer_topic_change_rate":  0.0,
            "reviewer_urgency_change_rate": 0.0,
            "avg_reviewer_seconds":        0.0,
        }

    reviewer_rows = [r for r in rows if r.get("reviewer_used")]
    n_reviewed = len(reviewer_rows)

    invocation_rate = round(n_reviewed / total, 4)

    topic_change_rate = 0.0
    urgency_change_rate = 0.0
    if n_reviewed > 0:
        topic_changes   = sum(1 for r in reviewer_rows if r.get("reviewer_changed_topic"))
        urgency_changes = sum(1 for r in reviewer_rows if r.get("reviewer_changed_urgency"))
        topic_change_rate   = round(topic_changes   / n_reviewed, 4)
        urgency_change_rate = round(urgency_changes / n_reviewed, 4)

    all_reviewer_seconds = [float(r.get("reviewer_seconds", 0.0)) for r in rows]
    avg_reviewer_seconds = round(sum(all_reviewer_seconds) / total, 4)

    return {
        "reviewer_invocation_rate":    invocation_rate,
        "reviewer_topic_change_rate":  topic_change_rate,
        "reviewer_urgency_change_rate": urgency_change_rate,
        "avg_reviewer_seconds":        avg_reviewer_seconds,
    }


def compute_timing_metrics(ticket_seconds: list[float]) -> dict[str, float]:
    """
    Compute aggregate timing statistics from per-ticket total_ticket_seconds.

    Returns a dict with run-level timing KPIs:
      avg_seconds_per_ticket  — arithmetic mean
      p50_seconds_per_ticket  — median (50th percentile)
      p95_seconds_per_ticket  — 95th percentile (nearest-rank method)
      min_seconds_per_ticket  — fastest ticket
      max_seconds_per_ticket  — slowest ticket

    Returns an empty dict when ticket_seconds is empty.
    """
    if not ticket_seconds:
        return {}

    n = len(ticket_seconds)
    sorted_s = sorted(ticket_seconds)

    # Nearest-rank p95: index = ceil(0.95 * n) - 1, clamped to valid range.
    p95_idx = min(max(int(0.95 * n + 0.5) - 1, 0), n - 1)

    return {
        "avg_seconds_per_ticket": round(sum(ticket_seconds) / n, 3),
        "p50_seconds_per_ticket": round(statistics.median(ticket_seconds), 3),
        "p95_seconds_per_ticket": round(sorted_s[p95_idx], 3),
        "min_seconds_per_ticket": round(sorted_s[0], 3),
        "max_seconds_per_ticket": round(sorted_s[-1], 3),
    }
