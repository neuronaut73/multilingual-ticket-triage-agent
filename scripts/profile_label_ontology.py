"""
Sprint 0.5 — Label Profiling and Ontology Discovery

Reads historical ticket metadata columns from the CSV, computes label
distributions and tag association statistics, saves CSV profile outputs,
and writes ontology/ticket_ontology.yaml.

Usage:
    python scripts/profile_label_ontology.py

Inputs:
    data/aa_dataset-tickets-multi-lang-5-2-50-version.csv

Outputs:
    outputs/label_profile/column_profile.csv
    outputs/label_profile/top_values_queue.csv
    outputs/label_profile/top_values_priority.csv
    outputs/label_profile/top_values_type.csv
    outputs/label_profile/global_tag_counts.csv
    outputs/label_profile/top_tags_by_position.csv
    outputs/label_profile/top_tags_by_queue.csv
    outputs/label_profile/top_tags_by_queue_priority_mix.csv
    outputs/label_profile/top_tags_by_queue_position.csv
    outputs/label_profile/p_queue_given_tag.csv
    outputs/label_profile/p_priority_given_tag.csv
    outputs/label_profile/lift_tag_queue.csv
    outputs/label_profile/lift_tag_priority.csv
    outputs/label_profile/proxy_topic_distribution.csv
    outputs/label_profile/proxy_topic_by_queue.csv
    ontology/ticket_ontology.yaml
"""

import os
import pandas as pd
import yaml

# ─── Configuration ────────────────────────────────────────────────────────────

CSV_PATH = "data/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
PROFILE_DIR = "outputs/label_profile"
ONTOLOGY_PATH = "ontology/ticket_ontology.yaml"

# Tags that appear fewer than this many times are excluded from the ontology.
MIN_TAG_COUNT = 5

LABEL_COLS = ["queue", "priority", "type"]
TAG_COLS = [f"tag_{i}" for i in range(1, 9)]


# ─── Assignment output schema ─────────────────────────────────────────────────
# These are the final triage fields required by the HDI assignment.
# They are NOT derived from the Kaggle dataset.

ASSIGNMENT_TOPICS = [
    "Policy / Contract",
    "Claims / Damage",
    "Billing / Payment",
    "Technical / Online Access",
    "Other",
]

ASSIGNMENT_URGENCIES = ["Low", "Medium", "High"]

ASSIGNMENT_NEXT_ACTIONS = [
    "send_standard_faq_or_self_service_link",
    "create_or_update_claim",
    "forward_to_billing_team",
    "forward_to_technical_support",
    "escalate_to_human_supervisor",
    "ask_for_more_information",
]


# ─── Deterministic proxy mappings ─────────────────────────────────────────────
# Used only for evaluation and prompt hints — not for routing new tickets.

PRIORITY_TO_URGENCY = {
    "low":    "Low",
    "medium": "Medium",
    "high":   "High",
}

QUEUE_TO_TOPIC = {
    "Billing and Payments":           "Billing / Payment",
    "Customer Service":               "Other",
    "General Inquiry":                "Other",
    "Human Resources":                "Other",
    "IT Support":                     "Technical / Online Access",
    "Product Support":                "Policy / Contract",
    "Returns and Exchanges":          "Claims / Damage",
    "Sales and Pre-Sales":            "Policy / Contract",
    "Service Outages and Maintenance":"Technical / Online Access",
    "Technical Support":              "Technical / Online Access",
}

TOPIC_TO_DEFAULT_NEXT_ACTION = {
    "Policy / Contract":        "send_standard_faq_or_self_service_link",
    "Claims / Damage":          "create_or_update_claim",
    "Billing / Payment":        "forward_to_billing_team",
    "Technical / Online Access":"forward_to_technical_support",
    "Other":                    "send_standard_faq_or_self_service_link",
}


# ─── Assignment topic hint descriptions ───────────────────────────────────────
# Static, human-authored descriptions for LLM prompt context.

ASSIGNMENT_TOPIC_HINTS = {
    "Policy / Contract": {
        "description": (
            "Policy documents, coverage, contract changes, cancellation, address changes, "
            "subscriptions, plans, warranty, product terms."
        ),
        "example_signals": [
            "policy", "contract", "coverage", "cancellation", "subscription",
            "plan", "warranty", "license", "pricing", "product",
        ],
    },
    "Claims / Damage": {
        "description": (
            "New or existing insurance claim, accident, damage, theft, loss, "
            "repair, reimbursement."
        ),
        "example_signals": [
            "claim", "damage", "accident", "theft", "loss",
            "repair", "reimbursement", "incident", "investigation",
        ],
    },
    "Billing / Payment": {
        "description": (
            "Premium, invoice, payment, refund, duplicate charge, direct debit, "
            "transaction or billing issue."
        ),
        "example_signals": [
            "billing", "payment", "invoice", "refund", "transaction",
            "reconciliation", "financial", "cost", "premium", "charge",
        ],
    },
    "Technical / Online Access": {
        "description": (
            "Login, app, portal, password, online account, authentication, access, "
            "outage or technical issue."
        ),
        "example_signals": [
            "login", "password", "authentication", "access", "portal",
            "app", "online", "IT", "technical", "bug", "outage", "network",
        ],
    },
    "Other": {
        "description": (
            "Unclear, out-of-scope, HR, marketing, sales, generic support or "
            "insufficient information."
        ),
        "example_signals": [
            "HR", "Human Resources", "Marketing", "Sales",
            "Advertising", "Feedback",
        ],
    },
}


# ─── Historical tag-to-topic signal hints ─────────────────────────────────────
# Maps Kaggle tag values to HDI assignment topics.
# Strong signals = high confidence tags.
# Weak signals   = contextual clues only.

HISTORICAL_TAG_TO_TOPIC_HINTS = {
    "Policy / Contract": {
        "strong_signals": [
            "Policy", "Contract", "Subscription", "Plan",
            "Pricing", "Warranty", "Product", "License",
        ],
        "weak_signals": [
            "Feature", "Product Support", "Sales", "Customer Service",
        ],
    },
    "Claims / Damage": {
        "strong_signals": [
            "Claim", "Damage", "Loss", "Accident",
            "Theft", "Repair", "Reimbursement",
            "Return", "Exchange", "Replacement",
        ],
        "weak_signals": [
            "Incident", "Case Study", "Issue", "Investigation",
        ],
    },
    "Billing / Payment": {
        "strong_signals": [
            "Billing", "Payment", "Invoice", "Refund",
            "Transaction", "Reconciliation", "Financial", "Cost",
        ],
        "weak_signals": [
            "Finance", "Adjustment", "Discrepancy",
        ],
    },
    "Technical / Online Access": {
        "strong_signals": [
            "Login", "Password", "Authentication", "Access",
            "Access Control", "Portal", "App", "Online",
            "Bug", "Outage", "Network", "VPN", "Error",
        ],
        "weak_signals": [
            "IT", "Tech Support", "Technical Support",
            "Software", "System", "Platform", "Performance", "Connectivity",
        ],
    },
    "Other": {
        "strong_signals": [
            "HR", "Human Resources", "Marketing", "Sales", "Advertising",
        ],
        "weak_signals": [
            "Feedback", "General Inquiry", "Customer Service",
        ],
    },
}


# ─── Historical type hints ─────────────────────────────────────────────────────

HISTORICAL_TYPE_HINTS = {
    "Incident": {
        "interpretation": (
            "Operational issue or interruption. "
            "Weak contextual signal only; not a hard urgency rule."
        ),
    },
    "Problem": {
        "interpretation": (
            "Underlying or recurring issue. Weak contextual signal only."
        ),
    },
    "Request": {
        "interpretation": (
            "Customer request or information need. Weak contextual signal only."
        ),
    },
    "Change": {
        "interpretation": (
            "Change request. Weak contextual signal only."
        ),
    },
}


# ─── Proxy tag signal sets ─────────────────────────────────────────────────────
# Used inside map_historical_to_assignment_topic() for fast set intersection.
# All values are lowercased for case-insensitive matching.

_BILLING_TAG_SIGNALS = frozenset([
    "billing", "payment", "invoice", "refund",
    "transaction", "reconciliation", "financial", "cost",
])

_TECHNICAL_TAG_SIGNALS = frozenset([
    "login", "password", "authentication", "access", "access control",
    "portal", "app", "online", "bug", "outage", "network", "vpn", "error",
])

_POLICY_TAG_SIGNALS = frozenset([
    "policy", "contract", "subscription", "plan",
    "pricing", "warranty", "product", "license",
])

_CLAIMS_TAG_SIGNALS = frozenset([
    "claim", "damage", "loss", "accident",
    "theft", "repair", "reimbursement",
    "return", "exchange", "replacement",
])


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Read the ticket CSV. Only metadata columns are used — not 'answer'."""
    df = pd.read_csv(path)
    return df


# ─── Column profiling ─────────────────────────────────────────────────────────

def profile_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    For each column compute: total rows, missing count, missing rate,
    and unique value count. Returns one row per column.
    """
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        rows.append({
            "column":        col,
            "total_rows":    len(df),
            "missing_count": int(df[col].isna().sum()),
            "missing_rate":  round(float(df[col].isna().mean()), 4),
            "unique_count":  int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


# ─── Top-value tables ─────────────────────────────────────────────────────────

def top_values(df: pd.DataFrame, col: str, n: int = 20) -> pd.DataFrame:
    """Return the top-n most frequent values with count and percentage."""
    vc = df[col].value_counts(dropna=False).head(n).reset_index()
    vc.columns = [col, "count"]
    vc["pct"] = (vc["count"] / len(df)).round(4)
    return vc


# ─── Tag flattening ───────────────────────────────────────────────────────────

def flatten_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt tag_1 … tag_8 into long format so each row is one (ticket, tag) pair.
    Retains queue and priority for association analysis.
    Returns columns: [ticket_id, queue, priority, tag_position, tag].
    """
    # Ensure we have a ticket_id to de-duplicate on later.
    if "ticket_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "ticket_id"})

    keep = ["ticket_id", "queue", "priority"] + TAG_COLS
    long = df[keep].melt(
        id_vars=["ticket_id", "queue", "priority"],
        value_vars=TAG_COLS,
        var_name="tag_position",
        value_name="tag",
    )
    # Drop empty/null tag slots — not every ticket fills all 8 positions.
    long = long.dropna(subset=["tag"])
    long["tag"] = long["tag"].astype(str).str.strip()
    long = long[long["tag"] != ""]
    return long.reset_index(drop=True)


# ─── Global tag counts ────────────────────────────────────────────────────────

def global_tag_counts(tags_long: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """
    Count how often each tag appears across all positions and tickets.
    Returns tags with count >= min_count.
    """
    counts = tags_long["tag"].value_counts().reset_index()
    counts.columns = ["tag", "count"]
    counts["pct"] = (counts["count"] / len(tags_long)).round(4)
    return counts[counts["count"] >= min_count].reset_index(drop=True)


# ─── Conditional distributions ────────────────────────────────────────────────

def conditional_distribution(tags_long: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Compute P(target_col | tag) as a wide matrix.
    Rows = tags, columns = unique values of target_col.
    Each row sums to 1.0.
    """
    ct = pd.crosstab(tags_long["tag"], tags_long[target_col], normalize="index")
    ct = ct.round(4)
    ct.index.name = "tag"
    ct.columns.name = None
    return ct.reset_index()


# ─── Lift ─────────────────────────────────────────────────────────────────────

def compute_lift(
    df: pd.DataFrame,
    tags_long: pd.DataFrame,
    target_col: str,
    valid_tags: set,
) -> pd.DataFrame:
    """
    Compute lift(tag -> label) = P(label | tag) / P(label).

    lift > 1 means the tag makes that label more likely than the base rate.
    lift < 1 means the tag makes it less likely.

    Parameters
    ----------
    df         : original DataFrame, used for overall base rates.
    tags_long  : melted tag table from flatten_tags().
    target_col : "queue" or "priority".
    valid_tags : tags that meet the minimum frequency threshold.

    Returns a long DataFrame sorted by tag then lift descending.
    """
    # Overall base rate P(label) computed from the full dataset (all tickets).
    base_rates = df[target_col].value_counts(normalize=True)

    # Work only with valid (frequent-enough) tags.
    filtered = tags_long[tags_long["tag"].isin(valid_tags)].copy()

    # Count (tag, label) co-occurrences.
    counts = filtered.groupby(["tag", target_col]).size().reset_index(name="count")

    # Total tag occurrences per tag (denominator for conditional probability).
    tag_totals = filtered.groupby("tag").size().reset_index(name="tag_total")
    counts = counts.merge(tag_totals, on="tag")

    counts["p_label_given_tag"] = (counts["count"] / counts["tag_total"]).round(4)

    # Rename target_col to "label" for a clean join with base_rates.
    counts = counts.rename(columns={target_col: "label"})

    base_df = base_rates.reset_index()
    base_df.columns = ["label", "p_label"]
    base_df["p_label"] = base_df["p_label"].round(4)

    counts = counts.merge(base_df, on="label")
    counts["lift"] = (counts["p_label_given_tag"] / counts["p_label"]).round(3)

    return (
        counts[["tag", "label", "count", "p_label_given_tag", "p_label", "lift"]]
        .sort_values(["tag", "lift"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ─── Position-aware tag profile ───────────────────────────────────────────────

def top_tags_by_position(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    For each tag position (tag_1 … tag_8), show the top-n most frequent values.

    share_within_position = count / (non-null rows for that position).
    missing_rate          = fraction of all rows where the position is null.

    Columns: tag_position, tag, count, share_within_position, missing_rate.
    """
    rows = []
    for col in TAG_COLS:
        if col not in df.columns:
            continue
        missing_rate = round(float(df[col].isna().mean()), 4)
        non_null = df[col].dropna()
        vc = non_null.value_counts().head(n)
        for tag, count in vc.items():
            rows.append({
                "tag_position":          col,
                "tag":                   tag,
                "count":                 int(count),
                "share_within_position": round(float(count / len(non_null)), 4),
                "missing_rate":          missing_rate,
            })
    return pd.DataFrame(rows)


# ─── Queue-centric tag profiles ───────────────────────────────────────────────

def top_tags_by_queue(tags_long: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    For each queue, show the top-n most frequent tags across all positions.

    share_within_queue = count / total tag occurrences for that queue.

    Columns: queue, tag, count, share_within_queue.
    """
    rows = []
    for queue, grp in tags_long.groupby("queue"):
        total = len(grp)
        vc = grp["tag"].value_counts().head(n)
        for tag, count in vc.items():
            rows.append({
                "queue":              queue,
                "tag":                tag,
                "count":              int(count),
                "share_within_queue": round(float(count / total), 4),
            })
    return pd.DataFrame(rows)


def top_tags_by_queue_priority_mix(tags_long: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    For each queue and its top-n tags, show the count plus a priority breakdown.

    Steps:
    1. Find the top-n tags per queue by total count.
    2. Pivot priority into columns (high, medium, low) — fill missing with 0.
    3. Compute share columns: high_share = high_count / count, etc.

    Columns: queue, tag, count,
             high_count, medium_count, low_count,
             high_share, medium_share, low_share.
    """
    # Step 1 — top-n tags per queue.
    totals = (
        tags_long.groupby(["queue", "tag"])
        .size()
        .reset_index(name="count")
    )
    top = (
        totals
        .sort_values(["queue", "count"], ascending=[True, False])
        .groupby("queue")
        .head(n)
        .reset_index(drop=True)
    )

    # Step 2 — priority breakdown for every (queue, tag) pair in the dataset.
    prio_counts = (
        tags_long.groupby(["queue", "tag", "priority"])
        .size()
        .reset_index(name="prio_count")
    )
    prio_wide = prio_counts.pivot_table(
        index=["queue", "tag"],
        columns="priority",
        values="prio_count",
        fill_value=0,
    ).reset_index()
    prio_wide.columns.name = None

    # Merge — keep only the top tags.
    result = top.merge(prio_wide, on=["queue", "tag"], how="left")

    # Step 3 — add share columns, then rename raw counts for clarity.
    for prio in ["high", "medium", "low"]:
        if prio not in result.columns:
            result[prio] = 0
        result[f"{prio}_share"] = (result[prio] / result["count"]).round(4)

    result = result.rename(columns={
        "high":   "high_count",
        "medium": "medium_count",
        "low":    "low_count",
    })

    return result.sort_values(["queue", "count"], ascending=[True, False]).reset_index(drop=True)


def top_tags_by_queue_position(tags_long: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    For each (queue, tag_position), show the top-n most frequent tag values.

    share_within_queue_position = count / total occurrences in that (queue, position).

    Columns: queue, tag_position, tag, count, share_within_queue_position.
    """
    rows = []
    for (queue, pos), grp in tags_long.groupby(["queue", "tag_position"]):
        total = len(grp)
        vc = grp["tag"].value_counts().head(n)
        for tag, count in vc.items():
            rows.append({
                "queue":                       queue,
                "tag_position":                pos,
                "tag":                         tag,
                "count":                       int(count),
                "share_within_queue_position": round(float(count / total), 4),
            })
    return pd.DataFrame(rows)


# ─── Proxy topic mapper ───────────────────────────────────────────────────────

def map_historical_to_assignment_topic(
    queue: str,
    tags: list,
    ticket_type: str,
) -> str:
    """
    Deterministic proxy mapper from historical Kaggle metadata to HDI assignment topics.

    Used only for evaluation and prompt hints — never for routing new tickets.

    Mapping logic (in priority order):
    1. Check strong tag signals (case-insensitive set intersection).
    2. Fall back to QUEUE_TO_TOPIC if no tag signal matched.
    3. Return "Other" if queue is unknown.
    """
    lower_tags = {
        t.lower().strip()
        for t in tags
        if isinstance(t, str) and t.strip()
    }

    if lower_tags & _BILLING_TAG_SIGNALS:
        return "Billing / Payment"
    if lower_tags & _TECHNICAL_TAG_SIGNALS:
        return "Technical / Online Access"
    if lower_tags & _POLICY_TAG_SIGNALS:
        return "Policy / Contract"
    if lower_tags & _CLAIMS_TAG_SIGNALS:
        return "Claims / Damage"

    return QUEUE_TO_TOPIC.get(queue, "Other")


def apply_proxy_topic_mapper(df: pd.DataFrame) -> pd.Series:
    """
    Apply map_historical_to_assignment_topic to every row in df.

    Collects tag_1 … tag_8 values per row, skipping nulls, and calls the mapper.
    Returns a Series of mapped assignment topic strings, aligned with df's index.
    """
    def _map_row(row):
        tags = [
            row[c] for c in TAG_COLS
            if c in row and pd.notna(row[c]) and str(row[c]).strip()
        ]
        queue = row["queue"] if pd.notna(row.get("queue")) else ""
        ticket_type = row["type"] if pd.notna(row.get("type")) else ""
        return map_historical_to_assignment_topic(queue, tags, ticket_type)

    return df.apply(_map_row, axis=1)


# ─── Proxy topic distribution ─────────────────────────────────────────────────

def proxy_topic_distribution(mapped_series: pd.Series) -> pd.DataFrame:
    """
    Count and percentage for each mapped_assignment_topic across all tickets.

    Columns: mapped_assignment_topic, count, pct.
    """
    counts = mapped_series.value_counts().reset_index()
    counts.columns = ["mapped_assignment_topic", "count"]
    counts["pct"] = (counts["count"] / len(mapped_series)).round(4)
    return counts.reset_index(drop=True)


def proxy_topic_by_queue(df: pd.DataFrame, mapped_series: pd.Series) -> pd.DataFrame:
    """
    For each (queue, mapped_assignment_topic), show count and pct_within_queue.

    Columns: queue, mapped_assignment_topic, count, pct_within_queue.
    """
    tmp = df[["queue"]].copy()
    tmp["mapped_assignment_topic"] = mapped_series.values

    group = tmp.groupby(["queue", "mapped_assignment_topic"]).size().reset_index(name="count")
    queue_totals = tmp.groupby("queue").size().reset_index(name="queue_total")
    group = group.merge(queue_totals, on="queue")
    group["pct_within_queue"] = (group["count"] / group["queue_total"]).round(4)
    group = group.drop(columns=["queue_total"])
    return (
        group
        .sort_values(["queue", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ─── Ontology builder ─────────────────────────────────────────────────────────

def build_ontology(df: pd.DataFrame, valid_tags: set, queue_tag_hints: dict) -> dict:
    """
    Build the ontology dict from observed label values and static assignment schema.

    Structure:
    - meta: source info and a note clarifying the Kaggle-vs-assignment distinction.
    - assignment_triage_fields: the final HDI output schema (topic, urgency, next_action).
    - assignment_topic_hints: static descriptions + example signals for LLM prompt context.
    - historical_label_space: observed values from the Kaggle dataset.
    - mapping: deterministic proxy mappings (priority->urgency, queue->topic, topic->next_action).
    - historical_tag_to_assignment_topic_hints: strong/weak tag signals per topic.
    - historical_queue_tag_hints: top-5 tags per Kaggle queue (mined from data).
    - historical_type_hints: weak interpretations of Kaggle type values.
    """
    queues     = sorted(df["queue"].dropna().unique().tolist())
    priorities = sorted(df["priority"].dropna().unique().tolist())
    types      = sorted(df["type"].dropna().unique().tolist())

    ontology = {
        "meta": {
            "source_csv":    CSV_PATH,
            "total_tickets": int(len(df)),
            "min_tag_count": MIN_TAG_COUNT,
            "note": (
                "Kaggle labels (queue, priority, type, tag_1..tag_8) are historical "
                "evidence and proxy evaluation data only. "
                "They are not the final HDI assignment output schema. "
                "The assignment output schema is defined under assignment_triage_fields."
            ),
        },
        "assignment_triage_fields": {
            "topic": {
                "allowed_values": list(ASSIGNMENT_TOPICS),
            },
            "urgency": {
                "allowed_values": list(ASSIGNMENT_URGENCIES),
            },
            "next_action": {
                "allowed_values": list(ASSIGNMENT_NEXT_ACTIONS),
            },
        },
        "assignment_topic_hints": ASSIGNMENT_TOPIC_HINTS,
        "historical_label_space": {
            "queues":     queues,
            "priorities": priorities,
            "types":      types,
            "tags":       sorted(valid_tags),
        },
        "mapping": {
            "priority_to_urgency":          PRIORITY_TO_URGENCY,
            "queue_to_topic":               QUEUE_TO_TOPIC,
            "topic_to_default_next_action": TOPIC_TO_DEFAULT_NEXT_ACTION,
        },
        "historical_tag_to_assignment_topic_hints": HISTORICAL_TAG_TO_TOPIC_HINTS,
        "historical_queue_tag_hints":               queue_tag_hints,
        "historical_type_hints":                    HISTORICAL_TYPE_HINTS,
    }
    return ontology


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PROFILE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ONTOLOGY_PATH), exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {CSV_PATH} ...")
    df = load_data(CSV_PATH)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # ── 1. Column profile ─────────────────────────────────────────────────────
    all_meta_cols = LABEL_COLS + TAG_COLS
    col_profile = profile_columns(df, all_meta_cols)
    col_profile.to_csv(f"{PROFILE_DIR}/column_profile.csv", index=False)
    print("\n=== Column profile ===")
    print(col_profile.to_string(index=False))

    # ── 2. Top values for each label column ───────────────────────────────────
    for col in LABEL_COLS:
        if col not in df.columns:
            continue
        tv = top_values(df, col)
        tv.to_csv(f"{PROFILE_DIR}/top_values_{col}.csv", index=False)
        print(f"\n=== Top values: {col} ===")
        print(tv.to_string(index=False))

    # ── 3. Flatten tags + global tag counts ───────────────────────────────────
    tags_long = flatten_tags(df)
    tag_counts = global_tag_counts(tags_long, min_count=MIN_TAG_COUNT)
    tag_counts.to_csv(f"{PROFILE_DIR}/global_tag_counts.csv", index=False)
    print(f"\n=== Global tag counts (min_count={MIN_TAG_COUNT}) ===")
    print(tag_counts.to_string(index=False))

    valid_tags = set(tag_counts["tag"].tolist())
    print(f"\nValid tags after threshold: {len(valid_tags)}")

    # ── 4. Position-aware tag profile ─────────────────────────────────────────
    pos_df = top_tags_by_position(df)
    pos_df.to_csv(f"{PROFILE_DIR}/top_tags_by_position.csv", index=False)
    print("\n=== Top tags by position (first 20 rows) ===")
    print(pos_df.head(20).to_string(index=False))

    # ── 5. Queue-centric tag profiles ─────────────────────────────────────────
    tq_df = top_tags_by_queue(tags_long)
    tq_df.to_csv(f"{PROFILE_DIR}/top_tags_by_queue.csv", index=False)
    print("\n=== Top tags by queue (first 20 rows) ===")
    print(tq_df.head(20).to_string(index=False))

    tqp_df = top_tags_by_queue_priority_mix(tags_long)
    tqp_df.to_csv(f"{PROFILE_DIR}/top_tags_by_queue_priority_mix.csv", index=False)
    print("\n=== Top tags by queue + priority mix (first 20 rows) ===")
    print(tqp_df.head(20).to_string(index=False))

    tqpos_df = top_tags_by_queue_position(tags_long)
    tqpos_df.to_csv(f"{PROFILE_DIR}/top_tags_by_queue_position.csv", index=False)
    print("\n=== Top tags by queue + position (first 20 rows) ===")
    print(tqpos_df.head(20).to_string(index=False))

    # ── 6. Conditional distributions P(label | tag) ───────────────────────────
    for target in ["queue", "priority"]:
        if target not in tags_long.columns:
            continue
        cond = conditional_distribution(tags_long, target)
        cond.to_csv(f"{PROFILE_DIR}/p_{target}_given_tag.csv", index=False)
        print(f"\n=== P({target} | tag) — first 10 rows ===")
        print(cond.head(10).to_string(index=False))

    # ── 7. Lift: lift(tag -> label) ───────────────────────────────────────────
    for target in ["queue", "priority"]:
        if target not in tags_long.columns:
            continue
        lift_df = compute_lift(df, tags_long, target, valid_tags)
        lift_df.to_csv(f"{PROFILE_DIR}/lift_tag_{target}.csv", index=False)
        print(f"\n=== Lift: tag -> {target} (top 20 by lift) ===")
        print(lift_df.nlargest(20, "lift").to_string(index=False))

    # ── 8. Build historical_queue_tag_hints: top-5 tags per queue ─────────────
    # tq_df is already sorted by queue + count descending from top_tags_by_queue.
    historical_queue_tag_hints = {}
    for queue in tq_df["queue"].unique():
        top5 = tq_df[tq_df["queue"] == queue].head(5)["tag"].tolist()
        historical_queue_tag_hints[queue] = top5

    # ── 9. Write ontology YAML ────────────────────────────────────────────────
    ontology = build_ontology(df, valid_tags, historical_queue_tag_hints)
    with open(ONTOLOGY_PATH, "w", encoding="utf-8") as f:
        yaml.dump(ontology, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"\nOntology written to {ONTOLOGY_PATH}")

    # ── 10. Proxy topic distribution ──────────────────────────────────────────
    print("\nApplying proxy topic mapper ...")
    mapped = apply_proxy_topic_mapper(df)

    dist_df = proxy_topic_distribution(mapped)
    dist_df.to_csv(f"{PROFILE_DIR}/proxy_topic_distribution.csv", index=False)
    print("\n=== Proxy topic distribution ===")
    print(dist_df.to_string(index=False))

    by_queue_df = proxy_topic_by_queue(df, mapped)
    by_queue_df.to_csv(f"{PROFILE_DIR}/proxy_topic_by_queue.csv", index=False)
    print("\n=== Proxy topic by queue ===")
    print(by_queue_df.to_string(index=False))

    print(f"\nAll CSV profiles saved to {PROFILE_DIR}/")


if __name__ == "__main__":
    main()
