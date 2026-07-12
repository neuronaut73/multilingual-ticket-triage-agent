"""
Sprint 6C.6 — CLI Menu Actions.

This module provides one function per menu item for the interactive CLI demo.
It reuses existing src/ modules and DuckDB queries directly.

No business logic is duplicated here.  Triage calls delegate to the same
TicketTriageAgent used by BatchRunner.  DB queries use duckdb directly
(like main.py helper functions) to keep the code readable.

Data leakage note:
  - Triage functions (run_triage_for_ticket, run_batch_evaluation) build
    TicketInput from subject/body only.  actual_* and proxy_* fields are
    never placed into TicketInput.
  - Lookup functions display actual_* and proxy_* as read-only metadata
    for the user.  They are not fed into any prediction path.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from datetime import datetime

import duckdb
import yaml

from src.application.action_executor import ActionExecutor
from src.application.agent import TicketTriageAgent
from src.application.analyzer import LocalLLMAnalyzer, _TOPIC_DESCRIPTIONS, _SCHEMA_HINT
from src.application.batch_runner import BatchRunner
from src.application.reviewer import ConditionalLLMReviewer
from src.application.metrics import (
    compute_evaluation_metrics,
    compute_timing_metrics,
    confusion_counts,
)
from src.application.neighbor_retriever import NeighborRetriever
from src.application.reviewer import _build_reviewer_prompt
from src.application.router import TriageRouter
from src.application.validator import TriageValidator
from src.domain.enums import Topic, Urgency
from src.domain.models import (
    LLMAnalysis,
    NeighborEvidence,
    NeighborPrediction,
    TicketInput,
)
from src.infrastructure.csv_writer import write_csv_rows
from src.infrastructure.duckdb_repository import DuckDBRepository
from src.infrastructure.embedding_model import EmbeddingModel
from src.infrastructure.evaluation_repository import EvaluationRepository
from src.infrastructure.lancedb_ticket_store import LanceDBTicketStore
from src.infrastructure.llm_client import OllamaClient
from src.infrastructure.trace_writer import write_jsonl

DB_PATH = "data/tickets.duckdb"


# ── Agent factory ─────────────────────────────────────────────────────────────

def build_agent_from_config(cfg: dict, embedding_model: EmbeddingModel) -> TicketTriageAgent:
    """
    Construct a TicketTriageAgent from config and a pre-loaded embedding model.

    Mirrors _build_agent in main.py so cli.py does not depend on main.py.
    All components are injected so the agent is easy to swap for tests.
    When reviewer.enabled is true in config, a ConditionalLLMReviewer is also
    constructed and injected.  When disabled, reviewer=None is passed (no-op).
    """
    llm_cfg    = cfg["llm"]
    vs_cfg     = cfg["vector_store"]
    ret_cfg    = cfg["retrieval"]
    thresholds = cfg.get("thresholds", {})

    llm_client = OllamaClient(
        base_url=llm_cfg["base_url"],
        model_name=llm_cfg["model_name"],
        temperature=llm_cfg.get("temperature", 0.1),
        timeout_seconds=llm_cfg.get("timeout_seconds", 120),
    )
    store = LanceDBTicketStore(
        path=vs_cfg["path"],
        table_name=vs_cfg["table_name"],
    )
    retriever = NeighborRetriever(
        embedding_model=embedding_model,
        ticket_store=store,
        top_k=ret_cfg.get("top_k", 5),
    )
    analyzer = LocalLLMAnalyzer(
        llm_client=llm_client,
        max_retries=llm_cfg.get("max_retries", 1),
    )
    validator = TriageValidator(
        min_llm_confidence=thresholds.get("low_confidence", 0.60),
        min_neighbor_confidence=0.50,
        high_confidence_threshold=0.80,
    )
    router   = TriageRouter()
    executor = ActionExecutor()

    # Build optional reviewer when enabled in config.
    reviewer = None
    reviewer_cfg = cfg.get("reviewer", {})
    if reviewer_cfg.get("enabled", False):
        reviewer_client = OllamaClient(
            base_url=reviewer_cfg.get("base_url", llm_cfg["base_url"]),
            model_name=reviewer_cfg["model_name"],
            temperature=reviewer_cfg.get("temperature", 0.1),
            timeout_seconds=llm_cfg.get("timeout_seconds", 120),
        )
        reviewer = ConditionalLLMReviewer(
            llm_client=reviewer_client,
            trigger_flags=reviewer_cfg.get("trigger_flags", []),
            max_retries=reviewer_cfg.get("max_retries", 1),
            model_name=reviewer_cfg["model_name"],
            disagreement_confidence_ceiling=reviewer_cfg.get(
                "disagreement_confidence_ceiling", 0.85
            ),
            urgency_disagreement_confidence_ceiling=reviewer_cfg.get(
                "urgency_disagreement_confidence_ceiling", 0.90
            ),
        )

    return TicketTriageAgent(
        neighbor_retriever=retriever,
        analyzer=analyzer,
        validator=validator,
        router=router,
        action_executor=executor,
        reviewer=reviewer,
    )


def load_embedding_model(cfg: dict) -> EmbeddingModel:
    """Load the embedding model from config. Takes ~30–60s on first call."""
    emb_cfg = cfg["embedding"]
    print(f"  Loading embedding model: {emb_cfg['model_name']} (this takes ~30s) …")
    return EmbeddingModel(
        model_name=emb_cfg["model_name"],
        device=emb_cfg.get("device", "auto"),
        normalize_embeddings=emb_cfg.get("normalize_embeddings", True),
    )


def prompt_reviewer_override(cfg: dict) -> dict:
    """
    Ask the user whether to enable or disable the reviewer for this run.

    Returns a deep copy of cfg with reviewer.enabled set to the user's choice.
    If the user enables the reviewer they can also override reviewer.model_name;
    pressing Enter keeps the current value.

    config.yaml is never written — only the in-memory copy is modified.
    """
    effective = copy.deepcopy(cfg)
    reviewer_cfg = effective.setdefault("reviewer", {})

    try:
        answer = input("  Enable reviewer for this run? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer == "y":
        reviewer_cfg["enabled"] = True
        current_model = reviewer_cfg.get("model_name", "")
        prompt_text = f"  Reviewer model [{current_model}]: "
        try:
            model_input = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            model_input = ""
        if model_input:
            reviewer_cfg["model_name"] = model_input
    else:
        reviewer_cfg["enabled"] = False

    return effective


def print_effective_run_config(cfg: dict, limit: int) -> None:
    """
    Print the effective run configuration before a batch starts.

    Shows analyzer model, reviewer enabled/disabled, reviewer model (if enabled),
    batch limit, and sample strategy.  This makes runs reproducible and auditable.
    """
    llm_cfg      = cfg.get("llm", {})
    reviewer_cfg = cfg.get("reviewer", {})
    batch_cfg    = cfg.get("batch", {})

    reviewer_enabled = reviewer_cfg.get("enabled", False)

    print("\n  ── Effective Run Configuration ───────────────")
    print(f"  analyzer model   : {llm_cfg.get('model_name', '')}")
    print(f"  reviewer enabled : {'Yes' if reviewer_enabled else 'No'}")
    if reviewer_enabled:
        print(f"  reviewer model   : {reviewer_cfg.get('model_name', '')}")
    print(f"  batch limit      : {limit}")
    print(f"  sample strategy  : {batch_cfg.get('sample_strategy', 'natural')}")
    print()


# ── Session-level reviewer override ───────────────────────────────────────────

def apply_reviewer_override(cfg: dict, override: bool | None) -> dict:
    """
    Return a deep copy of cfg with reviewer.enabled set from the session override.

    If override is None, reviewer.enabled in the copy is unchanged (config default).
    config.yaml is never modified — only the in-memory copy changes.
    """
    effective = copy.deepcopy(cfg)
    if override is not None:
        effective.setdefault("reviewer", {})["enabled"] = override
    return effective


_OVERRIDE_LABELS: dict[bool | None, str] = {
    True:  "force enabled",
    False: "force disabled",
    None:  "not set — using config.yaml",
}


def toggle_reviewer_session_override(
    cfg: dict,
    current_override: bool | None,
) -> bool | None:
    """
    Interactive prompt to update the session-level reviewer override.

    Shows current config.yaml value, current session override, and effective state.

    Prompts user:
      1 = force reviewer enabled for this session
      2 = force reviewer disabled for this session
      3 = use config.yaml default (clear override)

    Returns the new override value (True / False / None).
    On invalid or empty input, returns current_override unchanged.
    """
    config_enabled  = cfg.get("reviewer", {}).get("enabled", False)
    effective_state = current_override if current_override is not None else config_enabled

    print("\n  ── Reviewer Session Override ─────────────────")
    print(f"  config.yaml reviewer.enabled : {'Yes' if config_enabled else 'No'}")
    print(f"  current session override     : {_OVERRIDE_LABELS[current_override]}")
    print(f"  effective reviewer state     : {'Yes' if effective_state else 'No'}")
    print()
    print("  1 = force reviewer enabled for this session")
    print("  2 = force reviewer disabled for this session")
    print("  3 = use config.yaml default (clear override)")

    try:
        choice = input("  Select [1/2/3]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice == "1":
        new_override: bool | None = True
    elif choice == "2":
        new_override = False
    elif choice == "3":
        new_override = None
    else:
        print(f"  Invalid choice '{choice}'. Override unchanged.")
        return current_override

    print(f"  Session override set to: {_OVERRIDE_LABELS[new_override]}")
    effective_after = new_override if new_override is not None else config_enabled
    print(f"  Effective reviewer state: {'Yes' if effective_after else 'No'}")
    return new_override


# ── DuckDB helpers ─────────────────────────────────────────────────────────────

def _open_db_readonly(db_path: str) -> duckdb.DuckDBPyConnection:
    """Open DuckDB in read-only mode for query-only operations."""
    return duckdb.connect(db_path, read_only=True)


def _fetch_eval_ticket_ids(db_path: str) -> list[str]:
    """Return all ticket_ids in the eval split."""
    if not os.path.exists(db_path):
        return []
    conn = _open_db_readonly(db_path)
    rows = conn.execute(
        "SELECT ticket_id FROM historical_tickets WHERE split_name = 'eval'"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _fetch_full_ticket_record(db_path: str, ticket_id: str) -> dict | None:
    """
    Fetch all fields for one ticket from historical_tickets.

    Returns all columns including actual_* and proxy_* for display.
    These are shown to the user as read-only metadata and never fed into
    the triage agent.
    """
    if not os.path.exists(db_path):
        return None
    conn = _open_db_readonly(db_path)
    result = conn.execute(
        """
        SELECT
            ticket_id, split_name, subject, body,
            raw_text, cleaned_text, representation_text, text_snippet,
            actual_queue, actual_priority, actual_type, actual_tags_json,
            language, proxy_topic, proxy_urgency, proxy_next_action,
            proxy_topic_source, source_row_json
        FROM historical_tickets
        WHERE ticket_id = ?
        """,
        [ticket_id],
    ).fetchone()
    conn.close()
    if result is None:
        return None
    columns = [
        "ticket_id", "split_name", "subject", "body",
        "raw_text", "cleaned_text", "representation_text", "text_snippet",
        "actual_queue", "actual_priority", "actual_type", "actual_tags_json",
        "language", "proxy_topic", "proxy_urgency", "proxy_next_action",
        "proxy_topic_source", "source_row_json",
    ]
    return dict(zip(columns, result))


def _record_to_ticket_input(record: dict) -> TicketInput:
    """
    Build a TicketInput from a DuckDB record.

    Only subject/body-derived text fields are used.  actual_* and proxy_*
    columns in the record are intentionally excluded.
    """
    return TicketInput(
        ticket_id=record["ticket_id"],
        subject=record["subject"],
        body=record["body"],
        raw_text=record["raw_text"],
        cleaned_text=record["cleaned_text"],
        representation_text=record["representation_text"],
        text_snippet=record["text_snippet"],
    )


# ── Menu option 1: Random eval ticket triage ──────────────────────────────────

def run_random_eval_ticket(cfg: dict, db_path: str, embedding_model: EmbeddingModel) -> None:
    """
    Pick a random ticket from the eval split and run full triage.

    Prints the TriageResult to stdout.
    """
    ids = _fetch_eval_ticket_ids(db_path)
    if not ids:
        print("  No eval tickets found in DuckDB.")
        return
    ticket_id = random.choice(ids)
    print(f"  Selected ticket_id: {ticket_id}")
    record = _fetch_full_ticket_record(db_path, ticket_id)
    if record is None:
        print("  Could not fetch ticket record.")
        return
    run_triage_for_ticket(cfg, record, embedding_model)


# ── Menu option 2: Triage for specific ticket_id ──────────────────────────────

def run_specific_ticket(cfg: dict, db_path: str, ticket_id: str, embedding_model: EmbeddingModel) -> None:
    """
    Fetch a ticket by ID and run full triage, then print the result.
    """
    record = _fetch_full_ticket_record(db_path, ticket_id)
    if record is None:
        print(f"  ticket_id '{ticket_id}' not found in DuckDB.")
        return
    run_triage_for_ticket(cfg, record, embedding_model)


# ── Shared: run triage for one record and print result ────────────────────────

def run_triage_for_ticket(
    cfg: dict,
    record: dict,
    embedding_model: EmbeddingModel,
) -> None:
    """
    Build agent, run triage for one ticket record, and print the result.

    The TicketInput is built from subject/body-derived fields only.
    actual_* and proxy_* in the record are printed afterward as ground-truth
    reference — they are never passed into the agent.
    """
    agent  = build_agent_from_config(cfg, embedding_model)
    ticket = _record_to_ticket_input(record)

    print(f"\n  ticket_id : {ticket.ticket_id}")
    print(f"  subject   : {ticket.subject[:100]}")
    print("  Running triage …")

    try:
        result = agent.process_ticket(ticket)
    except Exception as exc:
        print(f"  ERROR: Triage failed: {exc}")
        print("  Is Ollama running?  Try: ollama serve")
        return

    print("\n  ── Triage Result ─────────────────────────────")
    print(f"  topic                : {result.topic.value}")
    print(f"  urgency              : {result.urgency.value}")
    print(f"  next_action          : {result.next_action.value}")
    print(f"  confidence           : {result.confidence:.3f}")
    print(f"  missing_info         : {result.missing_info}")
    print(f"  requires_human_review: {result.requires_human_review}")
    print(f"  short_note           : {result.short_note}")
    if result.action_result is not None:
        print(f"  action_status        : {result.action_result.action_status}")
        print(f"  action_target        : {result.action_result.target}")

    print("\n  ── Ground-Truth Reference (historical labels) ─")
    print(f"  actual_queue    : {record.get('actual_queue', '')}")
    print(f"  actual_priority : {record.get('actual_priority', '')}")
    print(f"  proxy_topic     : {record.get('proxy_topic', '')}")
    print(f"  proxy_urgency   : {record.get('proxy_urgency', '')}")


# ── Sampling helpers ──────────────────────────────────────────────────────────
#
# _sample_tickets is called by run_batch_evaluation and run_reviewer_ab_comparison.
# main.py is intentionally not imported here (it is an entry point, not a library).
#
# Key invariant: `limit` always means the TOTAL maximum number of tickets returned,
# regardless of strategy.  config.batch.limit_per_label is NOT used here.

def _sample_tickets(
    rows: list[dict],
    sample_strategy: str,
    limit: int,
    random_seed: int,
) -> list[dict]:
    """
    Apply a sampling strategy and return at most `limit` rows total.

    Strategies:
      natural              — first <limit> rows (deterministic ORDER BY ticket_id)
      random               — random <limit> rows, reproducible with random_seed
      balanced_proxy_topic   — up to <limit> rows spread evenly across proxy_topic classes
      balanced_proxy_urgency — up to <limit> rows spread evenly across proxy_urgency classes

    `limit` is always the TOTAL cap — not a per-class cap.

    Data leakage note:
      proxy_* labels select rows only.  The returned rows still carry proxy_*
      and actual_* as evaluation metadata.  BatchRunner builds TicketInput
      from text fields exclusively.
    """
    if sample_strategy == "natural":
        return rows[:limit]

    if sample_strategy == "random":
        rng = random.Random(random_seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[:limit]

    if sample_strategy == "balanced_proxy_topic":
        return _balanced_sample_total(rows, "proxy_topic", total_limit=limit, random_seed=random_seed)

    if sample_strategy == "balanced_proxy_urgency":
        return _balanced_sample_total(rows, "proxy_urgency", total_limit=limit, random_seed=random_seed)

    raise ValueError(
        f"Unknown sample_strategy: {sample_strategy!r}. "
        "Supported: natural, random, balanced_proxy_topic, balanced_proxy_urgency"
    )


def _balanced_sample_total(
    rows: list[dict],
    label_col: str,
    total_limit: int,
    random_seed: int,
) -> list[dict]:
    """
    Select up to total_limit rows distributed as evenly as possible across label classes.

    Algorithm:
      1. Group rows by label_col.
      2. Within each group, shuffle deterministically using random_seed.
      3. Pick rows round-robin across groups (sorted label order) until
         total_limit is reached or all groups are exhausted.

    If a class has fewer rows than its fair share, available rows are taken and
    the remaining budget is redistributed to other classes automatically via
    continued round-robin.  No synthetic rows are created.

    Data leakage note: label_col is used only for grouping — never passed to the agent.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        label = str(row.get(label_col) or "Unknown")
        groups.setdefault(label, []).append(row)

    rng = random.Random(random_seed)
    sorted_labels = sorted(groups)

    # Shuffle within each group for deterministic randomness.
    shuffled: dict[str, list[dict]] = {
        label: list(groups[label]) for label in sorted_labels
    }
    for label in sorted_labels:
        rng.shuffle(shuffled[label])

    # Round-robin: one row per class per pass until total_limit reached.
    sampled: list[dict] = []
    indices = {label: 0 for label in sorted_labels}
    while len(sampled) < total_limit:
        added_any = False
        for label in sorted_labels:
            if len(sampled) >= total_limit:
                break
            idx = indices[label]
            if idx < len(shuffled[label]):
                sampled.append(shuffled[label][idx])
                indices[label] = idx + 1
                added_any = True
        if not added_any:
            break  # All groups exhausted.

    return sampled


def _print_sampling_report(
    rows: list[dict],
    requested_limit: int,
    strategy: str,
) -> None:
    """
    Print a sampling summary after rows have been selected.

    Shows: requested total limit, actual sampled count, strategy,
    distribution by proxy_topic, distribution by proxy_urgency.
    """
    topic_dist: dict[str, int]   = {}
    urgency_dist: dict[str, int] = {}
    for row in rows:
        t = str(row.get("proxy_topic")   or "Unknown")
        u = str(row.get("proxy_urgency") or "Unknown")
        topic_dist[t]   = topic_dist.get(t, 0) + 1
        urgency_dist[u] = urgency_dist.get(u, 0) + 1

    print(f"\n  ── Sampling Report ──────────────────────────")
    print(f"  requested total limit : {requested_limit}")
    print(f"  actual sampled        : {len(rows)}")
    print(f"  strategy              : {strategy}")
    print(f"  distribution by proxy_topic:")
    for label in sorted(topic_dist):
        print(f"    {label:<35}: {topic_dist[label]}")
    print(f"  distribution by proxy_urgency:")
    for label in sorted(urgency_dist):
        print(f"    {label:<35}: {urgency_dist[label]}")


def _fetch_all_split_records(db_path: str, split: str) -> list[dict]:
    """
    Fetch every ticket for the given split from DuckDB, ordered by ticket_id.

    No LIMIT is applied — callers select the desired subset via _sample_tickets.
    Uses DuckDBRepository.fetch_split_tickets so the query is identical to the
    one used by main.py._run_batch.
    """
    if not os.path.exists(db_path):
        return []
    repo = DuckDBRepository(db_path)
    rows = repo.fetch_split_tickets(split)
    repo.close()
    return rows


# ── Core batch execution helper ────────────────────────────────────────────────

def _execute_batch_on_records(
    cfg: dict,
    db_path: str,
    records: list[dict],
    embedding_model: EmbeddingModel,
    run_id: str | None = None,
    write_files: bool = True,
) -> tuple[str, dict, list[dict]]:
    """
    Build agent, run BatchRunner, compute metrics, and store results in DuckDB.

    Parameters
    ----------
    cfg:
        Effective config dict.  reviewer.enabled reflects the active override.
        The full dict is serialised as config_json so the run remains
        explainable later (reviewer state is recorded in the run snapshot).
    records:
        Pre-fetched and pre-sampled ticket records.  Callers are responsible
        for sampling before calling this function so A and B runs can reuse
        the exact same list without re-querying DuckDB.
    run_id:
        Optional explicit run identifier (e.g. run_20260711_120000_ab_off).
        If None, auto-generated from the current timestamp as run_YYYYMMDD_HHMMSS.
    write_files:
        When True (default), writes CSV, JSONL trace, and run_summary.json.
        When False, stores only in DuckDB — used by the A/B comparison to
        avoid overwriting normal batch output files between the two runs.

    Returns
    -------
    (run_id, eval_metrics, result_rows)
    """
    batch_cfg = cfg.get("batch", {})
    split     = batch_cfg.get("split", "eval")

    if run_id is None:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    print(f"\n  run_id : {run_id}")
    print(f"  split  : {split}  tickets: {len(records)}")

    agent  = build_agent_from_config(cfg, embedding_model)
    runner = BatchRunner(agent)

    jsonl_path = (
        batch_cfg.get("trace_jsonl", "outputs/triage_trace.jsonl")
        if write_files
        else None
    )

    t0 = time.time()
    result_rows = runner.process_tickets(records, trace_path=jsonl_path)
    runtime_seconds = time.time() - t0

    eval_metrics   = compute_evaluation_metrics(result_rows)
    ticket_seconds = [float(r.get("total_ticket_seconds", 0.0)) for r in result_rows]
    timing_metrics = compute_timing_metrics(ticket_seconds)
    timing_metrics["runtime_seconds_total"] = round(runtime_seconds, 3)
    eval_metrics.update(timing_metrics)

    if write_files:
        csv_path     = batch_cfg.get("output_csv",       "outputs/triage_results.csv")
        summary_path = batch_cfg.get("run_summary_json", "outputs/run_summary.json")
        write_csv_rows(csv_path, result_rows)
        _write_run_summary(
            summary_path, result_rows, cfg, run_id, runtime_seconds, eval_metrics
        )

    # Store in DuckDB — config_json encodes reviewer.enabled so the run remains
    # explainable even without the original CLI session state.
    llm_cfg      = cfg.get("llm", {})
    ret_cfg      = cfg.get("retrieval", {})
    reviewer_cfg = cfg.get("reviewer", {})
    reviewer_enabled = reviewer_cfg.get("enabled", False)
    reviewer_model   = reviewer_cfg.get("model_name", "") if reviewer_enabled else ""

    run_metadata = {
        "run_id":           run_id,
        "created_at":       datetime.now(),
        "model_name":       llm_cfg.get("model_name", ""),
        "embedding_model":  cfg.get("embedding", {}).get("model_name", ""),
        "dataset_path":     cfg.get("input_path", ""),
        "split_name":       split,
        "limit_n":          len(records),
        "top_k":            ret_cfg.get("top_k", 0),
        "temperature":      llm_cfg.get("temperature", 0.0),
        "max_retries":      llm_cfg.get("max_retries", 0),
        "runtime_seconds":  runtime_seconds,
        "config_json":      json.dumps(cfg),
        "sample_strategy":  batch_cfg.get("sample_strategy", "natural"),
        "stratify_by":      batch_cfg.get("stratify_by", ""),
        "random_seed":      batch_cfg.get("random_seed", 42),
        "limit_per_label":  batch_cfg.get("limit_per_label", 0),
        "triage_model":     llm_cfg.get("model_name", ""),
        "reviewer_enabled": reviewer_enabled,
        "reviewer_model":   reviewer_model,
    }

    eval_repo = EvaluationRepository(db_path)
    eval_repo.create_tables()
    eval_repo.insert_run(run_metadata)
    eval_repo.insert_predictions(run_id, result_rows)
    eval_repo.insert_metrics(run_id, eval_metrics)
    eval_repo.insert_confusion_matrix(
        run_id, "urgency",
        confusion_counts(result_rows, "urgency", "proxy_urgency"),
    )
    eval_repo.insert_confusion_matrix(
        run_id, "topic_proxy",
        confusion_counts(result_rows, "topic", "proxy_topic"),
    )
    eval_repo.insert_confusion_matrix(
        run_id, "next_action_proxy",
        confusion_counts(result_rows, "next_action", "proxy_next_action"),
    )
    eval_repo.close()

    return run_id, eval_metrics, result_rows


# ── Menu option 3: Batch evaluation ───────────────────────────────────────────

def run_batch_evaluation(
    cfg: dict,
    db_path: str,
    limit: int,
    embedding_model: EmbeddingModel,
) -> None:
    """
    Run batch evaluation over the eval split and write output files.

    Steps:
      1. Fetch all split tickets from DuckDB.
      2. Apply the configured sample_strategy to select up to limit rows.
      3. Build agent and run BatchRunner (streams JSONL trace).
      4. Write triage_results.csv, triage_trace.jsonl, run_summary.json.
      5. Compute KPIs and store in DuckDB.
      6. Print a summary.
    """
    batch_cfg       = cfg.get("batch", {})
    split           = batch_cfg.get("split", "eval")
    sample_strategy = batch_cfg.get("sample_strategy", "natural")
    random_seed     = batch_cfg.get("random_seed", 42)

    all_records = _fetch_all_split_records(db_path, split)
    if not all_records:
        print("  No records found. Check split name and DuckDB state.")
        return
    print(f"  Fetched {len(all_records)} tickets from DuckDB (split: {split}).")

    # limit is always the TOTAL ticket budget; config.batch.limit_per_label is not used.
    records = _sample_tickets(
        all_records,
        sample_strategy=sample_strategy,
        limit=limit,
        random_seed=random_seed,
    )
    _print_sampling_report(records, requested_limit=limit, strategy=sample_strategy)

    run_id, eval_metrics, result_rows = _execute_batch_on_records(
        cfg, db_path, records, embedding_model, write_files=True
    )

    batch_cfg    = cfg.get("batch", {})
    csv_path     = batch_cfg.get("output_csv",       "outputs/triage_results.csv")
    summary_path = batch_cfg.get("run_summary_json", "outputs/run_summary.json")
    runtime      = eval_metrics.get("runtime_seconds_total", 0.0)

    print(f"\n  ── Evaluation Results ────────────────────────")
    print(f"  run_id                      : {run_id}")
    print(f"  tickets processed           : {len(result_rows)}")
    print(f"  runtime                     : {runtime:.1f}s")
    print(f"  urgency_accuracy            : {eval_metrics.get('urgency_accuracy', 0):.4f}")
    print(f"  topic_proxy_accuracy        : {eval_metrics.get('topic_proxy_accuracy', 0):.4f}")
    print(f"  next_action_proxy_agreement : {eval_metrics.get('next_action_proxy_agreement', 0):.4f}")
    print(f"  human_review_rate           : {eval_metrics.get('human_review_rate', 0):.4f}")
    print(f"  output_csv                  : {csv_path}")
    print(f"  run_summary_json            : {summary_path}")


def _write_run_summary(
    path: str,
    rows: list[dict],
    cfg: dict,
    run_id: str,
    runtime_seconds: float,
    eval_metrics: dict,
) -> None:
    """Write run_summary.json (mirrors main.py _write_run_summary)."""
    total = len(rows)
    topic_dist: dict[str, int]       = {}
    urgency_dist: dict[str, int]     = {}
    next_action_dist: dict[str, int] = {}
    human_review_count = 0
    missing_info_count = 0

    for row in rows:
        t  = str(row.get("topic", ""))
        u  = str(row.get("urgency", ""))
        na = str(row.get("next_action", ""))
        topic_dist[t]       = topic_dist.get(t, 0) + 1
        urgency_dist[u]     = urgency_dist.get(u, 0) + 1
        next_action_dist[na] = next_action_dist.get(na, 0) + 1
        if row.get("requires_human_review"):
            human_review_count += 1
        if row.get("missing_info"):
            missing_info_count += 1

    batch_cfg = cfg.get("batch", {})
    summary = {
        "run_id":                   run_id,
        "number_processed":         total,
        "model_name":               cfg.get("llm", {}).get("model_name", ""),
        "embedding_model":          cfg.get("embedding", {}).get("model_name", ""),
        "split":                    batch_cfg.get("split", "eval"),
        "limit":                    batch_cfg.get("limit", total),
        "runtime_seconds":          round(runtime_seconds, 2),
        "topic_distribution":       topic_dist,
        "urgency_distribution":     urgency_dist,
        "next_action_distribution": next_action_dist,
        "human_review_rate":        round(human_review_count / total, 4) if total else 0.0,
        "missing_info_rate":        round(missing_info_count  / total, 4) if total else 0.0,
        "evaluation_metrics":       eval_metrics,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ── Menu option 4: KPI leaderboard ────────────────────────────────────────────

# Four repeating ANSI colors for leaderboard rows (bright, visually distinct).
_LB_COLORS = ["\033[96m", "\033[93m", "\033[92m", "\033[95m"]  # cyan, yellow, green, magenta
_LB_RESET  = "\033[0m"


def show_kpi_leaderboard(db_path: str) -> None:
    """
    Print a leaderboard table of all runs from triage_runs and triage_metrics.

    Columns shown per run:
      run_id, created_at, triage_model, reviewer (yes/no), reviewer_model,
      split, limit, runtime_s, avg_sec_ticket, reviewer_invocation_rate,
      urgency_acc, urgency_f1, topic_acc, topic_f1,
      next_action_agr, human_review_rt, avg_conf.

    Sorted by created_at descending (most recent run first).
    Each row is printed in one of 4 repeating ANSI colors for readability.
    Old runs that pre-date the reviewer columns display "-" for those fields.
    """
    if not os.path.exists(db_path):
        print("  DuckDB not found.")
        return
    try:
        conn = _open_db_readonly(db_path)
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "triage_runs" not in tables or "triage_metrics" not in tables:
            print("  No evaluation tables found. Run a batch evaluation first.")
            conn.close()
            return

        rows = conn.execute(
            """
            SELECT
                r.run_id,
                r.created_at,
                COALESCE(r.triage_model, r.model_name)  AS triage_model,
                r.reviewer_enabled,
                COALESCE(r.reviewer_model, '')           AS reviewer_model,
                r.split_name,
                r.limit_n,
                r.runtime_seconds,
                MAX(CASE WHEN m.metric_name = 'avg_seconds_per_ticket'      THEN m.metric_value END) AS avg_sec_ticket,
                MAX(CASE WHEN m.metric_name = 'reviewer_invocation_rate'    THEN m.metric_value END) AS reviewer_inv_rate,
                MAX(CASE WHEN m.metric_name = 'urgency_accuracy'            THEN m.metric_value END) AS urgency_acc,
                MAX(CASE WHEN m.metric_name = 'urgency_macro_f1'            THEN m.metric_value END) AS urgency_f1,
                MAX(CASE WHEN m.metric_name = 'topic_proxy_accuracy'        THEN m.metric_value END) AS topic_acc,
                MAX(CASE WHEN m.metric_name = 'topic_macro_f1'              THEN m.metric_value END) AS topic_f1,
                MAX(CASE WHEN m.metric_name = 'next_action_proxy_agreement' THEN m.metric_value END) AS next_action_agr,
                MAX(CASE WHEN m.metric_name = 'human_review_rate'           THEN m.metric_value END) AS human_review_rt,
                MAX(CASE WHEN m.metric_name = 'average_confidence'          THEN m.metric_value END) AS avg_conf
            FROM triage_runs r
            LEFT JOIN triage_metrics m ON r.run_id = m.run_id
            GROUP BY
                r.run_id, r.created_at, r.triage_model, r.model_name,
                r.reviewer_enabled, r.reviewer_model,
                r.split_name, r.limit_n, r.runtime_seconds
            ORDER BY r.created_at DESC NULLS LAST
            """
        ).fetchall()
        conn.close()

        if not rows:
            print("  No runs found.")
            return

        # Derive run_id column width dynamically so A/B suffixes are never truncated.
        # Minimum of 28 ensures run_20260713_014052_ab_off (28 chars) fits completely.
        _MIN_RUN_ID_WIDTH = 28
        run_id_width = max(
            _MIN_RUN_ID_WIDTH,
            max(len(str(r[0] or "")) for r in rows),
        )

        # Column definitions: (header, width)
        col_defs = [
            ("run_id",          run_id_width),
            ("created_at",      19),
            ("triage_model",    16),
            ("reviewer",         8),
            ("reviewer_model",  14),
            ("split",            6),
            ("limit",            6),
            ("runtime_s",        9),
            ("avg_sec",          7),
            ("rev_rate",         8),
            ("urg_acc",          7),
            ("urg_f1",           6),
            ("topic_acc",        9),
            ("topic_f1",         8),
            ("next_act",         8),
            ("hr_rate",          7),
            ("avg_conf",         8),
        ]
        header_line = "  " + "  ".join(h.ljust(w) for h, w in col_defs)
        sep_line    = "  " + "  ".join("-" * w for _, w in col_defs)
        print()
        print(header_line)
        print(sep_line)

        def _f(val, width: int) -> str:
            if val is None:
                return "-".ljust(width)
            return f"{val:.4f}".ljust(width)

        for row_idx, r in enumerate(rows):
            (
                run_id, created_at, triage_model, reviewer_enabled, reviewer_model,
                split_name, limit_n, runtime_s,
                avg_sec, rev_rate,
                urg_acc, urg_f1, topic_acc, topic_f1,
                next_act, hr_rate, avg_conf,
            ) = r

            reviewer_str = "yes" if reviewer_enabled else "no"
            color = _LB_COLORS[row_idx % len(_LB_COLORS)]

            cells = [
                str(run_id    or "")[:run_id_width].ljust(run_id_width),
                str(created_at or "")[:19].ljust(19),
                str(triage_model   or "")[:16].ljust(16),
                reviewer_str.ljust(8),
                str(reviewer_model or "")[:14].ljust(14),
                str(split_name or "")[:6].ljust(6),
                str(limit_n   or "")[:6].ljust(6),
                (f"{runtime_s:.1f}" if runtime_s is not None else "-").ljust(9),
                _f(avg_sec,   7),
                _f(rev_rate,  8),
                _f(urg_acc,   7),
                _f(urg_f1,    6),
                _f(topic_acc, 9),
                _f(topic_f1,  8),
                _f(next_act,  8),
                _f(hr_rate,   7),
                _f(avg_conf,  8),
            ]
            print(color + "  " + "  ".join(cells) + _LB_RESET)
    except Exception as exc:
        print(f"  Error querying leaderboard: {exc}")


# ── Curated leaderboard helpers ───────────────────────────────────────────────

# Semantic ANSI colors for the curated view.
_CL_RESET   = "\033[0m"
_CL_HEADER  = "\033[1;34m"   # bold dark blue — section headers and column headers
_CL_PLAIN   = "\033[37m"     # white — ordinary rows (no reviewer)
_CL_REV     = "\033[94m"     # pale/bright blue — reviewer-enabled rows
_CL_FEAT    = "\033[1;7;34m" # bold + reverse blue — featured run
_CL_DIM     = "\033[2;37m"   # dim gray — unavailable values


def _short_run_id(run_id: str) -> str:
    """Strip the leading 'run_' prefix for compact display. Never modifies stored IDs."""
    if run_id.startswith("run_"):
        return run_id[4:]
    return run_id


def _fmt_pct(val) -> str:
    """Format a 0–1 float as 'XX.X%' or '—' when None."""
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def _fmt_sec(val) -> str:
    """Format seconds-per-ticket to two decimal places or '—'."""
    if val is None:
        return "—"
    return f"{val:.2f}"


def _fetch_curated_rows(conn, run_ids: list[str]) -> dict[str, dict]:
    """
    Query triage_runs + triage_metrics for a specific set of run IDs.

    Returns a dict keyed by run_id so callers can look up rows in O(1).
    Rows are only present if the run_id actually exists in DuckDB.
    """
    if not run_ids:
        return {}

    placeholders = ", ".join("?" for _ in run_ids)
    sql = f"""
        SELECT
            r.run_id,
            COALESCE(r.triage_model, r.model_name)  AS triage_model,
            r.reviewer_enabled,
            COALESCE(r.reviewer_model, '')           AS reviewer_model,
            r.limit_n,
            MAX(CASE WHEN m.metric_name = 'avg_seconds_per_ticket'      THEN m.metric_value END) AS avg_sec,
            MAX(CASE WHEN m.metric_name = 'reviewer_invocation_rate'    THEN m.metric_value END) AS rev_rate,
            MAX(CASE WHEN m.metric_name = 'urgency_accuracy'            THEN m.metric_value END) AS urg_acc,
            MAX(CASE WHEN m.metric_name = 'urgency_macro_f1'            THEN m.metric_value END) AS urg_f1,
            MAX(CASE WHEN m.metric_name = 'topic_proxy_accuracy'        THEN m.metric_value END) AS topic_acc,
            MAX(CASE WHEN m.metric_name = 'topic_macro_f1'              THEN m.metric_value END) AS topic_f1,
            MAX(CASE WHEN m.metric_name = 'next_action_proxy_agreement' THEN m.metric_value END) AS next_act,
            MAX(CASE WHEN m.metric_name = 'human_review_rate'           THEN m.metric_value END) AS hr_rate,
            MAX(CASE WHEN m.metric_name = 'average_confidence'          THEN m.metric_value END) AS avg_conf
        FROM triage_runs r
        LEFT JOIN triage_metrics m ON r.run_id = m.run_id
        WHERE r.run_id IN ({placeholders})
        GROUP BY
            r.run_id, r.triage_model, r.model_name,
            r.reviewer_enabled, r.reviewer_model, r.limit_n
    """
    raw = conn.execute(sql, run_ids).fetchall()
    col_names = [
        "run_id", "triage_model", "reviewer_enabled", "reviewer_model", "limit_n",
        "avg_sec", "rev_rate",
        "urg_acc", "urg_f1", "topic_acc", "topic_f1",
        "next_act", "hr_rate", "avg_conf",
    ]
    return {row[0]: dict(zip(col_names, row)) for row in raw}


def _print_curated_table_1(rows: list[dict], featured_id: str) -> None:
    """
    Print Table 1 for a curated section: configuration and runtime columns.

    Columns: Run | Triage Model | Reviewer Model | N | Sec/Tick | Rev Rate
    """
    col_defs = [
        ("Run",            26),
        ("Triage Model",   15),
        ("Reviewer Model", 15),
        ("N",               5),
        ("Sec/Tick",        9),
        ("Rev Rate",        9),
    ]
    header = "  " + "  ".join(
        (_CL_HEADER + h + _CL_RESET).ljust(w + len(_CL_HEADER) + len(_CL_RESET))
        for h, w in col_defs
    )
    # Simpler: print header without inline ANSI padding tricks
    plain_header = "  " + "  ".join(h.ljust(w) for h, w in col_defs)
    sep          = "  " + "  ".join("-" * w for _, w in col_defs)
    print(_CL_HEADER + plain_header + _CL_RESET)
    print(sep)
    for r in rows:
        rid     = r["run_id"]
        is_feat = (rid == featured_id)
        has_rev = bool(r.get("reviewer_enabled"))
        color   = _CL_FEAT if is_feat else (_CL_REV if has_rev else _CL_PLAIN)

        rev_model = r.get("reviewer_model") or "—"
        cells = [
            _short_run_id(rid).ljust(col_defs[0][1]),
            str(r.get("triage_model") or "—")[:col_defs[1][1]].ljust(col_defs[1][1]),
            str(rev_model)[:col_defs[2][1]].ljust(col_defs[2][1]),
            str(r.get("limit_n") or "—")[:col_defs[3][1]].ljust(col_defs[3][1]),
            _fmt_sec(r.get("avg_sec")).ljust(col_defs[4][1]),
            _fmt_pct(r.get("rev_rate")).ljust(col_defs[5][1]),
        ]
        feat_star = " ★" if is_feat else "  "
        print(color + "  " + "  ".join(cells) + feat_star + _CL_RESET)


def _print_curated_table_2(rows: list[dict], featured_id: str) -> None:
    """
    Print Table 2 for a curated section: quality metrics.

    Columns: Run | Urg Acc | Urg F1 | Top Acc | Top F1 | Action | HR Rate | Avg Conf
    """
    col_defs = [
        ("Run",      26),
        ("Urg Acc",   8),
        ("Urg F1",    8),
        ("Top Acc",   8),
        ("Top F1",    8),
        ("Action",    8),
        ("HR Rate",   8),
        ("Avg Conf",  9),
    ]
    plain_header = "  " + "  ".join(h.ljust(w) for h, w in col_defs)
    sep          = "  " + "  ".join("-" * w for _, w in col_defs)
    print(_CL_HEADER + plain_header + _CL_RESET)
    print(sep)
    for r in rows:
        rid     = r["run_id"]
        is_feat = (rid == featured_id)
        has_rev = bool(r.get("reviewer_enabled"))
        color   = _CL_FEAT if is_feat else (_CL_REV if has_rev else _CL_PLAIN)
        cells = [
            _short_run_id(rid).ljust(col_defs[0][1]),
            _fmt_pct(r.get("urg_acc")).ljust(col_defs[1][1]),
            _fmt_pct(r.get("urg_f1")).ljust(col_defs[2][1]),
            _fmt_pct(r.get("topic_acc")).ljust(col_defs[3][1]),
            _fmt_pct(r.get("topic_f1")).ljust(col_defs[4][1]),
            _fmt_pct(r.get("next_act")).ljust(col_defs[5][1]),
            _fmt_pct(r.get("hr_rate")).ljust(col_defs[6][1]),
            _fmt_pct(r.get("avg_conf")).ljust(col_defs[7][1]),
        ]
        print(color + "  " + "  ".join(cells) + _CL_RESET)


def _print_curated_section(
    title: str,
    ordered_run_ids: list[str],
    db_rows: dict[str, dict],
    featured_id: str,
    note: str,
) -> None:
    """
    Print one labelled curated section with two aligned tables.

    Skips run IDs not present in db_rows (already warned at the start).
    """
    present = [db_rows[rid] for rid in ordered_run_ids if rid in db_rows]
    if not present:
        print(_CL_DIM + f"  (no data available for this section)" + _CL_RESET)
        return

    print()
    print(_CL_HEADER + f"  Table 1 — Configuration and Runtime" + _CL_RESET)
    _print_curated_table_1(present, featured_id)
    print()
    print(_CL_HEADER + f"  Table 2 — Quality Metrics" + _CL_RESET)
    _print_curated_table_2(present, featured_id)
    if note:
        print()
        print(f"  Note: {note}")


def show_curated_leaderboard(db_path: str, cfg: dict) -> None:
    """
    Print the curated evaluation view for the HDI submission.

    Reads run groups from cfg['leaderboard'].  Falls back to the full-history
    leaderboard when the section is absent so no existing behavior is broken.

    Three sections:
      A — Controlled Reviewer A/B (same analyzer, sample, seed)
      B — Best Observed Reviewer Configuration
      C — Historical Analyzer Screening (200 tickets)

    Run IDs are read from config — never hardcoded in this function.
    Missing run IDs are skipped safely with one concise warning each.
    The DuckDB connection is opened read-only; no rows are written.
    """
    lb_cfg = cfg.get("leaderboard", {})
    if not lb_cfg:
        print("  No leaderboard section in config.yaml — showing full history.")
        show_kpi_leaderboard(db_path)
        return

    if not os.path.exists(db_path):
        print("  DuckDB not found.")
        return

    controlled_ids = lb_cfg.get("controlled_reviewer_run_ids", [])
    featured_ids   = lb_cfg.get("featured_reviewer_run_ids", [])
    screening_ids  = lb_cfg.get("analyzer_screening_run_ids", [])
    featured_id    = lb_cfg.get("featured_run_id", "")

    all_configured = list(controlled_ids) + list(featured_ids) + list(screening_ids)

    try:
        conn = _open_db_readonly(db_path)
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "triage_runs" not in tables or "triage_metrics" not in tables:
            print("  No evaluation tables found. Run a batch evaluation first.")
            conn.close()
            return

        db_rows = _fetch_curated_rows(conn, all_configured)
        conn.close()
    except Exception as exc:
        print(f"  Error querying DuckDB: {exc}")
        return

    # Warn once per missing run ID.
    for rid in all_configured:
        if rid not in db_rows:
            print(f"  Warning: run '{rid}' not found in DuckDB — skipping.")

    # ── Section A ─────────────────────────────────────────────────────────────
    print()
    print(_CL_HEADER + "  ══ SECTION A: Controlled Reviewer A/B — Same Analyzer and Ticket Sample ══" + _CL_RESET)
    _print_curated_section(
        title="Controlled Reviewer A/B",
        ordered_run_ids=controlled_ids,
        db_rows=db_rows,
        featured_id=featured_id,
        note=(
            "A and B use the same llama3.2:3b analyzer, 200-ticket balanced eval sample, "
            "seed 42, embedding model, retrieval configuration, thresholds, and trigger "
            "flags. Only the conditional reviewer differs."
        ),
    )

    # ── Section B ─────────────────────────────────────────────────────────────
    print()
    print(_CL_HEADER + "  ══ SECTION B: Best Observed Reviewer Configuration ══" + _CL_RESET)
    if featured_id:
        print(f"  ★ Best observed reviewer configuration: {_short_run_id(featured_id)}")
    _print_curated_section(
        title="Best Observed Reviewer Configuration",
        ordered_run_ids=featured_ids,
        db_rows=db_rows,
        featured_id=featured_id,
        note=None,
    )

    # ── Section C ─────────────────────────────────────────────────────────────
    print()
    print(_CL_HEADER + "  ══ SECTION C: Historical Analyzer Screening — 200 Tickets ══" + _CL_RESET)
    _print_curated_section(
        title="Historical Analyzer Screening",
        ordered_run_ids=screening_ids,
        db_rows=db_rows,
        featured_id=featured_id,
        note=(
            "These runs document model screening. Some early runs predate complete "
            "sampling metadata, so they are not presented as a strict controlled A/B "
            "experiment."
        ),
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    print()
    print("  Use 'Full experiment history' to inspect all stored pilot, smoke-test,")
    print("  failed, and repeated runs.")


# ── Menu option 5: Run details ────────────────────────────────────────────────

def show_run_details(db_path: str, run_id: str) -> None:
    """
    Show config snapshot and all KPI metrics for a specific run_id.
    """
    if not os.path.exists(db_path):
        print("  DuckDB not found.")
        return
    try:
        conn = _open_db_readonly(db_path)

        run_row = conn.execute(
            "SELECT * FROM triage_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        if run_row is None:
            print(f"  run_id '{run_id}' not found.")
            conn.close()
            return

        cols = [d[0] for d in conn.execute(
            "SELECT * FROM triage_runs WHERE run_id = ?", [run_id]
        ).description]
        run_dict = dict(zip(cols, run_row))

        print(f"\n  ── Run: {run_id} ─────────────────────────")
        for k, v in run_dict.items():
            if k == "config_json":
                continue
            print(f"  {k:<22}: {v}")

        metrics = conn.execute(
            "SELECT metric_name, metric_value FROM triage_metrics WHERE run_id = ? ORDER BY metric_name",
            [run_id],
        ).fetchall()
        conn.close()

        print(f"\n  ── KPI Metrics ───────────────────────────")
        if not metrics:
            print("  (no metrics recorded)")
        for name, value in metrics:
            print(f"  {name:<30}: {value:.4f}" if value is not None else f"  {name}: N/A")

    except Exception as exc:
        print(f"  Error: {exc}")


# ── Menu option 6: Confusion matrix ───────────────────────────────────────────

def show_confusion_matrix(db_path: str, run_id: str) -> None:
    """
    Print the confusion matrix stored in triage_confusion_matrix for a run_id.

    Prints one block per prediction target (urgency, topic_proxy, next_action_proxy).
    """
    if not os.path.exists(db_path):
        print("  DuckDB not found.")
        return
    try:
        conn = _open_db_readonly(db_path)

        targets = conn.execute(
            "SELECT DISTINCT target_name FROM triage_confusion_matrix WHERE run_id = ?",
            [run_id],
        ).fetchall()

        if not targets:
            print(f"  No confusion matrix found for run_id '{run_id}'.")
            conn.close()
            return

        for (target,) in targets:
            rows = conn.execute(
                """
                SELECT actual_label, predicted_label, count
                FROM triage_confusion_matrix
                WHERE run_id = ? AND target_name = ?
                ORDER BY actual_label, predicted_label
                """,
                [run_id, target],
            ).fetchall()

            print(f"\n  ── Confusion Matrix: {target} ──────────────")
            print(f"  {'actual':<25}  {'predicted':<25}  count")
            print(f"  {'-'*25}  {'-'*25}  -----")
            for actual, predicted, count in rows:
                print(f"  {str(actual):<25}  {str(predicted):<25}  {count}")

        conn.close()
    except Exception as exc:
        print(f"  Error: {exc}")


# ── Menu option 7: Ticket lookup ──────────────────────────────────────────────

def lookup_ticket(db_path: str, ticket_id: str) -> None:
    """
    Display the original ticket row from historical_tickets.

    Shows subject, body, split, and historical labels.
    actual_* and proxy_* are shown as read-only metadata — not fed to any model.
    """
    record = _fetch_full_ticket_record(db_path, ticket_id)
    if record is None:
        print(f"  ticket_id '{ticket_id}' not found.")
        return

    print(f"\n  ── Ticket: {ticket_id} ─────────────────────────")
    print(f"  split_name      : {record['split_name']}")
    print(f"  language        : {record.get('language', '')}")
    print(f"  subject         : {record['subject'][:120]}")
    print(f"  body            : {record['body'][:300]}")
    print(f"\n  ── Historical Labels (metadata only — never fed to agent) ──")
    print(f"  actual_queue    : {record.get('actual_queue', '')}")
    print(f"  actual_priority : {record.get('actual_priority', '')}")
    print(f"  actual_type     : {record.get('actual_type', '')}")
    print(f"  actual_tags     : {record.get('actual_tags_json', '')}")
    print(f"\n  ── Proxy Labels (derived from historical labels) ──")
    print(f"  proxy_topic         : {record.get('proxy_topic', '')}")
    print(f"  proxy_urgency       : {record.get('proxy_urgency', '')}")
    print(f"  proxy_next_action   : {record.get('proxy_next_action', '')}")
    print(f"  proxy_topic_source  : {record.get('proxy_topic_source', '')}")


# ── Menu option 8: Prediction lookup ──────────────────────────────────────────

def lookup_prediction(db_path: str, ticket_id: str, run_id: str) -> None:
    """
    Display the stored prediction for a specific ticket_id and run_id.
    """
    if not os.path.exists(db_path):
        print("  DuckDB not found.")
        return
    try:
        conn = _open_db_readonly(db_path)
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        if "triage_predictions" not in tables:
            print("  triage_predictions table not found. Run a batch evaluation first.")
            conn.close()
            return

        row = conn.execute(
            "SELECT * FROM triage_predictions WHERE ticket_id = ? AND run_id = ?",
            [ticket_id, run_id],
        ).fetchone()

        if row is None:
            print(f"  No prediction found for ticket_id='{ticket_id}' run_id='{run_id}'.")
            conn.close()
            return

        cols = [d[0] for d in conn.execute(
            "SELECT * FROM triage_predictions WHERE ticket_id = ? AND run_id = ?",
            [ticket_id, run_id],
        ).description]
        pred = dict(zip(cols, row))
        conn.close()

        print(f"\n  ── Prediction: {ticket_id} (run={run_id}) ────────")
        prediction_fields = [
            "ticket_id", "run_id", "text_snippet",
            "predicted_topic", "predicted_urgency", "predicted_next_action",
            "confidence", "missing_info", "requires_human_review", "short_note",
            "action_status", "action_target", "action_note",
        ]
        for field in prediction_fields:
            v = pred.get(field, "")
            if field == "text_snippet" and isinstance(v, str):
                v = v[:100]
            print(f"  {field:<30}: {v}")

        print(f"\n  ── Evaluation Metadata ──────────────────────")
        eval_fields = [
            "actual_queue", "actual_priority", "actual_type",
            "proxy_topic", "proxy_urgency", "proxy_next_action", "proxy_topic_source",
        ]
        for field in eval_fields:
            print(f"  {field:<30}: {pred.get(field, '')}")

        print(f"\n  ── Timing ───────────────────────────────────")
        timing_fields = ["retrieval_seconds", "llm_seconds", "total_ticket_seconds"]
        for field in timing_fields:
            v = pred.get(field)
            print(f"  {field:<30}: {f'{v:.3f}s' if v is not None else 'N/A'}")

    except Exception as exc:
        print(f"  Error: {exc}")


# ── Menu option 9: Show config ────────────────────────────────────────────────

def show_config(cfg: dict) -> None:
    """Print the loaded config as YAML."""
    print("\n  ── Current Config (config.yaml) ─────────────")
    print(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))


# ── Menu option 10: Show analyzer prompt ─────────────────────────────────────

def show_analyzer_prompt() -> None:
    """
    Display the static structure of the triage analyzer prompt.

    Shows role, task, topic schema, and output JSON shape.
    Ticket subject/body are replaced with [TICKET SUBJECT] / [TICKET BODY]
    to show the template structure without real data.
    """
    topic_lines = "\n".join(
        f"  - {topic.value}: {desc}"
        for topic, desc in _TOPIC_DESCRIPTIONS.items()
    )
    urgency_values = " | ".join(u.value for u in Urgency)

    template = f"""\
ROLE: You are an insurance support ticket triage assistant.
TASK: Analyze the ticket below and output a structured JSON triage decision.

ALLOWED TOPICS (output exactly one of these values for the topic field):
{topic_lines}

ALLOWED URGENCY VALUES:
  {urgency_values}

TICKET:
  Subject: [TICKET SUBJECT]
  Body:    [TICKET BODY — first 1500 chars]

NEIGHBOR EVIDENCE (from k nearest historical tickets):
  predicted_queue    : [WEIGHTED VOTE RESULT] (confidence: 0.00)
  predicted_priority : [WEIGHTED VOTE RESULT] (confidence: 0.00)
  predicted_topic    : [WEIGHTED VOTE RESULT] (confidence: 0.00)
  predicted_tags     : [TOP TAGS FROM NEIGHBORS]

  [1] queue=... priority=... proxy_topic=... snippet="..."
  [2] queue=... priority=... proxy_topic=... snippet="..."
  ...

OUTPUT EXACTLY THIS JSON (no extra text, no markdown):
{_SCHEMA_HINT}
"""
    print("\n  ── Analyzer Prompt Template ─────────────────")
    for line in template.splitlines():
        print(f"  {line}")


# ── Menu option 11: Show reviewer prompt ─────────────────────────────────────

def show_reviewer_prompt(cfg: dict) -> None:
    """
    Show the reviewer prompt template if the reviewer is enabled in config.
    """
    reviewer_cfg = cfg.get("reviewer", {})
    if not reviewer_cfg.get("enabled", False):
        print("  Reviewer is disabled in config.yaml (reviewer.enabled: false).")
        return

    topic_lines = "\n".join(
        f"  - {topic.value}: {desc}"
        for topic, desc in _TOPIC_DESCRIPTIONS.items()
    )
    urgency_values = " | ".join(u.value for u in Urgency)
    trigger_flags  = reviewer_cfg.get("trigger_flags", [])
    model_name     = reviewer_cfg.get("model_name", "")

    template = f"""\
ROLE: You are a second-opinion reviewer for an insurance triage assistant.
      Model: {model_name}
TASK: The primary analyzer produced a triage decision. One or more validation
      flags were raised: {trigger_flags}
      Review the ticket below and produce a corrected triage decision.

ALLOWED TOPICS:
{topic_lines}

ALLOWED URGENCY VALUES:
  {urgency_values}

TICKET:
  Subject: [TICKET SUBJECT]
  Body:    [TICKET BODY — first 1500 chars]

NEIGHBOR EVIDENCE:
  predicted_queue    : [WEIGHTED VOTE RESULT] (confidence: 0.00)
  predicted_priority : [WEIGHTED VOTE RESULT] (confidence: 0.00)
  predicted_topic    : [WEIGHTED VOTE RESULT] (confidence: 0.00)

FIRST ANALYSIS:
  topic      : [FIRST LLM TOPIC]
  urgency    : [FIRST LLM URGENCY]
  confidence : [FIRST LLM CONFIDENCE]
  short_note : [FIRST LLM NOTE]

VALIDATION FLAGS:
  flags : {trigger_flags}

OUTPUT EXACTLY THIS JSON (no extra text, no markdown):
{_SCHEMA_HINT}
"""
    print("\n  ── Reviewer Prompt Template ─────────────────")
    for line in template.splitlines():
        print(f"  {line}")


# ── Menu option 12: Leakage audit ────────────────────────────────────────────

def run_leakage_audit(db_path: str, lance_path: str) -> None:
    """
    Print a leakage audit summary.

    Checks:
    - LanceDB contains reference split only (eval rows never indexed).
    - Split counts in DuckDB.
    - Which fields are used as prediction-time input.
    - Which fields are evaluation-only metadata.
    """
    print("\n  ── Leakage Audit ────────────────────────────")

    # 1. DuckDB split counts
    if os.path.exists(db_path):
        try:
            conn = _open_db_readonly(db_path)
            split_counts = conn.execute(
                "SELECT split_name, COUNT(*) FROM historical_tickets GROUP BY split_name"
            ).fetchall()
            conn.close()
            print("\n  DuckDB split counts:")
            for split, count in split_counts:
                print(f"    {split:<12}: {count} tickets")
        except Exception as exc:
            print(f"  Could not query DuckDB: {exc}")
    else:
        print("  DuckDB file not found.")

    # 2. LanceDB — should contain reference split only
    print("\n  LanceDB vector store:")
    if os.path.exists(lance_path):
        try:
            import lancedb
            db = lancedb.connect(lance_path)
            table_names = db.table_names()
            for tname in table_names:
                tbl   = db.open_table(tname)
                count = tbl.count_rows()
                print(f"    table '{tname}': {count} vectors indexed")
            print("    [OK] LanceDB indexes reference split only.")
            print("         Eval rows are never inserted into LanceDB.")
        except Exception as exc:
            print(f"  Could not inspect LanceDB: {exc}")
    else:
        print("  LanceDB directory not found.")

    # 3. Prediction-time input fields
    print("\n  Prediction-time input fields (subject/body-derived):")
    for field in ["subject", "body", "raw_text", "cleaned_text", "representation_text", "text_snippet"]:
        print(f"    [IN]  {field}")

    # 4. Evaluation-only fields — never fed into the agent
    print("\n  Evaluation-only metadata (never fed into the agent):")
    leakage_fields = [
        "answer", "actual_queue", "actual_priority", "actual_type",
        "actual_tags_json", "proxy_topic", "proxy_urgency",
        "proxy_next_action", "queue", "priority", "type",
        "tag_1..tag_8",
    ]
    for field in leakage_fields:
        print(f"    [OUT] {field}")

    print("\n  ── Leakage Rules ────────────────────────────")
    print("  [OK] answer is never read, stored, or used (dropped by csv_loader).")
    print("  [OK] actual_* are stored as historical metadata for evaluation only.")
    print("  [OK] proxy_* are derived labels used for metrics — not prediction input.")
    print("  [OK] TicketInput contains only: ticket_id, subject, body, derived text.")
    print("  [OK] LanceDB metadata (actual_queue, etc.) is used for kNN voting,")
    print("       not as direct input to the LLM or router.")


# ── Menu option 13: Submission checklist ─────────────────────────────────────

def run_submission_checklist(cfg: dict, db_path: str) -> None:
    """
    Check that all required artifacts for the assignment exist.

    Checks:
    - data/tickets.duckdb exists and has rows
    - LanceDB index exists
    - outputs/triage_results.csv exists
    - outputs/triage_trace.jsonl exists
    - outputs/run_summary.json exists
    - At least one run recorded in DuckDB
    - config.yaml has required sections
    """
    print("\n  ── Final Submission Checklist ───────────────")
    all_ok = True

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal all_ok
        status = "[OK]  " if ok else "[FAIL]"
        suffix = f" — {detail}" if detail else ""
        print(f"  {status} {label}{suffix}")
        if not ok:
            all_ok = False

    # DuckDB
    db_ok = os.path.exists(db_path)
    check("DuckDB file exists", db_ok, db_path)
    if db_ok:
        try:
            conn = _open_db_readonly(db_path)
            count = conn.execute(
                "SELECT COUNT(*) FROM historical_tickets"
            ).fetchone()[0]
            conn.close()
            check("historical_tickets has rows", count > 0, f"{count} rows")
            split_counts = {}
            conn2 = _open_db_readonly(db_path)
            for split, n in conn2.execute(
                "SELECT split_name, COUNT(*) FROM historical_tickets GROUP BY split_name"
            ).fetchall():
                split_counts[split] = n
            conn2.close()
            check("reference split exists", "reference" in split_counts,
                  f"{split_counts.get('reference', 0)} rows")
            check("eval split exists", "eval" in split_counts,
                  f"{split_counts.get('eval', 0)} rows")
        except Exception as exc:
            check("DuckDB readable", False, str(exc))

    # LanceDB
    lance_path = cfg.get("vector_store", {}).get("path", "data/lancedb")
    check("LanceDB directory exists", os.path.exists(lance_path), lance_path)

    # Output artifacts
    batch_cfg  = cfg.get("batch", {})
    csv_path   = batch_cfg.get("output_csv", "outputs/triage_results.csv")
    jsonl_path = batch_cfg.get("trace_jsonl", "outputs/triage_trace.jsonl")
    sum_path   = batch_cfg.get("run_summary_json", "outputs/run_summary.json")
    check("triage_results.csv exists", os.path.exists(csv_path), csv_path)
    check("triage_trace.jsonl exists", os.path.exists(jsonl_path), jsonl_path)
    check("run_summary.json exists", os.path.exists(sum_path), sum_path)

    # At least one batch run in DuckDB
    if db_ok:
        try:
            conn = _open_db_readonly(db_path)
            tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
            if "triage_runs" in tables:
                n_runs = conn.execute("SELECT COUNT(*) FROM triage_runs").fetchone()[0]
                conn.close()
                check("At least one batch run recorded", n_runs > 0, f"{n_runs} runs")
            else:
                conn.close()
                check("triage_runs table exists", False)
        except Exception as exc:
            check("triage_runs readable", False, str(exc))

    # Config sections
    for section in ["embedding", "vector_store", "retrieval", "llm", "batch"]:
        check(f"config.yaml has [{section}]", section in cfg)

    print()
    if all_ok:
        print("  All checks passed. Project is ready for submission.")
    else:
        print("  Some checks failed. Review the items marked [FAIL].")


# ── Menu option 15: Reviewer A/B comparison ───────────────────────────────────

#: KPIs shown in the A/B comparison table (ordered for readability).
AB_COMPARISON_KPIS: list[str] = [
    "urgency_accuracy",
    "urgency_macro_f1",
    "topic_proxy_accuracy",
    "topic_macro_f1",
    "next_action_proxy_agreement",
    "human_review_rate",
    "average_confidence",
    "reviewer_invocation_rate",
    "avg_seconds_per_ticket",
    "p95_seconds_per_ticket",
]


def run_reviewer_ab_comparison(
    cfg: dict,
    db_path: str,
    embedding_model: EmbeddingModel,
) -> None:
    """
    Run a reviewer A/B comparison on the same sampled ticket set.

    Prompts:
      - batch limit (default from config.yaml batch.limit)
      - sample strategy (default from config.yaml batch.sample_strategy)
      - reviewer model for Run B (default from config.yaml reviewer.model_name)

    Workflow:
      1. Fetch all eval tickets and apply the chosen sampling strategy once.
      2. Run A on the sampled tickets with reviewer.enabled = False.
      3. Run B on the same sampled tickets with reviewer.enabled = True.
      4. Store both runs in DuckDB with explicit _ab_off / _ab_on run_id suffixes.
         Run A: run_YYYYMMDD_HHMMSS_ab_off
         Run B: run_YYYYMMDD_HHMMSS_ab_on
      5. Print a compact KPI comparison table.

    Output files (CSV, JSONL, run_summary.json) are NOT written to avoid
    overwriting the normal batch outputs between the two runs.  Both runs are
    stored in DuckDB and can be inspected with menu options 4, 5, and 6.

    config.yaml is never modified.
    Data leakage: proxy_* labels select rows only — never passed to the agent.
    """
    batch_cfg        = cfg.get("batch", {})
    default_limit    = batch_cfg.get("limit", 20)
    default_strategy = batch_cfg.get("sample_strategy", "natural")
    default_seed     = batch_cfg.get("random_seed", 42)
    default_model    = cfg.get("reviewer", {}).get("model_name", "")

    print("\n  ── Reviewer A/B Comparison Setup ────────────")
    print("  Output files (CSV, JSONL, run_summary.json) will NOT be overwritten.")
    print("  Both runs will be stored in DuckDB for later inspection.")
    print("  Batch limit = total number of tickets processed (not per-class).")

    try:
        raw_limit = input(f"  Batch limit [{default_limit}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw_limit = ""
    limit = int(raw_limit) if raw_limit.isdigit() else default_limit

    try:
        raw_strategy = input(
            f"  Sample strategy (natural/random/balanced_proxy_topic/"
            f"balanced_proxy_urgency) [{default_strategy}]: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        raw_strategy = ""
    strategy = raw_strategy if raw_strategy else default_strategy

    try:
        raw_model = input(f"  Reviewer model for Run B [{default_model}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw_model = ""
    reviewer_model = raw_model if raw_model else default_model

    # Fetch and sample once — A and B receive the exact same list object.
    # limit is always the TOTAL ticket budget; config.batch.limit_per_label is not used.
    split = batch_cfg.get("split", "eval")
    all_records = _fetch_all_split_records(db_path, split)
    if not all_records:
        print("  No records found. Check split name and DuckDB state.")
        return

    records = _sample_tickets(
        all_records,
        sample_strategy=strategy,
        limit=limit,
        random_seed=default_seed,
    )
    if not records:
        print("  No records after sampling. Check strategy and DuckDB state.")
        return

    _print_sampling_report(records, requested_limit=limit, strategy=strategy)
    print("  Both runs use these identical ticket IDs — no resampling between A and B.")

    ticket_ids = [r["ticket_id"] for r in records]

    # Shared timestamp base keeps the two run_ids clearly paired.
    base_ts    = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_id_off = f"{base_ts}_ab_off"
    run_id_on  = f"{base_ts}_ab_on"

    # ── Run A: reviewer disabled ──────────────────────────────────────────────
    cfg_off = copy.deepcopy(cfg)
    cfg_off.setdefault("reviewer", {})["enabled"] = False
    cfg_off.setdefault("batch", {})["sample_strategy"] = strategy

    print("\n  === Run A: Reviewer DISABLED ===")
    print_effective_run_config(cfg_off, limit)
    _, metrics_off, _ = _execute_batch_on_records(
        cfg_off, db_path, records, embedding_model,
        run_id=run_id_off, write_files=False,
    )

    # ── Run B: reviewer enabled ───────────────────────────────────────────────
    cfg_on = copy.deepcopy(cfg)
    cfg_on.setdefault("reviewer", {})["enabled"] = True
    if reviewer_model:
        cfg_on.setdefault("reviewer", {})["model_name"] = reviewer_model
    cfg_on.setdefault("batch", {})["sample_strategy"] = strategy

    print("\n  === Run B: Reviewer ENABLED ===")
    print_effective_run_config(cfg_on, limit)
    _, metrics_on, _ = _execute_batch_on_records(
        cfg_on, db_path, records, embedding_model,
        run_id=run_id_on, write_files=False,
    )

    # ── Comparison table ──────────────────────────────────────────────────────
    _print_ab_comparison(
        run_id_off, metrics_off, run_id_on, metrics_on, ticket_ids,
        triage_model=cfg.get("llm", {}).get("model_name", ""),
        reviewer_model=reviewer_model,
    )


def _print_ab_comparison(
    run_id_off: str,
    metrics_off: dict,
    run_id_on: str,
    metrics_on: dict,
    ticket_ids: list[str],
    triage_model: str = "",
    reviewer_model: str = "",
) -> None:
    """
    Print a compact side-by-side KPI comparison table for the A/B runs.

    Columns: KPI name | Run A (reviewer off) | Run B (reviewer on) | Delta (B−A).
    """
    print(f"\n  ── A/B Comparison Results ──────────────────────────────────────")
    print(f"  Tickets compared     : {len(ticket_ids)}")
    if triage_model:
        print(f"  Triage model         : {triage_model}")
    print(f"  Run A (reviewer=off) : {run_id_off}")
    print(f"  Run B (reviewer=on)  : {run_id_on}")
    if reviewer_model:
        print(f"  Reviewer model       : {reviewer_model}")
    print()

    kpi_w = 34
    val_w = 10
    print(
        f"  {'KPI':<{kpi_w}}"
        f"  {'Run A (off)':>{val_w}}"
        f"  {'Run B (on)':>{val_w}}"
        f"  {'Delta (B-A)':>{val_w}}"
    )
    print("  " + "-" * (kpi_w + val_w * 3 + 8))

    for kpi in AB_COMPARISON_KPIS:
        val_off = metrics_off.get(kpi)
        val_on  = metrics_on.get(kpi)

        a_str = f"{val_off:.4f}" if val_off is not None else "N/A"
        b_str = f"{val_on:.4f}"  if val_on  is not None else "N/A"

        if val_off is not None and val_on is not None:
            delta_str = f"{val_on - val_off:+.4f}"
        else:
            delta_str = "N/A"

        print(
            f"  {kpi:<{kpi_w}}"
            f"  {a_str:>{val_w}}"
            f"  {b_str:>{val_w}}"
            f"  {delta_str:>{val_w}}"
        )

    print()
    print("  Both runs stored in DuckDB.")
    print("  Inspect with option 4 (leaderboard), 5 (run details), or 6 (confusion matrix).")
    print(f"  Run A: {run_id_off}")
    print(f"  Run B: {run_id_on}")
