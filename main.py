"""
Main entry point — Sprint 2–5F + Sprint 6A + Sprint 6B.

Sprint 2:  CSV Import, DuckDB Storage and Proxy Labels.
Sprint 3:  Embedding Model and LanceDB Vector Store.
Sprint 4:  Neighbor Retrieval and Weighted Voting.
Sprint 5E: End-to-End TicketTriageAgent Orchestration.
Sprint 5F: Runtime Modes and Rebuild Control.
Sprint 6A: Batch Processing and Output Files.
Sprint 6B: Evaluation Metrics and Run Tracking.

Prediction-time input is subject + body only.
answer is never read, stored, or used here.
"""

import json
import os
import random as _random
import time
from datetime import datetime

import duckdb
import lancedb
import yaml

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
from src.application.action_executor import ActionExecutor
from src.application.agent import TicketTriageAgent
from src.application.analyzer import LocalLLMAnalyzer
from src.application.reviewer import ConditionalLLMReviewer
from src.application.batch_runner import BatchRunner
from src.application.metrics import compute_evaluation_metrics, compute_timing_metrics, confusion_counts
from src.application.neighbor_retriever import NeighborRetriever
from src.application.router import TriageRouter
from src.application.validator import TriageValidator
from src.domain.models import TicketInput
from src.infrastructure.csv_loader import collect_tags, load_csv
from src.infrastructure.csv_writer import write_csv_rows
from src.infrastructure.duckdb_repository import DuckDBRepository
from src.infrastructure.embedding_model import EmbeddingModel
from src.infrastructure.evaluation_repository import EvaluationRepository
from src.infrastructure.lancedb_ticket_store import LanceDBTicketStore
from src.infrastructure.llm_client import OllamaClient
from src.infrastructure.trace_writer import write_jsonl

CSV_PATH    = "data/dataset-tickets-multi-lang3-4k.csv"
DB_PATH     = "data/tickets.duckdb"
CONFIG_PATH = "config.yaml"

# Set to True to print raw LLM responses during smoke runs.
DEBUG_LLM = False


# ── Data preparation helpers ───────────────────────────────────────────────────

def build_rows(df) -> list[dict]:
    """
    Convert each DataFrame row into the dict shape expected by the repository.

    Text fields computed here:
      raw_text            = build_raw_text(subject, body)
      cleaned_text        = normalize_text(raw_text)
      representation_text = build_representation_text(subject, body)
      text_snippet        = make_text_snippet(raw_text)

    Proxy labels derived from historical metadata (queue + tags):
      proxy_topic, proxy_topic_source  via map_proxy_topic
      proxy_urgency                    via map_proxy_urgency
      proxy_next_action                via map_proxy_next_action

    answer is not in df (dropped by load_csv) so source_row_json is leak-free.
    """
    rows = []
    for i, row in df.iterrows():
        subject = str(row.get("subject", "")).strip()
        body    = str(row.get("body",    "")).strip()

        raw_text            = build_raw_text(subject, body)
        cleaned_text        = normalize_text(raw_text)
        representation_text = build_representation_text(subject, body)
        text_snippet        = make_text_snippet(raw_text)

        tags = collect_tags(row)

        proxy_topic, proxy_topic_source = map_proxy_topic(
            queue=str(row.get("queue", "")),
            tags=tags,
        )
        proxy_urgency     = map_proxy_urgency(str(row.get("priority", "")))
        proxy_next_action = map_proxy_next_action(proxy_topic)

        # source_row_json stores the original CSV row for auditability.
        # answer has already been dropped by load_csv — no leakage risk.
        source_row = {col: str(row[col]) for col in df.columns}

        rows.append({
            "_row_index":          i,
            "subject":             subject,
            "body":                body,
            "raw_text":            raw_text,
            "cleaned_text":        cleaned_text,
            "representation_text": representation_text,
            "text_snippet":        text_snippet,
            "actual_queue":        str(row.get("queue",    "")),
            "actual_priority":     str(row.get("priority", "")),
            "actual_type":         str(row.get("type",     "")),
            "actual_tags_json":    json.dumps(tags),
            "language":            str(row.get("language", "")),
            "proxy_topic":         proxy_topic,
            "proxy_urgency":       proxy_urgency,
            "proxy_next_action":   proxy_next_action,
            "proxy_topic_source":  proxy_topic_source,
            "source_row_json":     json.dumps(source_row),
        })
    return rows


def _print_sample(sample: list[dict]) -> None:
    """Print a small table of sample rows to stdout."""
    cols = [
        "ticket_id", "split_name", "subject",
        "actual_queue", "actual_priority",
        "proxy_topic", "proxy_urgency", "proxy_next_action", "proxy_topic_source",
    ]
    widths = {c: max(len(c), max((len(str(r[c])) for r in sample), default=0)) for c in cols}

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c]    for c in cols)
    print(header)
    print(sep)
    for r in sample:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


# ── DuckDB query helpers ───────────────────────────────────────────────────────

def _fetch_reference_rows(db_path: str) -> list[dict]:
    """
    Read all reference-split tickets from DuckDB.

    Returns only the columns needed to build LanceDB rows.
    Eval rows are intentionally excluded — they must not be indexed.
    """
    repo = DuckDBRepository(db_path)
    result = repo.conn.execute(
        """
        SELECT
            ticket_id,
            split_name,
            representation_text,
            text_snippet,
            actual_queue,
            actual_priority,
            actual_type,
            actual_tags_json,
            proxy_topic,
            proxy_urgency,
            proxy_next_action,
            proxy_topic_source
        FROM historical_tickets
        WHERE split_name = 'reference'
        ORDER BY ticket_id
        """
    ).fetchall()
    columns = [
        "ticket_id", "split_name", "representation_text", "text_snippet",
        "actual_queue", "actual_priority", "actual_type", "actual_tags_json",
        "proxy_topic", "proxy_urgency", "proxy_next_action", "proxy_topic_source",
    ]
    rows = [dict(zip(columns, row)) for row in result]
    repo.close()
    return rows


def _fetch_one_eval_row(db_path: str) -> dict | None:
    """
    Read one eval-split ticket from DuckDB for the smoke search.

    Returns a single dict or None if no eval rows exist.
    """
    repo = DuckDBRepository(db_path)
    result = repo.conn.execute(
        """
        SELECT ticket_id, subject, representation_text
        FROM historical_tickets
        WHERE split_name = 'eval'
        LIMIT 1
        """
    ).fetchone()
    repo.close()
    if result is None:
        return None
    return {"ticket_id": result[0], "subject": result[1], "representation_text": result[2]}


def _fetch_one_eval_ticket_input(db_path: str) -> TicketInput | None:
    """
    Read one eval-split ticket from DuckDB and return it as a TicketInput.

    Returns None if no eval rows exist.
    All text fields are read from DuckDB so no preprocessing is repeated here.
    """
    repo   = DuckDBRepository(db_path)
    result = repo.conn.execute(
        """
        SELECT ticket_id, subject, body, raw_text, cleaned_text,
               representation_text, text_snippet
        FROM historical_tickets
        WHERE split_name = 'eval'
        LIMIT 1
        """
    ).fetchone()
    repo.close()
    if result is None:
        return None
    cols = [
        "ticket_id", "subject", "body", "raw_text", "cleaned_text",
        "representation_text", "text_snippet",
    ]
    return TicketInput(**dict(zip(cols, result)))


# ── Sprint 6A: Agent factory ───────────────────────────────────────────────────

def _build_agent(cfg: dict, embedding_model: EmbeddingModel) -> TicketTriageAgent:
    """
    Construct a TicketTriageAgent from config and a pre-loaded embedding model.

    All five pipeline components are built here and injected into the agent.
    When reviewer.enabled is true in config, a ConditionalLLMReviewer is also
    constructed and injected.  When disabled, reviewer=None is passed (no-op).

    Reused by run_sprint5e and _run_batch so the model is never loaded twice.
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
    store = LanceDBTicketStore(path=vs_cfg["path"], table_name=vs_cfg["table_name"])
    retriever = NeighborRetriever(
        embedding_model=embedding_model,
        ticket_store=store,
        top_k=ret_cfg.get("top_k", 5),
    )
    analyzer = LocalLLMAnalyzer(
        llm_client=llm_client,
        max_retries=llm_cfg.get("max_retries", 1),
        debug=DEBUG_LLM,
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


# ── Sprint 6A/6B: Run summary ─────────────────────────────────────────────────

def _write_run_summary(
    path: str,
    rows: list[dict],
    cfg: dict,
    run_id: str = "",
    runtime_seconds: float = 0.0,
    eval_metrics: dict | None = None,
) -> None:
    """
    Compute distribution statistics from result rows and write run_summary.json.

    Sprint 6A: distributions + operational rates.
    Sprint 6B: adds run_id, runtime_seconds, evaluation_metrics.

    Distributions count how many tickets landed in each topic / urgency /
    next_action bucket.  Rates are computed as fractions in [0, 1].
    """
    import json as _json

    total = len(rows)

    topic_dist: dict[str, int]       = {}
    urgency_dist: dict[str, int]     = {}
    next_action_dist: dict[str, int] = {}
    human_review_count = 0
    missing_info_count = 0

    for row in rows:
        topic       = str(row.get("topic", ""))
        urgency     = str(row.get("urgency", ""))
        next_action = str(row.get("next_action", ""))

        topic_dist[topic]             = topic_dist.get(topic, 0) + 1
        urgency_dist[urgency]         = urgency_dist.get(urgency, 0) + 1
        next_action_dist[next_action] = next_action_dist.get(next_action, 0) + 1

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
        "sample_strategy":          batch_cfg.get("sample_strategy", "natural"),
        "runtime_seconds":          round(runtime_seconds, 2),
        "topic_distribution":       topic_dist,
        "urgency_distribution":     urgency_dist,
        "next_action_distribution": next_action_dist,
        "human_review_rate":        round(human_review_count / total, 4) if total else 0.0,
        "missing_info_rate":        round(missing_info_count  / total, 4) if total else 0.0,
        "evaluation_metrics":       eval_metrics or {},
    }

    import os as _os
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(summary, f, indent=2, ensure_ascii=False)


# ── Mini Sprint: Balanced Evaluation Sampling ─────────────────────────────────
#
# proxy_topic and proxy_urgency are used only to select which eval rows enter
# the batch.  They are never passed to TicketInput or the model.

def _sample_tickets(
    rows: list[dict],
    sample_strategy: str,
    limit: int,
    random_seed: int,
    limit_per_label: int,
) -> list[dict]:
    """
    Apply a sampling strategy to a list of ticket rows fetched from DuckDB.

    Strategies:
      natural              — first <limit> rows (deterministic ORDER BY ticket_id)
      random               — random <limit> rows, reproducible with random_seed
      balanced_proxy_topic   — up to limit_per_label rows per proxy_topic class
      balanced_proxy_urgency — up to limit_per_label rows per proxy_urgency class

    Data leakage note:
      Proxy labels select rows only.  The returned rows still contain proxy_*
      and actual_* for post-prediction evaluation, but BatchRunner builds
      TicketInput exclusively from text fields (subject, body, etc.).
    """
    if sample_strategy == "natural":
        return rows[:limit]

    if sample_strategy == "random":
        rng = _random.Random(random_seed)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        return shuffled[:limit]

    if sample_strategy == "balanced_proxy_topic":
        return _balanced_sample(rows, "proxy_topic", limit_per_label, random_seed)

    if sample_strategy == "balanced_proxy_urgency":
        return _balanced_sample(rows, "proxy_urgency", limit_per_label, random_seed)

    raise ValueError(
        f"Unknown sample_strategy: {sample_strategy!r}. "
        "Supported: natural, random, balanced_proxy_topic, balanced_proxy_urgency"
    )


def _balanced_sample(
    rows: list[dict],
    label_col: str,
    limit_per_label: int,
    random_seed: int,
) -> list[dict]:
    """
    Group rows by label_col and sample up to limit_per_label rows per class.

    Within each class, rows are shuffled with random_seed for reproducibility.
    If a class has fewer than limit_per_label rows, all are taken and a warning
    is printed.  No synthetic rows are created; no class is forced.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        label = str(row.get(label_col) or "Unknown")
        groups.setdefault(label, []).append(row)

    rng = _random.Random(random_seed)
    sampled: list[dict] = []
    for label in sorted(groups):
        group_rows = list(groups[label])
        rng.shuffle(group_rows)
        if len(group_rows) < limit_per_label:
            print(
                f"  WARNING: balanced sampling — class '{label}' has only "
                f"{len(group_rows)} rows (< limit_per_label={limit_per_label}), "
                "taking all available."
            )
        sampled.extend(group_rows[:limit_per_label])

    return sampled


def _print_sampling_report(
    rows: list[dict],
    sample_strategy: str,
    split: str,
) -> None:
    """
    Print sampling summary before batch processing begins.

    Shows strategy, split, total fetched, and the class distributions of
    proxy_topic and proxy_urgency in the sampled batch.  These distributions
    help verify that balanced sampling worked as intended.
    """
    topic_dist: dict[str, int] = {}
    urgency_dist: dict[str, int] = {}
    for row in rows:
        t = str(row.get("proxy_topic")   or "Unknown")
        u = str(row.get("proxy_urgency") or "Unknown")
        topic_dist[t]   = topic_dist.get(t, 0) + 1
        urgency_dist[u] = urgency_dist.get(u, 0) + 1

    print(f"\n--- Batch Sampling ---")
    print(f"  sample_strategy : {sample_strategy}")
    print(f"  split           : {split}")
    print(f"  tickets fetched : {len(rows)}")
    print("  proxy_topic distribution:")
    for label, count in sorted(topic_dist.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}")
    print("  proxy_urgency distribution:")
    for label, count in sorted(urgency_dist.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}")


# ── Sprint 6A/6B: Batch run ────────────────────────────────────────────────────

def _run_batch(cfg: dict, embedding_model: EmbeddingModel) -> None:
    """
    Process batch.limit eval tickets and write output files.

    Sprint 6A steps:
      1. Fetch ticket records from DuckDB (eval split, up to limit).
      2. Build the TicketTriageAgent from config.
      3. Run BatchRunner over all records.
      4. Write triage_results.csv.
      5. Write triage_trace.jsonl.
      6. Write run_summary.json.

    Sprint 6B additions:
      7. Generate run_id (run_YYYYMMDD_HHMMSS).
      8. Time the batch run.
      9. Compute evaluation KPIs from result rows.
      10. Enrich run_summary.json with run_id, runtime_seconds, evaluation_metrics.
      11. Store run metadata, predictions, KPIs, and confusion counts in DuckDB.
      12. Print Sprint 6B evaluation summary.
    """
    batch_cfg = cfg["batch"]
    split           = batch_cfg["split"]
    limit           = batch_cfg["limit"]
    csv_path        = batch_cfg["output_csv"]
    jsonl_path      = batch_cfg["trace_jsonl"]
    summary_path    = batch_cfg["run_summary_json"]
    sample_strategy = batch_cfg.get("sample_strategy", "natural")
    random_seed     = batch_cfg.get("random_seed", 42)
    stratify_by     = batch_cfg.get("stratify_by", "proxy_topic")
    limit_per_label = batch_cfg.get("limit_per_label", 50)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    print(f"\n--- Sprint 6A: Batch Processing (split={split}, strategy={sample_strategy}) ---")
    print(f"  run_id: {run_id}")

    t0 = time.time()

    # Fetch all rows for the split, then apply the configured sampling strategy.
    # proxy_* labels are used only to select rows — never passed to the model.
    repo = DuckDBRepository(DB_PATH)
    all_records = repo.fetch_split_tickets(split)
    repo.close()
    print(f"  total {split} rows in DuckDB: {len(all_records)}")

    records = _sample_tickets(
        all_records,
        sample_strategy=sample_strategy,
        limit=limit,
        random_seed=random_seed,
        limit_per_label=limit_per_label,
    )
    _print_sampling_report(records, sample_strategy, split)

    if not records:
        print("  No records found. Check split name and DuckDB state.")
        return

    log_decisions       = batch_cfg.get("log_decisions", True)
    log_reviewer_events = batch_cfg.get("log_reviewer_events", True)

    agent  = _build_agent(cfg, embedding_model)
    runner = BatchRunner(
        agent,
        log_decisions=log_decisions,
        log_reviewer_events=log_reviewer_events,
    )

    print("  Processing tickets …")
    result_rows = runner.process_tickets(records, trace_path=jsonl_path)

    runtime_seconds = time.time() - t0
    print(f"  Done. {len(result_rows)} tickets processed in {runtime_seconds:.1f}s.")

    # ── Sprint 6B: Compute evaluation metrics and timing aggregates ───────────
    eval_metrics = compute_evaluation_metrics(result_rows)

    ticket_seconds = [
        float(r.get("total_ticket_seconds", 0.0)) for r in result_rows
    ]
    timing_metrics = compute_timing_metrics(ticket_seconds)
    timing_metrics["runtime_seconds_total"] = round(runtime_seconds, 3)
    eval_metrics.update(timing_metrics)

    # ── Write output files ────────────────────────────────────────────────────
    # JSONL trace was written incrementally by process_tickets.
    # CSV and summary are written once at the end (need all rows).
    write_csv_rows(csv_path, result_rows)
    _write_run_summary(
        summary_path,
        result_rows,
        cfg,
        run_id=run_id,
        runtime_seconds=runtime_seconds,
        eval_metrics=eval_metrics,
    )

    # ── Sprint 6B: Store evaluation data in DuckDB ────────────────────────────
    llm_cfg = cfg.get("llm", {})
    ret_cfg = cfg.get("retrieval", {})

    reviewer_cfg     = cfg.get("reviewer", {})
    reviewer_enabled = reviewer_cfg.get("enabled", False)
    reviewer_model   = reviewer_cfg.get("model_name", "") if reviewer_enabled else ""

    run_metadata = {
        "run_id":            run_id,
        "created_at":        datetime.now(),
        "model_name":        llm_cfg.get("model_name", ""),
        "embedding_model":   cfg.get("embedding", {}).get("model_name", ""),
        "dataset_path":      cfg.get("input_path", ""),
        "split_name":        split,
        "limit_n":           limit,
        "top_k":             ret_cfg.get("top_k", 0),
        "temperature":       llm_cfg.get("temperature", 0.0),
        "max_retries":       llm_cfg.get("max_retries", 0),
        "runtime_seconds":   runtime_seconds,
        "config_json":       json.dumps(cfg),
        "sample_strategy":   sample_strategy,
        "stratify_by":       stratify_by,
        "random_seed":       random_seed,
        "limit_per_label":   limit_per_label,
        "triage_model":      llm_cfg.get("model_name", ""),
        "reviewer_enabled":  reviewer_enabled,
        "reviewer_model":    reviewer_model,
    }

    eval_repo = EvaluationRepository(DB_PATH)
    eval_repo.create_tables()
    eval_repo.insert_run(run_metadata)
    eval_repo.insert_predictions(run_id, result_rows)
    eval_repo.insert_metrics(run_id, eval_metrics)

    # Confusion counts for three prediction targets
    eval_repo.insert_confusion_matrix(
        run_id,
        "urgency",
        confusion_counts(result_rows, "urgency", "proxy_urgency"),
    )
    eval_repo.insert_confusion_matrix(
        run_id,
        "topic_proxy",
        confusion_counts(result_rows, "topic", "proxy_topic"),
    )
    eval_repo.insert_confusion_matrix(
        run_id,
        "next_action_proxy",
        confusion_counts(result_rows, "next_action", "proxy_next_action"),
    )
    eval_repo.close()

    # ── Sprint 6B: Print evaluation summary ───────────────────────────────────
    print(f"\n--- Sprint 6B: Evaluation ---")
    print(f"  run_id                    : {run_id}")
    print(f"  urgency_accuracy          : {eval_metrics.get('urgency_accuracy', 0):.4f}")
    print(f"  topic_proxy_accuracy      : {eval_metrics.get('topic_proxy_accuracy', 0):.4f}")
    print(f"  next_action_proxy_agreement: {eval_metrics.get('next_action_proxy_agreement', 0):.4f}")
    print(f"  human_review_rate         : {eval_metrics.get('human_review_rate', 0):.4f}")
    print(f"  missing_info_rate         : {eval_metrics.get('missing_info_rate', 0):.4f}")
    print(f"  average_confidence        : {eval_metrics.get('average_confidence', 0):.4f}")
    print(f"\n  Stored evaluation tables in {DB_PATH}")

    print(f"\n  output_csv       : {csv_path}")
    print(f"  trace_jsonl      : {jsonl_path}")
    print(f"  run_summary_json : {summary_path}")
    print(f"  tickets processed: {len(result_rows)}")


# ── Sprint 5F: Runtime existence checks ───────────────────────────────────────

def _duckdb_exists(db_path: str) -> bool:
    """
    Return True if the DuckDB file exists, contains historical_tickets, and has rows.

    Using duckdb directly (not DuckDBRepository) so we never run CREATE TABLE
    during a mere existence check — that would create an empty table and mask
    the missing-database case.
    """
    if not os.path.exists(db_path):
        return False
    try:
        conn = duckdb.connect(db_path, read_only=True)
        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        if "historical_tickets" not in tables:
            conn.close()
            return False
        count = conn.execute("SELECT COUNT(*) FROM historical_tickets").fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def _lancedb_table_exists(path: str, table_name: str) -> bool:
    """Return True if the LanceDB directory exists and the named table can be opened."""
    if not os.path.exists(path):
        return False
    try:
        db = lancedb.connect(path)
        db.open_table(table_name)
        return True
    except Exception:
        return False


# ── Sprint 5F: Embedding model (shared) ───────────────────────────────────────

def _load_embedding_model(cfg: dict) -> EmbeddingModel:
    """
    Build and return the EmbeddingModel from config.

    Intended to be called once in main() and the result passed to all phases
    that need it — avoids loading the model multiple times.
    """
    emb_cfg = cfg["embedding"]
    print(f"\nLoading embedding model: {emb_cfg['model_name']}")
    model = EmbeddingModel(
        model_name=emb_cfg["model_name"],
        device=emb_cfg.get("device", "auto"),
        normalize_embeddings=emb_cfg.get("normalize_embeddings", True),
    )
    print(f"  device        : {model.device}")
    print(f"  embedding dim : {model.get_dimension()}")
    return model


# ── Sprint 5F: DuckDB phases ───────────────────────────────────────────────────

def _import_csv_to_duckdb() -> None:
    """
    Sprint 2 path: load CSV and write all tickets into DuckDB.

    Drops and recreates the historical_tickets table, then inserts all rows
    with a deterministic 80/20 reference/eval split.
    """
    print("\nDuckDB (rebuild):")
    print(f"  loading CSV: {CSV_PATH}")
    df = load_csv(CSV_PATH)
    print(f"  loaded {len(df)} rows")

    rows = build_rows(df)

    print(f"  importing {len(rows)} tickets into DuckDB: {DB_PATH}")
    repo = DuckDBRepository(DB_PATH, recreate=True)
    repo.insert_tickets(rows, eval_fraction=0.2, random_seed=42)
    repo.close()

    repo = DuckDBRepository(DB_PATH)
    split_counts = repo.count_by_split()
    total = sum(split_counts.values())
    topic_counts = repo.count_by_proxy_topic()
    sample = repo.sample_rows(n=5)
    repo.close()

    print(f"  total rows: {total}")
    for split, count in sorted(split_counts.items()):
        print(f"    {split}: {count}")
    print("  row counts by proxy_topic:")
    for topic, count in topic_counts.items():
        print(f"    {topic}: {count}")
    if sample:
        print("\nSample rows:")
        _print_sample(sample)


def _connect_existing_duckdb(db_path: str) -> None:
    """
    Reuse path: verify the existing DuckDB database and print row counts.

    Raises RuntimeError with a clear message if the database is missing or empty.
    """
    if not _duckdb_exists(db_path):
        raise RuntimeError(
            "DuckDB database not found. Set runtime.rebuild_duckdb=true for the first run."
        )
    repo = DuckDBRepository(db_path)
    split_counts = repo.count_by_split()
    total = sum(split_counts.values())
    topic_counts = repo.count_by_proxy_topic()
    repo.close()

    print("\nDuckDB:")
    print(f"  using existing database: {db_path}")
    print(f"  total rows: {total}")
    for split, count in sorted(split_counts.items()):
        print(f"    {split}: {count}")
    print("  row counts by proxy_topic:")
    for topic, count in topic_counts.items():
        print(f"    {topic}: {count}")


# ── Sprint 5F: LanceDB phases ─────────────────────────────────────────────────

def _rebuild_lancedb_index(cfg: dict, model: EmbeddingModel) -> None:
    """
    Sprint 3 rebuild path: embed all reference tickets and write them to LanceDB.

    Drops and recreates the ticket_embeddings table. Asserts that the written
    row count matches the number of reference rows fetched from DuckDB.
    """
    emb_cfg = cfg["embedding"]
    vs_cfg  = cfg["vector_store"]

    print("\n--- Sprint 3: Building LanceDB Index ---")
    ref_rows = _fetch_reference_rows(DB_PATH)
    print(f"  reference rows: {len(ref_rows)}")

    if not ref_rows:
        print("  No reference rows found. Run with rebuild_duckdb=true first.")
        return

    print("  Embedding reference tickets (passage prefix) …")
    texts = [row["representation_text"] for row in ref_rows]
    vectors = model.encode_passages(texts, batch_size=emb_cfg.get("batch_size", 32))

    store = LanceDBTicketStore(path=vs_cfg["path"], table_name=vs_cfg["table_name"])

    lancedb_rows = []
    for row, vec in zip(ref_rows, vectors):
        entry = dict(row)
        entry["vector"] = vec.tolist()
        lancedb_rows.append(entry)

    print(f"  Writing {len(lancedb_rows)} rows to LanceDB: {vs_cfg['path']}")
    store.recreate_table(lancedb_rows)

    indexed_count = store.count()
    print(f"  LanceDB table '{vs_cfg['table_name']}' count: {indexed_count}")

    assert indexed_count == len(ref_rows), (
        f"Count mismatch: LanceDB has {indexed_count} rows but expected {len(ref_rows)}"
    )

    print("\nLanceDB:")
    print(f"  table: {vs_cfg['path']} / {vs_cfg['table_name']}")
    print(f"  table count: {indexed_count}")


def _connect_existing_lancedb(cfg: dict) -> None:
    """
    Reuse path: verify the existing LanceDB table and print row count.

    Raises RuntimeError with a clear message if the table is missing.
    """
    vs_cfg = cfg["vector_store"]
    path       = vs_cfg["path"]
    table_name = vs_cfg["table_name"]

    if not _lancedb_table_exists(path, table_name):
        raise RuntimeError(
            "LanceDB table not found. Set runtime.rebuild_lancedb=true for the first run."
        )
    store = LanceDBTicketStore(path=path, table_name=table_name)
    count = store.count()

    print("\nLanceDB:")
    print(f"  using existing table: {path} / {table_name}")
    print(f"  table count: {count}")


# ── Sprint 3 smoke search ──────────────────────────────────────────────────────

def _run_sprint3_smoke(cfg: dict, model: EmbeddingModel) -> None:
    """
    Sprint 3 smoke: embed one eval ticket and print the top-k neighbors from LanceDB.

    Does not rebuild the index. Requires an existing LanceDB table.
    """
    vs_cfg  = cfg["vector_store"]
    ret_cfg = cfg["retrieval"]

    print("\n--- Sprint 3: Smoke Search ---")
    store = LanceDBTicketStore(path=vs_cfg["path"], table_name=vs_cfg["table_name"])

    eval_row = _fetch_one_eval_row(DB_PATH)
    if eval_row is None:
        print("  No eval rows found. Skipping smoke search.")
        return

    print(f"  Eval ticket id: {eval_row['ticket_id']}")
    print(f"  Eval subject  : {eval_row['subject'][:80]}")

    query_vectors = model.encode_queries([eval_row["representation_text"]])
    query_vec = query_vectors[0]
    top_k = ret_cfg.get("top_k", 5)
    neighbors = store.search(query_vec, top_k=top_k)

    print(f"\n  Top-{top_k} neighbors:")
    for rank, nb in enumerate(neighbors, start=1):
        dist = nb.get("_distance", "n/a")
        dist_str = f"{dist:.4f}" if isinstance(dist, float) else str(dist)
        print(
            f"  [{rank}] ticket_id={nb['ticket_id']}"
            f"  queue={nb['actual_queue']}"
            f"  priority={nb['actual_priority']}"
            f"  proxy_topic={nb['proxy_topic']}"
            f"  distance={dist_str}"
        )


# ── Sprint 4 smoke ─────────────────────────────────────────────────────────────

def _run_sprint4_smoke(cfg: dict, model: EmbeddingModel) -> None:
    """
    Sprint 4 smoke: retrieve top-k neighbors for one eval ticket and print the
    weighted queue/priority prediction.

    Does not rebuild the index. Requires an existing LanceDB table.
    """
    vs_cfg  = cfg["vector_store"]
    ret_cfg = cfg["retrieval"]

    print("\n--- Sprint 4: Retrieval & Weighted Voting ---")
    store = LanceDBTicketStore(path=vs_cfg["path"], table_name=vs_cfg["table_name"])

    eval_row = _fetch_one_eval_row(DB_PATH)
    if eval_row is None:
        print("  No eval rows found. Skipping Sprint 4 smoke run.")
        return

    top_k = ret_cfg.get("top_k", 5)
    retriever = NeighborRetriever(
        embedding_model=model,
        ticket_store=store,
        top_k=top_k,
    )

    prediction = retriever.retrieve_and_predict(
        ticket_id=eval_row["ticket_id"],
        representation_text=eval_row["representation_text"],
    )

    snippet = eval_row["representation_text"][:120].replace("\n", " ")
    print(f"\nEval ticket:")
    print(f"  ticket_id     : {eval_row['ticket_id']}")
    print(f"  subject       : {eval_row['subject'][:80]}")
    print(f"  representation: {snippet} …")

    print(f"\nTop-{top_k} neighbor evidence:")
    for rank, nb in enumerate(prediction.neighbors, start=1):
        print(
            f"  [{rank}]"
            f"  ticket_id={nb.ticket_id}"
            f"  dist={nb.distance:.4f}"
            f"  sim={nb.similarity:.4f}"
            f"  queue={nb.actual_queue}"
            f"  priority={nb.actual_priority}"
            f"  proxy_topic={nb.proxy_topic}"
            f"  snippet={nb.text_snippet[:40]!r}"
        )

    print(f"\nWeighted prediction:")
    print(f"  predicted_queue       : {prediction.predicted_queue}")
    print(f"  queue_confidence      : {prediction.queue_confidence:.3f}")
    print(f"  predicted_priority    : {prediction.predicted_priority}")
    print(f"  priority_confidence   : {prediction.priority_confidence:.3f}")
    print(f"  predicted_proxy_topic : {prediction.predicted_proxy_topic}")
    print(f"  proxy_topic_confidence: {prediction.proxy_topic_confidence:.3f}")
    print(f"  predicted_tags        : {prediction.predicted_tags}")


def run_sprint5e(
    cfg: dict,
    debug: bool = False,
    embedding_model: EmbeddingModel | None = None,
) -> None:
    """
    Sprint 5E smoke run: process one eval ticket end-to-end via TicketTriageAgent.

    Builds all five components from config, constructs the agent, fetches one
    eval ticket from DuckDB, calls process_ticket, and prints the TriageResult.

    Accepts an optional pre-loaded embedding model to avoid reloading it.

    Requires:
      - LanceDB index must already exist.
      - Ollama must be running locally with the configured model.
    """
    print("\n--- Sprint 5E: End-to-End TicketTriageAgent ---")

    emb_cfg = cfg["embedding"]

    if embedding_model is None:
        embedding_model = EmbeddingModel(
            model_name=emb_cfg["model_name"],
            device=emb_cfg.get("device", "auto"),
            normalize_embeddings=emb_cfg.get("normalize_embeddings", True),
        )

    agent  = _build_agent(cfg, embedding_model)
    ticket = _fetch_one_eval_ticket_input(DB_PATH)
    if ticket is None:
        print("  No eval ticket found. Skipping Sprint 5E smoke run.")
        return

    print(f"  ticket_id : {ticket.ticket_id}")
    print(f"  subject   : {ticket.subject[:80]}")

    try:
        result = agent.process_ticket(ticket)
    except Exception as exc:
        print(f"  ERROR: Agent failed: {exc}")
        print("  Is Ollama running? Try: ollama serve")
        return

    print(f"\n  ticket_id            : {result.ticket_id}")
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
        print(f"  action_note          : {result.action_result.action_note}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Sprint 5F runtime control.

    Reads the runtime section from config.yaml to decide:
      - whether to rebuild DuckDB from CSV, or reuse the existing database
      - whether to rebuild the LanceDB index, or reuse the existing table
      - whether to run the Sprint 3/4 smoke search
      - whether to run the Sprint 5A–5E end-to-end smoke

    The EmbeddingModel is built once and shared across all phases that need it.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rt = cfg.get("runtime", {})
    rebuild_duckdb       = rt.get("rebuild_duckdb", True)
    rebuild_lancedb      = rt.get("rebuild_lancedb", True)
    run_smoke_search     = rt.get("run_smoke_search", True)
    run_end_to_end_smoke = rt.get("run_end_to_end_smoke", True)

    print("Runtime:")
    print(f"  rebuild_duckdb      : {str(rebuild_duckdb).lower()}")
    print(f"  rebuild_lancedb     : {str(rebuild_lancedb).lower()}")
    print(f"  run_smoke_search    : {str(run_smoke_search).lower()}")
    print(f"  run_end_to_end_smoke: {str(run_end_to_end_smoke).lower()}")

    # --- DuckDB: rebuild from CSV or reuse existing ---
    if rebuild_duckdb:
        _import_csv_to_duckdb()
    else:
        _connect_existing_duckdb(DB_PATH)

    # Shared embedding model — built once, reused across all phases that need it.
    embedding_model: EmbeddingModel | None = None

    # --- LanceDB: rebuild index or reuse existing ---
    if rebuild_lancedb:
        embedding_model = _load_embedding_model(cfg)
        _rebuild_lancedb_index(cfg, embedding_model)
    else:
        _connect_existing_lancedb(cfg)

    # --- Sprint 3 + 4 smoke search ---
    if run_smoke_search:
        if embedding_model is None:
            embedding_model = _load_embedding_model(cfg)
        _run_sprint3_smoke(cfg, embedding_model)
        _run_sprint4_smoke(cfg, embedding_model)

    # --- Sprint 5E end-to-end single-ticket demo ---
    if run_end_to_end_smoke:
        if embedding_model is None:
            embedding_model = _load_embedding_model(cfg)
        run_sprint5e(cfg, debug=DEBUG_LLM, embedding_model=embedding_model)

    # --- Sprint 6A: Batch processing ---
    batch_cfg = cfg.get("batch", {})
    if batch_cfg.get("enabled", False):
        if embedding_model is None:
            embedding_model = _load_embedding_model(cfg)
        _run_batch(cfg, embedding_model)


if __name__ == "__main__":
    main()
