"""
Sprint 6A — Batch Runner.
Sprint 6B (timing) — Per-ticket step timings via process_ticket_timed.
Sprint 6C.5 (reviewer) — Reviewer metadata fields added to result rows.
Sprint 6D (logging) — Compact decision line and reviewer before/after block.

BatchRunner iterates over a list of ticket records fetched from DuckDB,
builds a TicketInput for each record from text fields only, calls the
TicketTriageAgent, and returns a list of flat result dicts ready for
CSV/JSONL export.

Timing:
  If the agent exposes process_ticket_timed, BatchRunner calls it and
  includes the timing dict in each result row.  If not (e.g. in tests
  using a simple FakeAgent), it falls back to process_ticket and the
  timing fields default to 0.0.

Data leakage note:
  TicketInput is built exclusively from subject/body-derived fields:
    ticket_id, subject, body, raw_text, cleaned_text,
    representation_text, text_snippet.
  actual_* and proxy_* columns are passed through to the result row as
  output/evaluation metadata only.  They never enter the agent.
"""

from src.application.agent import TicketTriageAgent
from src.domain.models import TicketInput, TriageResult
from src.infrastructure.trace_writer import append_jsonl_record, init_jsonl


class BatchRunner:
    """
    Process a list of ticket records through the TicketTriageAgent.

    Parameters
    ----------
    agent:
        A fully constructed TicketTriageAgent.
    log_decisions:
        Print a compact decision line after each ticket is processed.
        Format: Processing ticket N/total: <id> -> topic=..., urgency=...,
        action=..., final_conf=..., human_review=yes/no, reviewer=yes/no
    log_reviewer_events:
        When the conditional reviewer is used, print an indented before/after
        summary showing what changed.  Only active when log_decisions is True.
    """

    def __init__(
        self,
        agent: TicketTriageAgent,
        log_decisions: bool = True,
        log_reviewer_events: bool = True,
    ) -> None:
        self.agent = agent
        self.log_decisions = log_decisions
        self.log_reviewer_events = log_reviewer_events

    def process_tickets(
        self,
        ticket_records: list[dict],
        trace_path: str | None = None,
    ) -> list[dict]:
        """
        Process each record and return one flat result dict per ticket.

        If trace_path is provided, the JSONL file is created/truncated before
        the first ticket and one record is appended + flushed immediately after
        each ticket.  This lets another terminal observe progress in real time
        (e.g. Get-Content outputs/triage_trace.jsonl -Tail 5).

        Progress is printed to stdout as:
          Processing ticket N/total: <ticket_id>

        Parameters
        ----------
        ticket_records:
            Rows from DuckDB historical_tickets (eval split).
            Must include at minimum: ticket_id, subject, body, raw_text,
            cleaned_text, representation_text, text_snippet.
            May also include actual_* and proxy_* for output metadata.
        trace_path:
            Optional path for the streaming JSONL trace file.
            When None, no streaming write occurs.

        Returns
        -------
        list[dict]
            One dict per ticket with prediction fields and evaluation metadata.
        """
        total = len(ticket_records)

        if trace_path is not None:
            init_jsonl(trace_path)

        results = []
        for i, record in enumerate(ticket_records, start=1):
            ticket_id = record["ticket_id"]

            ticket = TicketInput(
                ticket_id=ticket_id,
                subject=record["subject"],
                body=record["body"],
                raw_text=record["raw_text"],
                cleaned_text=record["cleaned_text"],
                representation_text=record["representation_text"],
                text_snippet=record["text_snippet"],
            )

            # Use process_ticket_timed when available to collect step timings.
            # Fall back to process_ticket for agents that don't implement it
            # (e.g. FakeAgent in tests) — timing fields default to 0.0.
            if hasattr(self.agent, "process_ticket_timed"):
                triage_result, timing = self.agent.process_ticket_timed(ticket)
            else:
                triage_result = self.agent.process_ticket(ticket)
                timing = {}

            row = _build_result_row(triage_result, record, timing)
            results.append(row)

            _print_ticket_decision(i, total, row, self.log_decisions, self.log_reviewer_events)

            if trace_path is not None:
                append_jsonl_record(trace_path, row)

        return results


def _build_result_row(
    result: TriageResult,
    record: dict,
    timing: dict | None = None,
) -> dict:
    """
    Flatten a TriageResult and its source record into one CSV/JSONL row.

    Prediction fields come from the TriageResult.
    Evaluation metadata (actual_*, proxy_*) come from the original record.
    Timing fields come from the timing dict (defaults to 0.0 when absent).

    missing_fields is kept as a list here so the CSV writer can serialise
    it to a JSON string, and the JSONL writer can store it natively.
    """
    action_status = ""
    action_target = ""
    action_note   = ""
    if result.action_result is not None:
        action_status = result.action_result.action_status
        action_target = result.action_result.target or ""
        action_note   = result.action_result.action_note

    t = timing or {}

    return {
        # --- prediction output ---
        "ticket_id":                  result.ticket_id,
        "text_snippet":               result.text_snippet,
        "topic":                      result.topic.value,
        "urgency":                    result.urgency.value,
        "next_action":                result.next_action.value,
        "confidence":                 result.confidence,
        "missing_info":               result.missing_info,
        "missing_fields":             result.missing_fields,
        "requires_human_review":      result.requires_human_review,
        "short_note":                 result.short_note,
        "action_status":              action_status,
        "action_target":              action_target,
        "action_note":                action_note,
        # --- per-ticket timing (seconds) ---
        "retrieval_seconds":          t.get("retrieval_seconds", 0.0),
        "llm_seconds":                t.get("llm_seconds", 0.0),
        "validation_seconds":         t.get("validation_seconds", 0.0),
        "reviewer_seconds":           t.get("reviewer_seconds", 0.0),
        "routing_seconds":            t.get("routing_seconds", 0.0),
        "action_execution_seconds":   t.get("action_execution_seconds", 0.0),
        "total_ticket_seconds":       t.get("total_ticket_seconds", 0.0),
        # --- reviewer trace fields ---
        "reviewer_used":              t.get("reviewer_used", False),
        "reviewer_model":             t.get("reviewer_model", ""),
        "reviewer_changed_topic":     t.get("reviewer_changed_topic", False),
        "reviewer_changed_urgency":   t.get("reviewer_changed_urgency", False),
        "reviewer_trigger_flags":     t.get("reviewer_trigger_flags", "[]"),
        # Pre-reviewer LLM outputs — safe prediction fields, not historical labels.
        "first_topic":                t.get("first_topic", ""),
        "first_urgency":              t.get("first_urgency", ""),
        "first_confidence":           t.get("first_confidence", 0.0),
        # Explainability fields.
        "first_short_note":           t.get("first_short_note", ""),
        "reviewer_note":              t.get("reviewer_note", ""),
        "validator_flags":            t.get("validator_flags", "[]"),
        "validator_notes":            t.get("validator_notes", "[]"),
        # Neighbor retrieval evidence (historical, not current-ticket labels).
        "neighbor_predicted_topic":    t.get("neighbor_predicted_topic", ""),
        "neighbor_topic_confidence":   t.get("neighbor_topic_confidence", 0.0),
        "neighbor_predicted_priority": t.get("neighbor_predicted_priority", ""),
        "neighbor_priority_confidence": t.get("neighbor_priority_confidence", 0.0),
        # --- evaluation / reference metadata ---
        "actual_queue":               record.get("actual_queue", ""),
        "actual_priority":            record.get("actual_priority", ""),
        "actual_type":                record.get("actual_type", ""),
        "proxy_topic":                record.get("proxy_topic", ""),
        "proxy_urgency":              record.get("proxy_urgency", ""),
        "proxy_next_action":          record.get("proxy_next_action", ""),
        "proxy_topic_source":         record.get("proxy_topic_source", ""),
    }


def _print_ticket_decision(
    index: int,
    total: int,
    row: dict,
    log_decisions: bool,
    log_reviewer_events: bool,
) -> None:
    """
    Print a compact per-ticket decision line and optional explainability output.

    Decision line (when log_decisions is True):
      Processing ticket 18/200: <ticket_id> -> topic=<t>, urgency=<u>,
      action=<a>, final_conf=<c>, human_review=yes/no, reviewer=yes/no

    Validator flags are printed selectively — not for every normal ticket:
      - human_review=yes (but reviewer not used): one compact validator flags line.
      - reviewer_used=yes and log_reviewer_events=False: validator flags + trigger flags.
      - reviewer_used=yes and log_reviewer_events=True: full reviewer block (see below).

    Full reviewer block (when reviewer_used and log_reviewer_events are both True):
      REVIEWER used: <model>
        validator flags : <validator_flags>
        trigger flags   : <reviewer_trigger_flags>
        neighbor evidence: topic=<...> conf=<...>, priority=<...> conf=<...>
        before: <first_topic> / <first_urgency> / conf=<first_confidence>
        after : <topic> / <urgency> / conf=<confidence>
        note  : <reviewer_note>
        changed: topic=<bool>, urgency=<bool>, seconds=<reviewer_seconds>

    No actual_* or proxy_* values are printed.
    No raw LLM prompts or raw LLM responses are printed.
    """
    if not log_decisions:
        return

    ticket_id        = row["ticket_id"]
    topic            = row["topic"]
    urgency          = row["urgency"]
    action           = row["next_action"]
    confidence       = row["confidence"]
    human_review     = bool(row.get("requires_human_review"))
    reviewer_used    = bool(row.get("reviewer_used"))
    human_review_str = "yes" if human_review else "no"
    reviewer_str     = "yes" if reviewer_used else "no"

    print(
        f"  Processing ticket {index}/{total}: {ticket_id}"
        f" -> topic={topic}, urgency={urgency},"
        f" action={action}, final_conf={confidence:.2f},"
        f" human_review={human_review_str}, reviewer={reviewer_str}"
    )

    if reviewer_used and log_reviewer_events:
        # Full reviewer block — contains validator flags and trigger flags inside.
        reviewer_model    = row.get("reviewer_model", "")
        validator_flags   = row.get("validator_flags", "[]")
        trigger_flags     = row.get("reviewer_trigger_flags", "[]")
        nb_topic          = row.get("neighbor_predicted_topic", "")
        nb_topic_conf     = float(row.get("neighbor_topic_confidence", 0.0))
        nb_priority       = row.get("neighbor_predicted_priority", "")
        nb_priority_conf  = float(row.get("neighbor_priority_confidence", 0.0))
        first_topic       = row.get("first_topic", "")
        first_urgency     = row.get("first_urgency", "")
        first_confidence  = float(row.get("first_confidence", 0.0))
        reviewer_note     = row.get("reviewer_note", "")
        reviewer_seconds  = float(row.get("reviewer_seconds", 0.0))
        changed_topic     = row.get("reviewer_changed_topic", False)
        changed_urgency   = row.get("reviewer_changed_urgency", False)

        print(f"    REVIEWER used: {reviewer_model}")
        print(f"      validator flags : {validator_flags}")
        print(f"      trigger flags   : {trigger_flags}")
        print(
            f"      neighbor evidence: topic={nb_topic} conf={nb_topic_conf:.2f},"
            f" priority={nb_priority} conf={nb_priority_conf:.2f}"
        )
        print(f"      before: {first_topic} / {first_urgency} / conf={first_confidence:.2f}")
        print(f"      after : {topic} / {urgency} / conf={confidence:.2f}")
        print(f"      note  : {reviewer_note}")
        print(
            f"      changed: topic={changed_topic},"
            f" urgency={changed_urgency},"
            f" seconds={reviewer_seconds:.2f}"
        )
    elif reviewer_used:
        # Reviewer was used but full block is suppressed — print compact 2-liner.
        print(f"    validator flags : {row.get('validator_flags', '[]')}")
        print(f"    trigger flags   : {row.get('reviewer_trigger_flags', '[]')}")
    elif human_review:
        # Human review flagged but reviewer not used — print validator flags only.
        print(f"    validator flags: {row.get('validator_flags', '[]')}")
