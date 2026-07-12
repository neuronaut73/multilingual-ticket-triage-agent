"""
Sprint 6A — JSONL trace writer.

Writes one JSON object per line so the trace file can be read record-by-record
without loading the whole file into memory.

Three functions:
  init_jsonl          — create or truncate the file at the start of a run
  append_jsonl_record — append one record and flush immediately (streaming)
  write_jsonl         — write all records at once (bulk, used in tests)
"""

import json
import os


def init_jsonl(path: str) -> None:
    """
    Create or truncate the JSONL file so this run starts clean.

    Must be called once before the first append_jsonl_record call.
    Creates the parent directory if needed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w", encoding="utf-8").close()


def append_jsonl_record(path: str, record: dict) -> None:
    """
    Append one record to the JSONL file and flush immediately.

    Opening, writing, and closing per record is intentional: each LLM call
    takes 20–60 s, so the overhead is negligible.  The flush ensures the
    record is visible to other processes (e.g. Get-Content -Tail 5) without
    waiting for buffered writes or process exit.

    Parameters
    ----------
    path:
        File path for the JSONL file.  Must already exist (call init_jsonl first).
    record:
        Dict to serialise.  Values must be JSON-serialisable.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: str, records: list[dict]) -> None:
    """
    Write all records to a JSONL file at once (bulk write).

    Creates the parent directory if needed.
    Each record is serialised as a single-line JSON object followed by a newline.

    Parameters
    ----------
    path:
        File path for the output JSONL file, e.g. "outputs/triage_trace.jsonl".
    records:
        List of dicts.  Values must be JSON-serialisable.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
