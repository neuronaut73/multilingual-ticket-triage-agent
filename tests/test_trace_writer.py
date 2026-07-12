"""
Tests for Sprint 6A — trace_writer (JSONL output).

No DuckDB, LanceDB, embeddings, or Ollama calls.

Test coverage:
  write_jsonl:
    - creates the output file
    - creates parent directory if needed
    - one line per record, valid JSON
    - fields, lists, bools preserved
    - empty input → empty file
  init_jsonl:
    - creates the file
    - truncates an existing file
    - creates parent directory
  append_jsonl_record:
    - appends one record to an existing file
    - each call adds exactly one line
    - records are readable as JSON immediately after each append
    - file grows monotonically
"""

import json
import os

import pytest

from src.infrastructure.trace_writer import append_jsonl_record, init_jsonl, write_jsonl


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_record(**overrides) -> dict:
    record = {
        "ticket_id":             "t001",
        "topic":                 "Technical / Online Access",
        "urgency":               "High",
        "next_action":           "escalate_to_human_supervisor",
        "confidence":            0.88,
        "missing_info":          True,
        "missing_fields":        ["customer_identifier"],
        "requires_human_review": True,
        "short_note":            "Customer blocked from portal.",
        "action_result":         {"action_status": "simulated_success", "target": "supervisor"},
    }
    record.update(overrides)
    return record


def _read_lines(path: str) -> list[dict]:
    """Read all JSONL lines from path and return as list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestWriteJsonl:

    def test_creates_output_file(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        write_jsonl(path, [_make_record()])
        assert os.path.exists(path)

    def test_creates_parent_directory(self, tmp_path) -> None:
        path = str(tmp_path / "a" / "b" / "trace.jsonl")
        write_jsonl(path, [_make_record()])
        assert os.path.exists(path)

    def test_one_line_per_record(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        records = [_make_record(ticket_id=f"t{i}") for i in range(7)]
        write_jsonl(path, records)
        lines = _read_lines(path)
        assert len(lines) == 7

    def test_each_line_is_valid_json(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        write_jsonl(path, [_make_record(), _make_record(ticket_id="t002")])
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if raw_line:
                    json.loads(raw_line)  # must not raise

    def test_fields_are_preserved(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        write_jsonl(path, [_make_record(ticket_id="abc123")])
        lines = _read_lines(path)
        assert lines[0]["ticket_id"] == "abc123"
        assert lines[0]["topic"]     == "Technical / Online Access"
        assert lines[0]["confidence"] == 0.88

    def test_list_field_preserved_as_list(self, tmp_path) -> None:
        """missing_fields should remain a JSON array, not a string."""
        path = str(tmp_path / "trace.jsonl")
        write_jsonl(path, [_make_record(missing_fields=["field_a", "field_b"])])
        lines = _read_lines(path)
        assert lines[0]["missing_fields"] == ["field_a", "field_b"]

    def test_bool_fields_preserved(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        write_jsonl(path, [_make_record(missing_info=False, requires_human_review=True)])
        lines = _read_lines(path)
        assert lines[0]["missing_info"]          is False
        assert lines[0]["requires_human_review"] is True

    def test_empty_input_produces_empty_file(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        write_jsonl(path, [])
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert content == ""

    def test_ticket_ids_are_in_order(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        records = [_make_record(ticket_id=f"t{i:03d}") for i in range(5)]
        write_jsonl(path, records)
        lines = _read_lines(path)
        assert [l["ticket_id"] for l in lines] == [f"t{i:03d}" for i in range(5)]


class TestInitJsonl:

    def test_creates_empty_file(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        init_jsonl(path)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            assert f.read() == ""

    def test_truncates_existing_content(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"ticket_id": "old"}\n')
        init_jsonl(path)
        with open(path, encoding="utf-8") as f:
            assert f.read() == ""

    def test_creates_parent_directory(self, tmp_path) -> None:
        path = str(tmp_path / "new_dir" / "trace.jsonl")
        init_jsonl(path)
        assert os.path.exists(path)


class TestAppendJsonlRecord:

    def test_appends_one_line(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        init_jsonl(path)
        append_jsonl_record(path, _make_record(ticket_id="t001"))
        lines = _read_lines(path)
        assert len(lines) == 1

    def test_each_call_adds_exactly_one_line(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        init_jsonl(path)
        for i in range(5):
            append_jsonl_record(path, _make_record(ticket_id=f"t{i:03d}"))
        lines = _read_lines(path)
        assert len(lines) == 5

    def test_appended_record_is_valid_json(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        init_jsonl(path)
        append_jsonl_record(path, _make_record(ticket_id="abc"))
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        json.loads(raw)  # must not raise

    def test_ticket_id_preserved(self, tmp_path) -> None:
        path = str(tmp_path / "trace.jsonl")
        init_jsonl(path)
        append_jsonl_record(path, _make_record(ticket_id="xyz_999"))
        lines = _read_lines(path)
        assert lines[0]["ticket_id"] == "xyz_999"

    def test_file_grows_monotonically(self, tmp_path) -> None:
        """Line count must increase after every append."""
        path = str(tmp_path / "trace.jsonl")
        init_jsonl(path)
        previous_size = os.path.getsize(path)
        for i in range(3):
            append_jsonl_record(path, _make_record(ticket_id=f"t{i}"))
            current_size = os.path.getsize(path)
            assert current_size > previous_size
            previous_size = current_size

    def test_records_readable_after_each_append(self, tmp_path) -> None:
        """
        Simulate observing the file from a second process.
        After each append the file should contain exactly N complete JSON lines.
        """
        path = str(tmp_path / "trace.jsonl")
        init_jsonl(path)
        for n in range(1, 6):
            append_jsonl_record(path, _make_record(ticket_id=f"t{n:03d}"))
            lines = _read_lines(path)
            assert len(lines) == n, f"Expected {n} lines after {n} appends, got {len(lines)}"
