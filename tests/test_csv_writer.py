"""
Tests for Sprint 6A — csv_writer.

No DuckDB, LanceDB, embeddings, or Ollama calls.

Test coverage:
  - write_csv_rows creates the output file.
  - write_csv_rows creates the outputs directory if it does not exist.
  - Header row contains expected column names.
  - One data row per input dict.
  - List values are serialised as JSON strings.
  - Dict values are serialised as JSON strings.
  - text_snippet newlines are replaced with spaces.
  - Empty rows list writes nothing (no file created or empty file).
"""

import csv
import json
import os

import pytest

from src.infrastructure.csv_writer import write_csv_rows, _serialize_row


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_row(**overrides) -> dict:
    row = {
        "ticket_id":             "t001",
        "text_snippet":          "Customer cannot log in.",
        "topic":                 "Technical / Online Access",
        "urgency":               "High",
        "next_action":           "escalate_to_human_supervisor",
        "confidence":            0.88,
        "missing_info":          True,
        "missing_fields":        ["customer_identifier"],
        "requires_human_review": True,
        "short_note":            "Blocked after login failure.",
        "action_status":         "simulated_success",
        "action_target":         "human_supervisor_queue",
        "action_note":           "Escalated.",
        "actual_queue":          "Technical Support",
        "actual_priority":       "high",
        "actual_type":           "Incident",
        "proxy_topic":           "Technical / Online Access",
        "proxy_urgency":         "High",
        "proxy_next_action":     "forward_to_technical_support",
        "proxy_topic_source":    "queue_mapping",
    }
    row.update(overrides)
    return row


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestWriteCsvRows:

    def test_creates_output_file(self, tmp_path) -> None:
        path = str(tmp_path / "out" / "results.csv")
        write_csv_rows(path, [_make_row()])
        assert os.path.exists(path)

    def test_creates_parent_directory(self, tmp_path) -> None:
        nested = str(tmp_path / "a" / "b" / "results.csv")
        write_csv_rows(nested, [_make_row()])
        assert os.path.exists(nested)

    def test_header_row_contains_ticket_id(self, tmp_path) -> None:
        path = str(tmp_path / "out.csv")
        write_csv_rows(path, [_make_row()])
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert "ticket_id" in reader.fieldnames

    def test_header_contains_assignment_required_columns(self, tmp_path) -> None:
        required = {
            "ticket_id", "text_snippet", "topic", "urgency",
            "next_action", "short_note",
        }
        path = str(tmp_path / "out.csv")
        write_csv_rows(path, [_make_row()])
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for col in required:
                assert col in reader.fieldnames, f"Missing column: {col}"

    def test_header_contains_evaluation_columns(self, tmp_path) -> None:
        eval_cols = {"actual_queue", "actual_priority", "proxy_topic", "proxy_urgency"}
        path = str(tmp_path / "out.csv")
        write_csv_rows(path, [_make_row()])
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for col in eval_cols:
                assert col in reader.fieldnames, f"Missing column: {col}"

    def test_one_data_row_per_input_dict(self, tmp_path) -> None:
        path = str(tmp_path / "out.csv")
        rows = [_make_row(ticket_id=f"t{i}") for i in range(5)]
        write_csv_rows(path, rows)
        with open(path, encoding="utf-8") as f:
            data_rows = list(csv.DictReader(f))
        assert len(data_rows) == 5

    def test_list_value_serialised_as_json_string(self, tmp_path) -> None:
        path = str(tmp_path / "out.csv")
        write_csv_rows(path, [_make_row(missing_fields=["field_a", "field_b"])])
        with open(path, encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        parsed = json.loads(row["missing_fields"])
        assert parsed == ["field_a", "field_b"]

    def test_empty_list_serialised_as_json_empty_array(self, tmp_path) -> None:
        path = str(tmp_path / "out.csv")
        write_csv_rows(path, [_make_row(missing_fields=[])])
        with open(path, encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        assert row["missing_fields"] == "[]"

    def test_text_snippet_newlines_replaced(self, tmp_path) -> None:
        path = str(tmp_path / "out.csv")
        write_csv_rows(path, [_make_row(text_snippet="line one\nline two")])
        with open(path, encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        assert "\n" not in row["text_snippet"]
        assert "line one line two" == row["text_snippet"]

    def test_empty_input_does_not_raise(self, tmp_path) -> None:
        path = str(tmp_path / "out.csv")
        write_csv_rows(path, [])


class TestSerializeRow:

    def test_list_becomes_json_string(self) -> None:
        row = {"missing_fields": ["a", "b"]}
        out = _serialize_row(row)
        assert out["missing_fields"] == '["a", "b"]'

    def test_dict_becomes_json_string(self) -> None:
        row = {"metadata": {"k": 1}}
        out = _serialize_row(row)
        assert out["metadata"] == '{"k": 1}'

    def test_text_snippet_newline_replaced(self) -> None:
        row = {"text_snippet": "hello\nworld"}
        out = _serialize_row(row)
        assert out["text_snippet"] == "hello world"

    def test_other_strings_unchanged(self) -> None:
        row = {"short_note": "Some note with\nnewline"}
        out = _serialize_row(row)
        assert out["short_note"] == "Some note with\nnewline"

    def test_numeric_values_unchanged(self) -> None:
        row = {"confidence": 0.88}
        out = _serialize_row(row)
        assert out["confidence"] == 0.88

    def test_bool_values_unchanged(self) -> None:
        row = {"missing_info": True}
        out = _serialize_row(row)
        assert out["missing_info"] is True
