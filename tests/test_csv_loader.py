"""
Tests for src/infrastructure/csv_loader.py.

Verifies that:
- answer is excluded from the loaded DataFrame.
- tag columns are collected correctly.
- missing values are returned as empty strings, not 'nan'.
"""
import io

import pandas as pd
import pytest

from src.infrastructure.csv_loader import collect_tags, load_csv


# ─── Fixtures ────────────────────────────────────────────────────────────────

_CSV_WITH_ANSWER = """subject,body,answer,queue,priority,type,tag_1,tag_2
Login broken,Cannot log in,We fixed it,IT Support,high,Incident,Bug,Access
Billing query,Invoice wrong,,Billing and Payments,low,Request,Billing,
"""

_CSV_WITHOUT_ANSWER = """subject,body,queue,priority,type,tag_1,tag_2
Policy question,What is covered?,Product Support,medium,Request,Policy,
"""


def _df_from_string(csv_text: str) -> pd.DataFrame:
    return load_csv(io.StringIO(csv_text))  # type: ignore[arg-type]


# ─── answer exclusion ────────────────────────────────────────────────────────

def test_answer_column_is_dropped_when_present():
    df = _df_from_string(_CSV_WITH_ANSWER)
    assert "answer" not in df.columns


def test_answer_column_absent_does_not_raise():
    df = _df_from_string(_CSV_WITHOUT_ANSWER)
    assert "answer" not in df.columns


def test_other_columns_are_kept():
    df = _df_from_string(_CSV_WITH_ANSWER)
    for col in ["subject", "body", "queue", "priority", "type", "tag_1", "tag_2"]:
        assert col in df.columns


# ─── missing values ───────────────────────────────────────────────────────────

def test_missing_values_are_empty_strings_not_nan():
    df = _df_from_string(_CSV_WITH_ANSWER)
    # tag_2 of row 1 is empty in the CSV
    assert df.iloc[1]["tag_2"] == ""


# ─── collect_tags ─────────────────────────────────────────────────────────────

def test_collect_tags_returns_nonempty_tags():
    df = _df_from_string(_CSV_WITH_ANSWER)
    tags = collect_tags(df.iloc[0])
    assert "Bug" in tags
    assert "Access" in tags


def test_collect_tags_skips_empty_values():
    df = _df_from_string(_CSV_WITH_ANSWER)
    tags = collect_tags(df.iloc[1])
    assert "Billing" in tags
    # Second tag was empty — should not appear
    assert "" not in tags


def test_collect_tags_returns_list():
    df = _df_from_string(_CSV_WITH_ANSWER)
    tags = collect_tags(df.iloc[0])
    assert isinstance(tags, list)


def test_collect_tags_returns_empty_list_when_no_tags():
    row = pd.Series({"subject": "x", "body": "y"})
    assert collect_tags(row) == []
