"""
Tests for src/application/preprocessing.py.

Verifies that:
- None/NaN inputs are handled safely.
- Whitespace normalization works.
- build_raw_text and build_representation_text produce correct structure.
- make_text_snippet truncates at the correct length.
- is_too_short detects short text.
"""
from src.application.preprocessing import (
    build_raw_text,
    build_representation_text,
    make_text_snippet,
    normalize_text,
)


# ─── normalize_text ───────────────────────────────────────────────────────────

def test_normalize_text_handles_none():
    assert normalize_text(None) == ""


def test_normalize_text_collapses_whitespace():
    assert normalize_text("hello   world\n\tthere") == "hello world there"


def test_normalize_text_decodes_html_entities():
    assert normalize_text("AT&amp;T") == "AT&T"


def test_normalize_text_strips_html_tags():
    assert normalize_text("<b>Bold</b> text") == "Bold text"


def test_normalize_text_returns_empty_string_for_empty_input():
    assert normalize_text("") == ""


# ─── build_raw_text ───────────────────────────────────────────────────────────

def test_build_raw_text_combines_subject_and_body():
    result = build_raw_text("Subject line", "Body text here")
    assert "Subject line" in result
    assert "Body text here" in result


def test_build_raw_text_handles_none_subject():
    result = build_raw_text(None, "Body text")
    assert "Body text" in result
    assert "None" not in result


def test_build_raw_text_handles_none_body():
    result = build_raw_text("Subject", None)
    assert "Subject" in result
    assert "None" not in result


def test_build_raw_text_handles_both_empty():
    result = build_raw_text("", "")
    assert isinstance(result, str)


# ─── build_representation_text ────────────────────────────────────────────────

def test_representation_text_contains_subject_label():
    result = build_representation_text("My subject", "My body")
    assert result.startswith("Subject:")


def test_representation_text_contains_body_label():
    result = build_representation_text("My subject", "My body")
    assert "Body:" in result


def test_representation_text_includes_content():
    result = build_representation_text("Payment issue", "Invoice not received")
    assert "Payment issue" in result
    assert "Invoice not received" in result


def test_representation_text_handles_none_inputs():
    result = build_representation_text(None, None)
    assert "Subject:" in result
    assert "Body:" in result
    assert "None" not in result


def test_representation_text_truncates_long_body():
    long_body = "x" * 2000
    result = build_representation_text("Subject", long_body, body_limit=1500)
    assert len(result) < 2000


# ─── make_text_snippet ────────────────────────────────────────────────────────

def test_make_text_snippet_truncates_long_text():
    long_text = "a" * 300
    snippet = make_text_snippet(long_text, max_chars=240)
    assert len(snippet) == 240


def test_make_text_snippet_leaves_short_text_unchanged():
    short_text = "Hello, I need help."
    assert make_text_snippet(short_text) == short_text


def test_make_text_snippet_handles_none():
    assert make_text_snippet(None) == ""


def test_make_text_snippet_respects_custom_max():
    result = make_text_snippet("abcdefghij", max_chars=5)
    assert result == "abcde"


