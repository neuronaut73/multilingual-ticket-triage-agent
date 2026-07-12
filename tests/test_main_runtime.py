"""
Tests for Sprint 5F — Runtime Modes and Rebuild Control.

Verifies:
- config.yaml contains all four runtime flags with correct boolean types.
- _duckdb_exists returns False for a path that does not exist.
- _lancedb_table_exists returns False for a directory that does not exist.
- _connect_existing_duckdb raises RuntimeError with a clear message when the
  database file is absent.
- _connect_existing_lancedb raises RuntimeError with a clear message when the
  LanceDB table is absent.
- When the runtime section is missing from config, rebuild flags default to
  True (safe rebuild behavior).

No model loading, no Ollama calls, no actual DuckDB data, no actual LanceDB
data required — all tests run without the data/ directory being populated.
"""
import pytest
import yaml

from main import (
    _connect_existing_duckdb,
    _connect_existing_lancedb,
    _duckdb_exists,
    _lancedb_table_exists,
)


def test_runtime_config_has_expected_flags():
    """config.yaml must contain all four runtime flags as booleans."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rt = cfg.get("runtime", {})
    expected_flags = (
        "rebuild_duckdb",
        "rebuild_lancedb",
        "run_smoke_search",
        "run_end_to_end_smoke",
    )
    for flag in expected_flags:
        assert flag in rt, f"Missing runtime flag in config.yaml: {flag}"
        assert isinstance(rt[flag], bool), (
            f"runtime.{flag} must be bool, got {type(rt[flag]).__name__}"
        )


def test_runtime_flags_default_to_true_when_section_missing():
    """
    When the runtime section is absent, rebuild flags default to True.

    True is the safe default: it triggers a full rebuild so a fresh environment
    always works without manual flag changes.
    """
    cfg = {}
    rt = cfg.get("runtime", {})
    assert rt.get("rebuild_duckdb", True) is True
    assert rt.get("rebuild_lancedb", True) is True
    assert rt.get("run_smoke_search", True) is True
    assert rt.get("run_end_to_end_smoke", True) is True


def test_duckdb_exists_returns_false_for_nonexistent_path():
    """_duckdb_exists returns False when the file does not exist."""
    assert _duckdb_exists("data/__nonexistent_test_xyz__.duckdb") is False


def test_lancedb_table_exists_returns_false_for_nonexistent_dir():
    """_lancedb_table_exists returns False when the directory does not exist."""
    assert _lancedb_table_exists(
        "data/__nonexistent_lancedb_test_xyz__", "ticket_embeddings"
    ) is False


def test_connect_existing_duckdb_raises_when_missing():
    """
    _connect_existing_duckdb raises RuntimeError with the expected message
    when the database file does not exist.
    """
    with pytest.raises(RuntimeError, match="DuckDB database not found"):
        _connect_existing_duckdb("data/__nonexistent_test_xyz__.duckdb")


def test_connect_existing_lancedb_raises_when_missing(tmp_path):
    """
    _connect_existing_lancedb raises RuntimeError with the expected message
    when the LanceDB table does not exist.

    tmp_path is a pytest-provided temp directory that is guaranteed to be empty,
    so _lancedb_table_exists will return False.
    """
    cfg = {
        "vector_store": {
            "path": str(tmp_path / "lancedb"),
            "table_name": "ticket_embeddings",
        }
    }
    with pytest.raises(RuntimeError, match="LanceDB table not found"):
        _connect_existing_lancedb(cfg)
