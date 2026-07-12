"""
Tests for src/infrastructure/embedding_model.py

Strategy:
  - Test the prefix helper (_add_prefix) directly — fast, no model load.
  - Test device resolution (_resolve_device) — fast, no model load.
  - Keep the full-model encode test marked as a slow smoke test so the
    default pytest run remains fast.

Run the fast tests only:
    python -m pytest tests/test_embedding_model.py -m "not slow"

Run everything including the slow model smoke test:
    python -m pytest tests/test_embedding_model.py
"""

import numpy as np
import pytest

from src.infrastructure.embedding_model import (
    EmbeddingModel,
    _add_prefix,
    _resolve_device,
)


# ---------------------------------------------------------------------------
# _add_prefix
# ---------------------------------------------------------------------------

def test_add_prefix_passage():
    texts = ["Hello world", "Another text"]
    result = _add_prefix(texts, "passage: ")
    assert result == ["passage: Hello world", "passage: Another text"]


def test_add_prefix_query():
    texts = ["My query"]
    result = _add_prefix(texts, "query: ")
    assert result == ["query: My query"]


def test_add_prefix_empty_list():
    assert _add_prefix([], "passage: ") == []


def test_add_prefix_preserves_whitespace():
    texts = ["  leading space"]
    result = _add_prefix(texts, "passage: ")
    assert result == ["passage:   leading space"]


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------

def test_resolve_device_cpu():
    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_cuda():
    # Should return 'cuda' unchanged regardless of hardware availability
    assert _resolve_device("cuda") == "cuda"


def test_resolve_device_auto_returns_string():
    device = _resolve_device("auto")
    assert device in ("cuda", "cpu")


# ---------------------------------------------------------------------------
# EmbeddingModel — slow smoke test (requires model download, ~1 GB)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_embedding_model_encode_passages_shape():
    """
    Loads the real multilingual-e5-large model and encodes two short texts.
    Verifies shape and dtype — does not check exact values.
    Mark: slow (skipped in fast CI runs).
    """
    model = EmbeddingModel(
        model_name="intfloat/multilingual-e5-large",
        device="auto",
        normalize_embeddings=True,
    )
    assert model.get_dimension() == 1024

    vecs = model.encode_passages(["Hello world", "Bonjour le monde"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, 1024)
    assert vecs.dtype in (np.float32, np.float64)


@pytest.mark.slow
def test_embedding_model_encode_queries_shape():
    model = EmbeddingModel(
        model_name="intfloat/multilingual-e5-large",
        device="auto",
        normalize_embeddings=True,
    )
    vecs = model.encode_queries(["What is my policy number?"])
    assert vecs.shape == (1, 1024)


@pytest.mark.slow
def test_normalized_vectors_have_unit_norm():
    """Normalised vectors should have L2 norm ≈ 1.0."""
    model = EmbeddingModel(
        model_name="intfloat/multilingual-e5-large",
        device="auto",
        normalize_embeddings=True,
    )
    vecs = model.encode_passages(["Test sentence for norm check."])
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
