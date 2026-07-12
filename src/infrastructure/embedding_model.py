"""
Sprint 3 — Local multilingual embedding model wrapper.

Wraps SentenceTransformer for the intfloat/multilingual-e5-large model.

E5 models expect a task prefix on every input:
  - "passage: " for texts being indexed (reference tickets)
  - "query: "   for texts being searched (incoming / eval tickets)

Without the prefix the model still works, but retrieval quality degrades.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _resolve_device(device: str) -> str:
    """Return 'cuda' if available and requested via 'auto', otherwise 'cpu'."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _add_prefix(texts: list[str], prefix: str) -> list[str]:
    """Prepend a fixed prefix string to every text in the list."""
    return [prefix + t for t in texts]


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer for E5-style retrieval models.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        Resolved device string ('cuda' or 'cpu').
    normalize_embeddings : bool
        Whether to L2-normalise output vectors (required for cosine similarity).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(model_name, device=self.device)

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode reference / historical texts for indexing.
        Adds the required 'passage: ' prefix for E5 models.
        Returns a float32 numpy array of shape (len(texts), dimension).
        """
        prefixed = _add_prefix(texts, "passage: ")
        return self._model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode incoming / query texts for search.
        Adds the required 'query: ' prefix for E5 models.
        Returns a float32 numpy array of shape (len(texts), dimension).
        """
        prefixed = _add_prefix(texts, "query: ")
        return self._model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def get_dimension(self) -> int:
        """Return the output embedding dimension (1024 for multilingual-e5-large)."""
        return self._model.get_sentence_embedding_dimension()
