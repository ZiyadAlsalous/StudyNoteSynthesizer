"""Embedding backends behind one interface.

The interface exists because the spec calls for swapping Qwen3-Embedding-8B
down to the 4B or 0.6B variants by config, and because retrieval tests must run
without downloading a model.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import Settings

Vectors = NDArray[np.float32]

_WORD = re.compile(r"[a-z0-9]+")


class EmbeddingError(RuntimeError):
    pass


class EmbeddingBackend(ABC):
    """Documents and queries embed differently: the instruction prefix on a
    Qwen3 model belongs on queries only."""

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    @abstractmethod
    def embed_query(self, text: str) -> Vectors: ...


class MockEmbeddings(EmbeddingBackend):
    """Deterministic hashed bag-of-words, L2-normalized.

    Not a semantic model, but it produces real cosine geometry: identical text
    scores 1.0, paraphrases score high, unrelated text scores low. That is
    exactly the property the novelty filter (spec 7.5) is tested against.
    """

    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _vector(self, text: str) -> Vectors:
        vector = np.zeros(self._dimensions, dtype=np.float32)
        for word in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else (vector / norm).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self._dimensions), dtype=np.float32)
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> Vectors:
        return self._vector(text)


class QwenEmbeddings(EmbeddingBackend):
    """sentence-transformers, last-token pooling, normalized, cosine distance."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings.embeddings
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._settings.model)
        return self._model

    @property
    def dimensions(self) -> int:
        return self._settings.dimensions

    def _encode(self, texts: Sequence[str]) -> Vectors:
        encoded = self._load().encode(
            list(texts),
            batch_size=self._settings.batch_size,
            normalize_embeddings=True,
        )
        return np.asarray(encoded, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return self._encode(texts)

    def embed_query(self, text: str) -> Vectors:
        encoded: Vectors = self._encode([self._settings.query_prefix + text])[0]
        return encoded


def cosine(left: Vectors, right: Vectors) -> float:
    """Both sides are already normalized, so this is a dot product."""
    return float(np.dot(left, right))


def build(settings: Settings) -> EmbeddingBackend:
    backend = settings.embeddings.backend
    if backend == "mock":
        return MockEmbeddings(settings.embeddings.dimensions)
    if backend == "qwen":
        return QwenEmbeddings(settings)
    raise EmbeddingError(f"Unknown embedding backend '{backend}'")
