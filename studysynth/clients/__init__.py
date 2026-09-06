"""The two external services, each behind one swappable interface."""

from . import embeddings, llm
from ..clients.embeddings import EmbeddingBackend, EmbeddingError, cosine
from ..clients.llm import LlmClient, LlmError, MissingCredentials, PromptLibrary

__all__ = [
    "EmbeddingBackend", "EmbeddingError", "LlmClient", "LlmError",
    "MissingCredentials", "PromptLibrary", "cosine", "embeddings", "llm",
]
