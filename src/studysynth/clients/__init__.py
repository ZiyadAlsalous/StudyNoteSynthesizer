"""The two external services, each behind one swappable interface."""

from ..clients.embeddings import EmbeddingBackend, EmbeddingError, cosine
from ..clients.llm import LlmClient, LlmError, MissingCredentials, PromptLibrary
from . import embeddings, llm

__all__ = [
    "EmbeddingBackend", "EmbeddingError", "LlmClient", "LlmError",
    "MissingCredentials", "PromptLibrary", "cosine", "embeddings", "llm",
]
