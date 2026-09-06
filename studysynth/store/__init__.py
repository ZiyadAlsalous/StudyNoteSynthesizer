"""Persistence: SQLite metadata, disk layout, and the vector index."""

from .catalogue import Catalogue
from .errors import CollectionMissing, IndexBusy, StoreError
from .files import Places
from .vectors import VectorStore

__all__ = [
    "Catalogue", "CollectionMissing", "IndexBusy", "Places", "StoreError", "VectorStore",
]
