"""Failures the storage layer can raise."""

from __future__ import annotations

class StoreError(RuntimeError):
    """Raised when a store operation cannot complete."""


class CollectionMissing(StoreError):
    pass
