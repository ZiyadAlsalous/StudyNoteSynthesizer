"""Failures the storage layer can raise."""

from __future__ import annotations


class StoreError(RuntimeError):
    """Raised when a store operation cannot complete."""


class CollectionMissing(StoreError):
    pass


class IndexBusy(StoreError):
    """The on-disk index is held by another process."""
