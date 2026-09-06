"""Qdrant: one collection per course, payload indexes at creation time."""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ..config import Settings
from ..models import Chunk
from .errors import CollectionMissing, StoreError


def _connect(settings: Settings) -> QdrantClient:
    backend = settings.qdrant.backend
    if backend == "memory":
        return QdrantClient(location=":memory:")
    if backend == "server":
        return QdrantClient(url=settings.qdrant.url)
    if backend == "local":
        settings.paths.vectors.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.paths.vectors))
    raise StoreError(f"Unknown qdrant backend {backend!r}")


class VectorStore:
    """Qdrant: one collection per course, payload indexes at creation time."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = _connect(settings)

    def exists(self, course: str) -> bool:
        """Whether this course's textbook is already embedded."""
        return bool(self._client.collection_exists(self.collection_for(course)))

    def count(self, course: str) -> int:
        if not self.exists(course):
            return 0
        return int(self._client.count(self.collection_for(course)).count)

    def drop(self, course: str) -> None:
        if self.exists(course):
            self._client.delete_collection(self.collection_for(course))

    @staticmethod
    def collection_for(course: str) -> str:
        return f"course_{course}"

    def create(self, course: str, dimensions: int) -> str:
        name = self.collection_for(course)
        if self._client.collection_exists(name):
            return name
        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=dimensions,
                distance=qmodels.Distance[self._settings.qdrant.distance.upper()],
            ),
        )
        # Declared here, not later: chapter scoping (spec 7.2) must filter before the vector se.
        for field in self._settings.qdrant.payload_indexes:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=(
                    qmodels.PayloadSchemaType.INTEGER
                    if field.startswith("page")
                    else qmodels.PayloadSchemaType.KEYWORD
                ),
            )
        return name

    def upsert(self, course: str, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise StoreError(f"{len(chunks)} chunks against {len(vectors)} vectors")
        name = self.collection_for(course)
        if not self._client.collection_exists(name):
            raise CollectionMissing(f"Collection {name} does not exist")
        points = [
            qmodels.PointStruct(
                id=index,
                vector=list(vector),
                payload={
                    "chunk_id": chunk.id,
                    "chapter": chunk.chapter,
                    "section_path": chunk.section_path,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "parent_id": chunk.parent_id,
                    "text": chunk.text,
                    "token_estimate": chunk.token_estimate,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self._client.upsert(collection_name=name, points=points)
        return len(points)

    def search(
        self, course: str, vector: Sequence[float], chapters: Sequence[str], limit: int
    ) -> list[tuple[float, dict[str, object]]]:
        """Chapter filter is passed to Qdrant, never applied to the results."""
        name = self.collection_for(course)
        if not self._client.collection_exists(name):
            raise CollectionMissing(f"Collection {name} does not exist")
        condition = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="chapter", match=qmodels.MatchAny(any=list(chapters))
                )
            ]
        )
        found = self._client.query_points(
            collection_name=name,
            query=list(vector),
            query_filter=condition,
            limit=limit,
            with_payload=True,
        )
        return [(point.score, dict(point.payload or {})) for point in found.points]
