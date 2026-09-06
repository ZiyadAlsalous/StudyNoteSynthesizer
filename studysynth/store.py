"""Persistence: SQLite metadata, disk layout, Qdrant collection operations.

Three stores with one owner each. Nothing above this layer knows what a cursor
or a Qdrant point looks like.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import Settings
from .models import Chunk, ChapterRange, Parent, Rejection, RunRecord


class StoreError(RuntimeError):
    """Raised when a store operation cannot complete."""


class CollectionMissing(StoreError):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS courses (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chapters (
    course      TEXT NOT NULL,
    chapter     TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    page_start  INTEGER NOT NULL,
    page_end    INTEGER NOT NULL,
    manual      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (course, chapter)
);
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    course        TEXT NOT NULL,
    chapter       TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    document_path TEXT,
    error         TEXT
);
CREATE TABLE IF NOT EXISTS rejections (
    run_id       TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    gap_id       TEXT NOT NULL,
    mechanism    TEXT NOT NULL,
    reason       TEXT NOT NULL,
    score        REAL,
    detail       TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS rejections_run ON rejections (run_id);
CREATE TABLE IF NOT EXISTS parents (
    id           TEXT PRIMARY KEY,
    course       TEXT NOT NULL,
    chapter      TEXT NOT NULL,
    heading      TEXT NOT NULL,
    section_path TEXT NOT NULL,
    text         TEXT NOT NULL,
    page_start   INTEGER NOT NULL,
    page_end     INTEGER NOT NULL,
    tokens       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS parents_chapter ON parents (course, chapter);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


F = TypeVar("F", bound=Callable[..., Any])


def _locked(method: F) -> F:
    """Serialize access to the connection.

    Runs execute on their own threads while the API answers on another, and a
    sqlite3 connection is not safe to share across threads without this.
    """

    @wraps(method)
    def guarded(self: "Catalogue", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return guarded  # type: ignore[return-value]


class Catalogue:
    """SQLite metadata: courses, chapter ranges, runs, rejection log."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def __enter__(self) -> "Catalogue":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @_locked
    def close(self) -> None:
        self._db.close()

    @_locked
    def add_course(self, course_id: str, title: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO courses (id, title, created_at) VALUES (?, ?, ?)",
            (course_id, title, _now()),
        )
        self._db.commit()

    @_locked
    def courses(self) -> list[dict[str, str]]:
        rows = self._db.execute("SELECT id, title FROM courses ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    @_locked
    def set_chapters(self, course: str, ranges: Sequence[ChapterRange]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO chapters "
            "(course, chapter, title, page_start, page_end, manual) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (course, r.chapter, r.title, r.page_start, r.page_end, int(r.manual_override))
                for r in ranges
            ],
        )
        self._db.commit()

    @_locked
    def chapter(self, course: str, chapter: str) -> ChapterRange:
        row = self._db.execute(
            "SELECT chapter, title, page_start, page_end, manual FROM chapters "
            "WHERE course = ? AND chapter = ?",
            (course, chapter),
        ).fetchone()
        if row is None:
            raise StoreError(f"No page range recorded for {course}/{chapter}")
        return ChapterRange(
            chapter=row["chapter"],
            title=row["title"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            manual_override=bool(row["manual"]),
        )

    @_locked
    def chapters(self, course: str) -> list[ChapterRange]:
        rows = self._db.execute(
            "SELECT chapter, title, page_start, page_end, manual FROM chapters "
            "WHERE course = ? ORDER BY page_start",
            (course,),
        ).fetchall()
        return [
            ChapterRange(
                chapter=row["chapter"],
                title=row["title"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                manual_override=bool(row["manual"]),
            )
            for row in rows
        ]

    @_locked
    def start_run(self, run_id: str, course: str, chapter: str) -> RunRecord:
        stamp = _now()
        self._db.execute(
            "INSERT OR REPLACE INTO runs "
            "(id, course, chapter, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'running', ?, ?)",
            (run_id, course, chapter, stamp, stamp),
        )
        self._db.commit()
        return self.run(run_id)

    @_locked
    def finish_run(
        self, run_id: str, status: str, document_path: str | None = None, error: str | None = None
    ) -> None:
        self._db.execute(
            "UPDATE runs SET status = ?, updated_at = ?, document_path = ?, error = ? "
            "WHERE id = ?",
            (status, _now(), document_path, error, run_id),
        )
        self._db.commit()

    @_locked
    def run(self, run_id: str) -> RunRecord:
        row = self._db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise StoreError(f"No run {run_id}")
        return RunRecord(
            id=row["id"],
            course=row["course"],
            chapter=row["chapter"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            document_path=row["document_path"],
            error=row["error"],
        )

    @_locked
    def log_rejections(self, run_id: str, rejections: Iterable[Rejection]) -> None:
        self._db.executemany(
            "INSERT INTO rejections "
            "(run_id, candidate_id, gap_id, mechanism, reason, score, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (run_id, r.candidate_id, r.gap_id, r.mechanism, r.reason.value, r.score, r.detail)
                for r in rejections
            ],
        )
        self._db.commit()

    @_locked
    def rejections(self, run_id: str) -> list[Rejection]:
        rows = self._db.execute(
            "SELECT * FROM rejections WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [
            Rejection(
                candidate_id=row["candidate_id"],
                gap_id=row["gap_id"],
                mechanism=row["mechanism"],
                reason=row["reason"],
                score=row["score"],
                detail=row["detail"],
            )
            for row in rows
        ]

    @_locked
    def put_parents(self, parents: Iterable[Parent]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO parents "
            "(id, course, chapter, heading, section_path, text, page_start, page_end, tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    p.id, p.course, p.chapter, p.heading, p.section_path,
                    p.text, p.page_start, p.page_end, p.token_estimate,
                )
                for p in parents
            ],
        )
        self._db.commit()

    @_locked
    def parent(self, parent_id: str) -> Parent:
        row = self._db.execute("SELECT * FROM parents WHERE id = ?", (parent_id,)).fetchone()
        if row is None:
            raise StoreError(f"No parent section {parent_id}")
        return Parent(
            id=row["id"], course=row["course"], chapter=row["chapter"],
            heading=row["heading"], section_path=row["section_path"], text=row["text"],
            page_start=row["page_start"], page_end=row["page_end"],
            token_estimate=row["tokens"],
        )


class Places:
    """Disk layout. The only module that decides where a file goes."""

    def __init__(self, settings: Settings) -> None:
        self._paths = settings.paths

    def course(self, course: str) -> Path:
        return self._paths.courses / course

    def textbook(self, course: str) -> Path:
        return self.course(course) / "textbook.pdf"

    def run(self, run_id: str) -> Path:
        return self._paths.runs / run_id

    def artifact(self, run_id: str, name: str) -> Path:
        target = self.run(run_id) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def note_cache(self, content_hash: str) -> Path:
        target = self._paths.root / "cache" / "ocr" / f"{content_hash}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


class VectorStore:
    """Qdrant. One collection per course, payload indexes at creation time."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        qdrant = settings.qdrant
        self._client = (
            QdrantClient(location=":memory:")
            if qdrant.backend == "memory"
            else QdrantClient(url=qdrant.url)
        )

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
        # Declared here, not later: chapter scoping (spec 7.2) must filter
        # before the vector search, and that requires the index to exist first.
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
