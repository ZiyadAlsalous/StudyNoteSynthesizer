"""SQLite: courses, lectures, chapters, runs, the rejection log."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from types import TracebackType
from typing import Any

from ..models import ChapterRange, Lecture, Parent, Rejection, RunRecord
from .errors import StoreError

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
CREATE TABLE IF NOT EXISTS lectures (
    id          TEXT NOT NULL,
    course      TEXT NOT NULL,
    title       TEXT NOT NULL,
    chapter     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    slides_name TEXT NOT NULL DEFAULT '',
    note_count  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (course, id)
);
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    course        TEXT NOT NULL,
    chapter       TEXT NOT NULL,
    lecture       TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    document_path TEXT,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS runs_lecture ON runs (course, lecture);
CREATE TABLE IF NOT EXISTS textbooks (
    course     TEXT PRIMARY KEY,
    filename   TEXT NOT NULL,
    chunks     INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
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
    return datetime.now(UTC).isoformat()


def _lecture(row: sqlite3.Row) -> Lecture:
    return Lecture(
        id=row["id"], course=row["course"], title=row["title"], chapter=row["chapter"],
        created_at=datetime.fromisoformat(row["created_at"]),
        slides_name=row["slides_name"], note_count=row["note_count"],
    )


def _run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"], course=row["course"], chapter=row["chapter"], lecture=row["lecture"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        document_path=row["document_path"], error=row["error"],
    )


def _locked[F: Callable[..., Any]](method: F) -> F:
    """Serialize access to the connection."""

    @wraps(method)
    def guarded(self: Catalogue, *args: Any, **kwargs: Any) -> Any:
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

    def __enter__(self) -> Catalogue:
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
    def add_lecture(self, course: str, lecture_id: str, title: str, chapter: str = "") -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO lectures (id, course, title, chapter, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (lecture_id, course, title, chapter, _now()),
        )
        self._db.commit()

    @_locked
    def lectures(self, course: str) -> list[Lecture]:
        rows = self._db.execute(
            "SELECT * FROM lectures WHERE course = ? ORDER BY created_at", (course,)
        ).fetchall()
        return [_lecture(row) for row in rows]

    @_locked
    def lecture(self, course: str, lecture_id: str) -> Lecture:
        row = self._db.execute(
            "SELECT * FROM lectures WHERE course = ? AND id = ?", (course, lecture_id)
        ).fetchone()
        if row is None:
            raise StoreError(f"No lecture {lecture_id} in {course}")
        return _lecture(row)

    @_locked
    def set_lecture_sources(
        self, course: str, lecture_id: str, slides_name: str | None, note_count: int | None
    ) -> None:
        """Called after an upload; either half can change on its own."""
        if slides_name is not None:
            self._db.execute(
                "UPDATE lectures SET slides_name = ? WHERE course = ? AND id = ?",
                (slides_name, course, lecture_id),
            )
        if note_count is not None:
            self._db.execute(
                "UPDATE lectures SET note_count = ? WHERE course = ? AND id = ?",
                (note_count, course, lecture_id),
            )
        self._db.commit()

    @_locked
    def delete_lecture(self, course: str, lecture_id: str) -> None:
        self._db.execute(
            "DELETE FROM lectures WHERE course = ? AND id = ?", (course, lecture_id)
        )
        self._db.commit()

    @_locked
    def record_textbook(self, course: str, filename: str, chunks: int) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO textbooks (course, filename, chunks, indexed_at) "
            "VALUES (?, ?, ?, ?)",
            (course, filename, chunks, _now()),
        )
        self._db.commit()

    @_locked
    def textbook(self, course: str) -> dict[str, object] | None:
        """What is already indexed, so a book is never embedded twice."""
        row = self._db.execute(
            "SELECT filename, chunks, indexed_at FROM textbooks WHERE course = ?", (course,)
        ).fetchone()
        return dict(row) if row else None

    @_locked
    def runs_for(self, course: str, lecture_id: str) -> list[RunRecord]:
        rows = self._db.execute(
            "SELECT * FROM runs WHERE course = ? AND lecture = ? ORDER BY created_at DESC",
            (course, lecture_id),
        ).fetchall()
        return [_run(row) for row in rows]

    @_locked
    def start_run(
        self, run_id: str, course: str, chapter: str, lecture: str = ""
    ) -> RunRecord:
        stamp = _now()
        self._db.execute(
            "INSERT OR REPLACE INTO runs "
            "(id, course, chapter, lecture, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'running', ?, ?)",
            (run_id, course, chapter, lecture, stamp, stamp),
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
        return _run(row)

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
