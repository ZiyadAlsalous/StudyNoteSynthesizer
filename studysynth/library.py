"""Everything the interface needs, wired together once."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .clients import embeddings as embedding_backends
from .clients import llm as llm_backends
from .clients.embeddings import EmbeddingBackend
from .clients.llm import LlmClient
from .config import Settings
from .models import ChapterRange, Lecture, NotePage, RetrievalOutcome, RunRecord
from .pipeline.graph import EXTRACT_CONCEPTS, Nodes, Runner, note_pages
from .pipeline.ingest import OutlineMissing, TextbookIngestor, chapter_ranges, slug
from .pipeline.render import RenderError, provenance_report, to_pdf
from .pipeline.retrieval import TextbookGate
from .store import Catalogue, Places, VectorStore

SLIDE_SUFFIXES = {".pdf", ".pptx"}


class ServiceError(RuntimeError):
    pass


@dataclass
class Library:
    """Constructed once per process."""

    settings: Settings
    catalogue: Catalogue
    places: Places
    vectors: VectorStore
    llm: LlmClient
    embeddings: EmbeddingBackend
    runner: Runner = field(init=False)
    pdf_error: str = field(init=False, default="")

    def __post_init__(self) -> None:
        gate = TextbookGate(
            self.settings, self.llm, self.embeddings, self.vectors, self.catalogue
        )
        nodes = Nodes(
            self.settings, self.llm, self.embeddings, gate, self.catalogue, self.places
        )
        self.runner = Runner(self.settings, nodes)


    def courses(self) -> list[dict[str, str]]:
        return self.catalogue.courses()

    def add_course(self, course_id: str, title: str) -> None:
        self.catalogue.add_course(course_id, title)


    def textbook_status(self, course: str) -> dict[str, object] | None:
        """None when no book is indexed."""
        recorded = self.catalogue.textbook(course)
        if recorded and self.vectors.exists(course):
            return recorded
        return None

    def index_textbook(self, course: str, pdf: Path, filename: str) -> tuple[int, int]:
        """Parse, chunk, embed and index."""
        try:
            ranges = chapter_ranges(pdf)
        except OutlineMissing:
            import pypdf

            pages = len(pypdf.PdfReader(str(pdf)).pages)
            ranges = [
                ChapterRange(
                    chapter="whole-book", title="Whole book", page_start=1,
                    page_end=pages, manual_override=True,
                )
            ]
        return len(ranges), self._index(course, pdf, filename, ranges)

    def reindex(self, course: str, ranges: Sequence[ChapterRange]) -> int:
        """After the page ranges are corrected by hand."""
        recorded = self.catalogue.textbook(course)
        if not recorded:
            raise ServiceError(f"No textbook stored for {course}")
        pdf = self.places.textbook(course)
        return self._index(course, pdf, str(recorded["filename"]), ranges)

    def _index(
        self, course: str, pdf: Path, filename: str, ranges: Sequence[ChapterRange]
    ) -> int:
        self.catalogue.set_chapters(course, ranges)
        parents, chunks = TextbookIngestor(self.settings).ingest(pdf, course, ranges)
        if not chunks:
            raise ServiceError(f"{filename} produced no chunks")
        self.catalogue.put_parents(parents)
        self.vectors.drop(course)
        self.vectors.create(course, self.embeddings.dimensions)
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        self.vectors.upsert(course, chunks, vectors.tolist())
        self.catalogue.record_textbook(course, filename, len(chunks))
        return len(chunks)

    def store_textbook(self, course: str, data: bytes) -> Path:
        target = self.places.textbook(course)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target


    def lectures(self, course: str) -> list[Lecture]:
        return self.catalogue.lectures(course)

    def add_lecture(self, course: str, title: str, chapter: str) -> str:
        lecture_id = slug(title) or uuid.uuid4().hex[:8]
        self.catalogue.add_lecture(course, lecture_id, title, chapter)
        return lecture_id

    def delete_lecture(self, course: str, lecture_id: str) -> None:
        """Removes the sources. Finished documents survive under runs/."""
        shutil.rmtree(self.places.lecture(course, lecture_id), ignore_errors=True)
        self.catalogue.delete_lecture(course, lecture_id)

    def replace_slides(self, course: str, lecture_id: str, name: str, data: bytes) -> None:
        folder = self.places.slides_dir(course, lecture_id)
        self.places.clear(folder)
        (folder / _safe(name, SLIDE_SUFFIXES)).write_bytes(data)
        self.catalogue.set_lecture_sources(course, lecture_id, slides_name=name, note_count=None)

    def replace_notes(self, course: str, lecture_id: str, name: str, data: bytes) -> int:
        """Notes are one PDF of any length. Replacing it removes the previous one,
        so a run never mixes two versions of a page."""
        if Path(name).suffix.lower() != ".pdf":
            raise ServiceError(f"{name} must be a PDF")
        folder = self.places.notes_dir(course, lecture_id)
        self.places.clear(folder)
        shutil.rmtree(folder / "pages", ignore_errors=True)
        target = folder / "notes.pdf"
        target.write_bytes(data)

        pages = self._page_count(target)
        self.catalogue.set_lecture_sources(
            course, lecture_id, slides_name=None, note_count=pages
        )
        return pages

    @staticmethod
    def _page_count(pdf: Path) -> int:
        import pymupdf

        try:
            with pymupdf.open(str(pdf)) as document:  # type: ignore[no-untyped-call]
                return int(document.page_count)
        except Exception as error:
            pdf.unlink(missing_ok=True)
            raise ServiceError(f"{pdf.name} could not be read as a PDF") from error


    def history(self, course: str, lecture_id: str) -> list[RunRecord]:
        return self.catalogue.runs_for(course, lecture_id)

    def document(self, record: RunRecord) -> str:
        if not record.document_path:
            return ""
        path = Path(record.document_path)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def start(self, course: str, lecture: Lecture) -> tuple[str, Iterator[tuple[str, Any]]]:
        if not lecture.ready:
            raise ServiceError("Upload both the slides and the note photos first")
        run_id = uuid.uuid4().hex[:12]
        self.catalogue.start_run(run_id, course, lecture.chapter, lecture.id)
        stream = self.runner.stream(
            run_id,
            course=course,
            chapter=lecture.chapter,
            lecture=lecture.id,
            slides_dir=str(self.places.slides_dir(course, lecture.id)),
            notes_dir=str(self.places.notes_dir(course, lecture.id)),
        )
        return run_id, stream

    def resume(
        self, run_id: str, notes: Sequence[NotePage | dict[str, Any]] | None
    ) -> Iterator[tuple[str, Any]]:
        self.runner.approve_notes(run_id, notes)
        return self.runner.stream(run_id)

    def notes(self, run_id: str) -> list[NotePage]:
        """The transcript the review screen edits, always as typed pages."""
        return note_pages(self.runner.state(run_id))

    def awaiting_review(self, run_id: str) -> bool:
        return EXTRACT_CONCEPTS in self.runner.pending(run_id)

    def finish(self, run_id: str) -> RunRecord:
        """Write the document and the provenance report, then close the run."""
        state = self.runner.state(run_id)
        record = self.catalogue.run(run_id)
        document = state.get("document", "")
        target = self.places.artifact(run_id, "document.md")
        target.write_text(document, encoding="utf-8")

        outcome = state.get("retrieval") or RetrievalOutcome()
        self.places.artifact(run_id, "provenance.md").write_text(
            provenance_report(record.course, record.chapter, outcome), encoding="utf-8"
        )
        try:
            to_pdf(document, self.places.artifact(run_id, "document.pdf"), self.settings.pdf)
        except RenderError as error:
            self.pdf_error = str(error)
        else:
            self.pdf_error = ""
        self.catalogue.finish_run(run_id, "done", document_path=str(target))
        return self.catalogue.run(run_id)

    def pdf(self, record: RunRecord) -> bytes:
        """The typeset document, or empty when it could not be produced."""
        if not record.document_path:
            return b""
        candidate = Path(record.document_path).with_suffix(".pdf")
        return candidate.read_bytes() if candidate.exists() else b""

    def fail(self, run_id: str, error: str) -> None:
        self.catalogue.finish_run(run_id, "failed", error=error)


def build(settings: Settings) -> Library:
    return Library(
        settings=settings,
        catalogue=Catalogue(settings.paths.catalogue),
        places=Places(settings),
        vectors=VectorStore(settings),
        llm=llm_backends.build(settings),
        embeddings=embedding_backends.build(settings),
    )


def _safe(name: str, allowed: set[str]) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in allowed:
        raise ServiceError(f"{name} must be one of {sorted(allowed)}")
    return f"deck{suffix}"
