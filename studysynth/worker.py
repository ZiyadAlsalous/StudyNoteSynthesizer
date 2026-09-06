"""Background job runner and the one place services are wired together.

A chapter run takes minutes, so it cannot happen inside a request handler.
Progress is published to a queue per run and the API drains it over SSE.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import embeddings as embedding_backends
from . import llm as llm_backends
from .config import Settings
from .embeddings import EmbeddingBackend
from .graph import Nodes, Runner
from .ingest import TextbookIngestor, chapter_ranges
from .llm import LlmClient
from .models import ChapterRange, RetrievalOutcome
from .render import provenance_report
from .retrieval import TextbookGate
from .store import Catalogue, Places, VectorStore


class JobError(RuntimeError):
    pass


@dataclass
class Progress:
    run_id: str
    node: str
    detail: str = ""
    done: bool = False
    error: str | None = None


@dataclass
class Services:
    """Constructed once at startup. Nothing below this layer knows it exists."""

    settings: Settings
    catalogue: Catalogue
    places: Places
    vectors: VectorStore
    llm: LlmClient
    embeddings: EmbeddingBackend
    runner: Runner = field(init=False)

    def __post_init__(self) -> None:
        gate = TextbookGate(
            self.settings, self.llm, self.embeddings, self.vectors, self.catalogue
        )
        nodes = Nodes(
            self.settings, self.llm, self.embeddings, gate, self.catalogue, self.places
        )
        self.runner = Runner(self.settings, nodes)


def build_services(settings: Settings) -> Services:
    return Services(
        settings=settings,
        catalogue=Catalogue(settings.paths.catalogue),
        places=Places(settings),
        vectors=VectorStore(settings),
        llm=llm_backends.build(settings),
        embeddings=embedding_backends.build(settings),
    )


class Worker:
    """One thread per run. Small by design: the state of record is the
    checkpointer, not anything held here."""

    def __init__(self, services: Services) -> None:
        self._services = services
        self._queues: dict[str, queue.Queue[Progress]] = {}
        self._threads: dict[str, threading.Thread] = {}

    def channel(self, run_id: str) -> queue.Queue[Progress]:
        return self._queues.setdefault(run_id, queue.Queue())

    def ingest_textbook(self, course: str, pdf: Path, ranges: list[ChapterRange] | None) -> int:
        """Runs inline: ingestion is slow but the UI blocks on its result anyway."""
        settings = self._services.settings
        spans = ranges if ranges is not None else chapter_ranges(pdf)
        self._services.catalogue.set_chapters(course, spans)
        parents, chunks = TextbookIngestor(settings).ingest(pdf, course, spans)
        if not chunks:
            raise JobError(f"{pdf.name} produced no chunks for {course}")
        self._services.catalogue.put_parents(parents)
        self._services.vectors.create(course, self._services.embeddings.dimensions)
        vectors = self._services.embeddings.embed_documents([chunk.text for chunk in chunks])
        return self._services.vectors.upsert(course, chunks, vectors.tolist())

    def start(self, run_id: str, course: str, chapter: str, slides: Path, notes: Path) -> None:
        if run_id in self._threads and self._threads[run_id].is_alive():
            raise JobError(f"Run {run_id} is already in flight")
        self._services.catalogue.start_run(run_id, course, chapter)
        thread = threading.Thread(
            target=self._execute,
            args=(run_id, {"course": course, "chapter": chapter,
                           "slides_dir": str(slides), "notes_dir": str(notes)}),
            daemon=True,
            name=f"run-{run_id}",
        )
        self._threads[run_id] = thread
        thread.start()

    def resume(self, run_id: str, edited_notes: list[dict[str, Any]] | None = None) -> None:
        self._services.runner.approve_notes(run_id, edited_notes)
        thread = threading.Thread(
            target=self._execute, args=(run_id, None), daemon=True, name=f"run-{run_id}"
        )
        self._threads[run_id] = thread
        thread.start()

    def _execute(self, run_id: str, inputs: dict[str, Any] | None) -> None:
        channel = self.channel(run_id)
        try:
            stream = (
                self._services.runner.stream(run_id, **inputs)
                if inputs is not None
                else self._services.runner.stream(run_id)
            )
            for node, update in stream:
                channel.put(Progress(run_id=run_id, node=node, detail=_describe(update)))
        except Exception as error:  # noqa: BLE001 - reported, then re-raised into the record
            self._services.catalogue.finish_run(run_id, "failed", error=str(error))
            channel.put(Progress(run_id=run_id, node="error", error=traceback.format_exc(limit=3),
                                 done=True))
            return

        pending = self._services.runner.pending(run_id)
        if pending:
            channel.put(Progress(run_id=run_id, node="awaiting_review", detail=pending[0]))
            return

        self._finish(run_id, channel)

    def _finish(self, run_id: str, channel: queue.Queue[Progress]) -> None:
        state = self._services.runner.state(run_id)
        record = self._services.catalogue.run(run_id)
        document = state.get("document", "")
        target = self._services.places.artifact(run_id, "document.md")
        target.write_text(document, encoding="utf-8")

        outcome = state.get("retrieval") or RetrievalOutcome()
        self._services.places.artifact(run_id, "provenance.md").write_text(
            provenance_report(record.course, record.chapter, outcome), encoding="utf-8"
        )
        self._services.catalogue.finish_run(run_id, "done", document_path=str(target))
        channel.put(Progress(run_id=run_id, node="done", detail=str(target), done=True))

    def events(self, run_id: str, timeout: float = 30.0) -> Iterator[Progress]:
        channel = self.channel(run_id)
        while True:
            try:
                event = channel.get(timeout=timeout)
            except queue.Empty:
                return
            yield event
            if event.done or event.node == "awaiting_review":
                return


def _describe(update: dict[str, Any]) -> str:
    for key in ("slides", "notes", "concepts", "gaps", "drafts"):
        if key in update:
            return f"{len(update[key])} {key}"
    if "retrieval" in update:
        outcome = update["retrieval"]
        return f"{len(outcome.admitted)} admitted, {len(outcome.rejections)} rejected"
    if "document" in update:
        return f"{len(update['document'])} characters"
    return ""
