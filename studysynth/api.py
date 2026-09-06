"""FastAPI routes. Thin by rule: validate, call one function, return.

If a route grows a second decision, that decision belongs in worker.py or below.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable, Iterator, TypeVar

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings, load
from .models import ChapterRange, NotePage, RunRecord
from .render import provenance_report, to_html
from .store import StoreError
from .worker import Services, Worker, build_services

app = FastAPI(title="Study Note Synthesizer")

_services: Services | None = None
_worker: Worker | None = None


def services() -> Services:
    global _services
    if _services is None:
        _services = build_services(load())
    return _services


def worker() -> Worker:
    global _worker
    if _worker is None:
        _worker = Worker(services())
    return _worker


Wired = Annotated[Services, Depends(services)]
Jobs = Annotated[Worker, Depends(worker)]


class CourseIn(BaseModel):
    id: str
    title: str


class ChaptersIn(BaseModel):
    chapters: list[ChapterRange]


class RunIn(BaseModel):
    slides_dir: str
    notes_dir: str


class NotesIn(BaseModel):
    notes: list[dict[str, Any]] | None = None


@app.post("/courses", status_code=201)
def create_course(body: CourseIn, wired: Wired) -> dict[str, str]:
    wired.catalogue.add_course(body.id, body.title)
    return {"id": body.id}


@app.get("/courses")
def list_courses(wired: Wired) -> list[dict[str, str]]:
    return wired.catalogue.courses()


@app.post("/courses/{course}/textbook")
async def upload_textbook(course: str, file: UploadFile, wired: Wired, jobs: Jobs) -> dict[str, int]:
    target = wired.places.textbook(course)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"chunks": jobs.ingest_textbook(course, target, ranges=None)}


@app.get("/courses/{course}/chapters")
def list_chapters(course: str, wired: Wired) -> list[ChapterRange]:
    return wired.catalogue.chapters(course)


@app.put("/courses/{course}/chapters")
def set_chapters(course: str, body: ChaptersIn, wired: Wired, jobs: Jobs) -> dict[str, int]:
    """The manual override for when the PDF outline is missing or wrong."""
    return {"chunks": jobs.ingest_textbook(course, wired.places.textbook(course), body.chapters)}


@app.post("/courses/{course}/chapters/{chapter}/runs", status_code=202)
def start_run(course: str, chapter: str, body: RunIn, jobs: Jobs) -> dict[str, str]:
    run_id = uuid.uuid4().hex[:12]
    jobs.start(run_id, course, chapter, Path(body.slides_dir), Path(body.notes_dir))
    return {"run_id": run_id}


@app.get("/runs/{run_id}")
def read_run(run_id: str, wired: Wired) -> RunRecord:
    return _found(lambda: wired.catalogue.run(run_id))


@app.get("/runs/{run_id}/events")
def stream_events(run_id: str, jobs: Jobs) -> StreamingResponse:
    def lines() -> Iterator[str]:
        for event in jobs.events(run_id):
            yield f"data: {json.dumps(event.__dict__)}\n\n"

    return StreamingResponse(lines(), media_type="text/event-stream")


@app.get("/runs/{run_id}/notes")
def read_notes(run_id: str, wired: Wired) -> list[NotePage]:
    """The OCR transcript the review screen edits."""
    state = wired.runner.state(run_id)
    notes = state.get("notes")
    if not notes:
        raise HTTPException(status_code=409, detail="Run has not transcribed any notes yet")
    return list(notes)


@app.get("/runs/{run_id}/notes/{page}/image")
def read_note_image(run_id: str, page: int, wired: Wired) -> FileResponse:
    state = wired.runner.state(run_id)
    match = [note for note in state.get("notes", []) if note.page == page]
    if not match:
        raise HTTPException(status_code=404, detail=f"No note page {page} in run {run_id}")
    return FileResponse(match[0].image_path)


@app.post("/runs/{run_id}/approve", status_code=202)
def approve_notes(run_id: str, body: NotesIn, jobs: Jobs) -> dict[str, str]:
    jobs.resume(run_id, body.notes)
    return {"run_id": run_id}


@app.get("/runs/{run_id}/document")
def read_document(run_id: str, wired: Wired, highlight: bool = True) -> dict[str, str]:
    record: RunRecord = _found(lambda: wired.catalogue.run(run_id))
    if not record.document_path:
        raise HTTPException(status_code=409, detail="Run has not produced a document")
    markdown = Path(record.document_path).read_text(encoding="utf-8")
    return {
        "markdown": markdown,
        "html": to_html(markdown, title=record.chapter, highlight=highlight),
    }


@app.get("/runs/{run_id}/provenance")
def read_provenance(run_id: str, wired: Wired) -> dict[str, str]:
    record: RunRecord = _found(lambda: wired.catalogue.run(run_id))
    state = wired.runner.state(run_id)
    outcome = state.get("retrieval")
    if outcome is None:
        raise HTTPException(status_code=409, detail="Run has not reached retrieval")
    return {"report": provenance_report(record.course, record.chapter, outcome)}


T = TypeVar("T")


def _found(call: Callable[[], T]) -> T:
    """Turn a missing row into a 404 instead of a 500."""
    try:
        return call()
    except StoreError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


WEB_DIST = Path(__file__).parent.parent / "web" / "dist"


def mount_frontend() -> bool:
    """Serve the built single-page app under the API routes.

    Mounted last so every route above still wins; `html=True` sends index.html
    for unknown paths. Absent in development, where Vite serves the app.
    """
    if not (WEB_DIST / "index.html").exists():
        return False
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
    return True


def serve(settings: Settings | None = None) -> None:
    import uvicorn

    resolved = settings or load()
    if not mount_frontend():
        logging.getLogger(__name__).warning(
            "No built frontend at %s; serving the API only. Run `npm run build` in web/.",
            WEB_DIST,
        )
    uvicorn.run(app, host=resolved.server.host, port=resolved.server.port)
