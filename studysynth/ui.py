"""Streamlit interface. The whole UI, in one file.

Calls the pipeline in process rather than over HTTP: there is no second server
to keep running, and a run is a generator this script consumes directly.

Run with: streamlit run studysynth/ui.py
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from studysynth.config import Settings, load
from studysynth.graph import EXTRACT_CONCEPTS
from studysynth.ingest import OutlineMissing, chapter_ranges
from studysynth.models import ChapterRange, NotePage, RetrievalOutcome
from studysynth.render import provenance_report, to_html
from studysynth.store import StoreError
from studysynth.worker import Services, Worker, build_services

STEPS = {
    "ingest_slides": "Reading the slides",
    "ingest_notes": "Transcribing your notes",
    "extract_concepts": "Finding the concepts",
    "find_gaps": "Finding what the lecture leaves unresolved",
    "draft_concepts": "Drafting from slides and notes",
    "retrieve_textbook": "Consulting the textbook",
    "synthesize": "Writing the document",
    "verify": "Checking every claim",
}


@st.cache_resource
def services() -> tuple[Settings, Services, Worker]:
    """Built once per process. Holds the embedding model, so never per rerun."""
    settings = load()
    wired = build_services(settings)
    return settings, wired, Worker(wired)


def sidebar(settings: Settings, wired: Services, jobs: Worker) -> tuple[str, str]:
    st.sidebar.header("Course")
    courses = {row["title"]: row["id"] for row in wired.catalogue.courses()}

    with st.sidebar.form("new_course", clear_on_submit=True):
        identifier = st.text_input("id", placeholder="cs3340")
        title = st.text_input("title", placeholder="Analysis of Algorithms")
        if st.form_submit_button("Add course") and identifier:
            wired.catalogue.add_course(identifier, title or identifier)
            st.rerun()

    if not courses:
        st.sidebar.info("Add a course to begin.")
        return "", ""

    chosen = st.sidebar.selectbox("Course", list(courses))
    course = courses[chosen]

    st.sidebar.header("Textbook")
    upload = st.sidebar.file_uploader("Textbook PDF", type="pdf")
    if upload is not None and st.sidebar.button("Index this textbook"):
        target = wired.places.textbook(course)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(upload.getbuffer())
        _index(settings, wired, jobs, course, target)

    chapters = wired.catalogue.chapters(course)
    if not chapters:
        st.sidebar.warning("No textbook indexed yet.")
        return course, ""

    with st.sidebar.expander(f"{len(chapters)} chapters — check the page ranges"):
        edited = st.data_editor(
            [
                {"chapter": c.title or c.chapter, "first page": c.page_start,
                 "last page": c.page_end}
                for c in chapters
            ],
            hide_index=True,
            use_container_width=True,
        )
        if st.button("Save ranges and re-index"):
            wired.catalogue.set_chapters(
                course,
                [
                    ChapterRange(
                        chapter=old.chapter, title=old.title,
                        page_start=_page(row["first page"], old.page_start),
                        page_end=_page(row["last page"], old.page_end),
                        manual_override=True,
                    )
                    for old, row in zip(chapters, edited)
                ],
            )
            _index(settings, wired, jobs, course, wired.places.textbook(course))

    labels = {c.title or c.chapter: c.chapter for c in chapters}
    return course, labels[st.sidebar.selectbox("Chapter", list(labels))]


def _page(value: object, fallback: int) -> int:
    """The data editor hands back whatever was typed, including blanks."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return fallback


def _index(settings: Settings, wired: Services, jobs: Worker, course: str, pdf: Path) -> None:
    with st.spinner("Parsing, chunking and embedding. This is the slow step."):
        try:
            ranges = chapter_ranges(pdf)
        except OutlineMissing:
            import pypdf

            pages = len(pypdf.PdfReader(str(pdf)).pages)
            st.warning(f"No outline in this PDF. Treating all {pages} pages as one chapter.")
            ranges = [
                ChapterRange(chapter="chapter-1", title="Chapter 1", page_start=1,
                             page_end=pages, manual_override=True)
            ]
        chunks = jobs.ingest_textbook(course, pdf, ranges)
    st.success(f"{len(ranges)} chapters, {chunks} chunks indexed.")
    st.rerun()


def run_chapter(wired: Services, course: str, chapter: str, slides: Path, notes: Path) -> None:
    run_id = uuid.uuid4().hex[:12]
    st.session_state.run_id = run_id
    wired.catalogue.start_run(run_id, course, chapter)
    _consume(
        wired, run_id,
        course=course, chapter=chapter, slides_dir=str(slides), notes_dir=str(notes),
    )


def _consume(wired: Services, run_id: str, **inputs: Any) -> None:
    """Drive the graph to its next stop, showing each node as it completes."""
    with st.status("Running", expanded=True) as status:
        try:
            for node, update in wired.runner.stream(run_id, **inputs):
                if node in STEPS:
                    st.write(f"{STEPS[node]} — {_detail(update)}")
        except Exception as error:  # noqa: BLE001 - surfaced to the page, run marked failed
            wired.catalogue.finish_run(run_id, "failed", error=str(error))
            status.update(label="Failed", state="error")
            st.exception(error)
            return
        pending = wired.runner.pending(run_id)
        status.update(
            label="Waiting for you to check the transcript" if pending else "Done",
            state="complete",
        )
    if not pending:
        _finish(wired, run_id)


def _finish(wired: Services, run_id: str) -> None:
    state = wired.runner.state(run_id)
    document = state.get("document", "")
    target = wired.places.artifact(run_id, "document.md")
    target.write_text(document, encoding="utf-8")
    wired.catalogue.finish_run(run_id, "done", document_path=str(target))


def _detail(update: dict[str, Any]) -> str:
    for key in ("slides", "notes", "concepts", "gaps", "drafts"):
        if key in update:
            return f"{len(update[key])} {key}"
    if "retrieval" in update:
        outcome = update["retrieval"]
        return f"{len(outcome.admitted)} admitted, {len(outcome.rejections)} rejected"
    if "document" in update:
        return f"{len(update['document'])} characters"
    return ""


def review(wired: Services, run_id: str) -> None:
    """The interrupt. Handwriting OCR is the least reliable input, so it is
    corrected before anything is built on top of it."""
    st.subheader("Check the transcript")
    st.caption("Anything wrong here is wrong in the finished document.")

    pages: list[NotePage] = list(wired.runner.state(run_id).get("notes", []))
    if not pages:
        st.warning("No transcript yet.")
        return

    tabs = st.tabs([f"Page {page.page}" for page in pages])
    edited: list[dict[str, Any]] = []
    for tab, page in zip(tabs, pages):
        with tab:
            left, right = st.columns(2)
            with left:
                if Path(page.image_path).exists():
                    st.image(page.image_path, use_container_width=True)
            with right:
                text = st.text_area(
                    "Transcript", page.markdown, height=520, key=f"note-{page.page}"
                )
            edited.append(page.model_copy(
                update={"markdown": text, "edited_by_student": text != page.markdown}
            ).model_dump())

    if st.button("Looks right, carry on", type="primary"):
        wired.runner.approve_notes(run_id, edited)
        _consume(wired, run_id)
        st.rerun()


def document(wired: Services, run_id: str) -> None:
    record = wired.catalogue.run(run_id)
    state = wired.runner.state(run_id)
    markdown = state.get("document", "")
    if not markdown:
        return

    st.subheader("Your study document")
    highlight = st.toggle("Show where each passage came from", value=True)
    if highlight:
        st.caption(
            "Highlighted passages came from the textbook and carry a page reference. "
            "**Check this:** marks where your notes and the slides disagree."
        )
    st.html(to_html(markdown, title=record.chapter, highlight=highlight))

    outcome = state.get("retrieval") or RetrievalOutcome()
    with st.expander("What the textbook gate rejected"):
        st.markdown(provenance_report(record.course, record.chapter, outcome))

    st.download_button("Download the markdown", markdown, f"{record.chapter}.md", "text/markdown")


def main() -> None:
    st.set_page_config(page_title="Study Note Synthesizer", layout="wide")
    st.title("Study Note Synthesizer")

    settings, wired, jobs = services()
    course, chapter = sidebar(settings, wired, jobs)
    run_id = st.session_state.get("run_id", "")

    if run_id:
        try:
            pending = wired.runner.pending(run_id)
        except StoreError:
            pending = ()
        if EXTRACT_CONCEPTS in pending:
            review(wired, run_id)
        else:
            document(wired, run_id)
        if st.button("Start another chapter"):
            del st.session_state["run_id"]
            st.rerun()
        return

    if not (course and chapter):
        st.info("Add a course and index a textbook in the sidebar to begin.")
        return

    st.subheader("Build a chapter")
    slides = st.text_input("Slides folder", "inputs/slides")
    notes = st.text_input("Note photos folder", "inputs/notes")
    ready = Path(slides).is_dir() and Path(notes).is_dir()
    if not ready:
        st.caption("Both folders must exist. Notes must be images, one file per page.")
    if st.button("Build this chapter", type="primary", disabled=not ready):
        run_chapter(wired, course, chapter, Path(slides), Path(notes))
        st.rerun()


main()
