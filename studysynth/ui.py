"""The whole interface: a home page of courses, and a page per course."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from studysynth.config import Settings, load
from studysynth.models import ChapterRange, Lecture, NotePage, RetrievalOutcome, RunRecord
from studysynth.pipeline.render import provenance_report, to_html
from studysynth.services import Library, ServiceError, build
from studysynth.store import StoreError

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
def library() -> tuple[Settings, Library]:
    """Built once per process: it holds the embedding model and the open index."""
    settings = load()
    return settings, build(settings)


def go(**params: str) -> None:
    st.query_params.clear()
    for key, value in params.items():
        if value:
            st.query_params[key] = value
    st.rerun()


# --- home -------------------------------------------------------------------


def home(shelf: Library) -> None:
    st.title("Study Note Synthesizer")
    st.caption(
        "Your handwritten notes, completed against your professor's slides, "
        "with the textbook allowed in only where they leave a gap."
    )

    courses = shelf.courses()
    if courses:
        for row in _rows(courses, per_row=3):
            for column, course in zip(st.columns(3), row):
                with column:
                    if course is None:
                        continue
                    _course_box(shelf, course)
    else:
        st.info("No courses yet. Create one below.")

    st.divider()
    with st.form("new_course", clear_on_submit=True):
        st.subheader("New course")
        left, right = st.columns([1, 2])
        identifier = left.text_input("Short id", placeholder="cs3340")
        title = right.text_input("Name", placeholder="Analysis of Algorithms")
        if st.form_submit_button("Create course", type="primary") and identifier.strip():
            shelf.add_course(identifier.strip(), title.strip() or identifier.strip())
            go(course=identifier.strip())


def _course_box(shelf: Library, course: dict[str, str]) -> None:
    identifier = course["id"]
    with st.container(border=True):
        st.markdown(f"### {course['title']}")
        book = shelf.textbook_status(identifier)
        lectures = shelf.lectures(identifier)
        st.caption(
            f"{'Textbook indexed' if book else 'No textbook'} · "
            f"{len(lectures)} lecture{'s' if len(lectures) != 1 else ''}"
        )
        if st.button("Open", key=f"open-{identifier}", use_container_width=True):
            go(course=identifier)


def _rows(items: list[Any], per_row: int) -> list[list[Any]]:
    padded = items + [None] * (-len(items) % per_row)
    return [padded[i : i + per_row] for i in range(0, len(padded), per_row)]


# --- course -----------------------------------------------------------------


def course_page(shelf: Library, course: str) -> None:
    title = next((c["title"] for c in shelf.courses() if c["id"] == course), course)
    if st.button("← All courses"):
        go()
    st.title(title)

    textbook_section(shelf, course)
    st.divider()
    lectures_section(shelf, course)


def textbook_section(shelf: Library, course: str) -> None:
    st.subheader("Textbook")
    st.caption("Optional, and indexed once. Every later run reuses the same index.")

    book = shelf.textbook_status(course)
    if book:
        st.success(
            f"**{book['filename']}** · {book['chunks']} chunks indexed · "
            f"{str(book['indexed_at'])[:10]}"
        )
        with st.expander("Chapter page ranges"):
            _chapter_editor(shelf, course)
        if st.button("Replace the textbook"):
            st.session_state["replace_book"] = True

    if not book or st.session_state.get("replace_book"):
        upload = st.file_uploader("Textbook PDF", type="pdf", key="book")
        if upload is not None and st.button("Index it", type="primary"):
            with st.spinner("Parsing, chunking and embedding. This runs once."):
                path = shelf.store_textbook(course, upload.getbuffer().tobytes())
                chapters, chunks = shelf.index_textbook(course, path, upload.name)
            st.session_state.pop("replace_book", None)
            st.success(f"{chapters} chapters, {chunks} chunks indexed.")
            st.rerun()


def _chapter_editor(shelf: Library, course: str) -> None:
    chapters = shelf.catalogue.chapters(course)
    if not chapters:
        return
    st.caption("A wrong range silently narrows what the textbook may be searched for.")
    edited = st.data_editor(
        [
            {"chapter": c.title or c.chapter, "first page": c.page_start, "last page": c.page_end}
            for c in chapters
        ],
        hide_index=True,
        use_container_width=True,
        key=f"ranges-{course}",
    )
    if st.button("Save ranges and re-index"):
        with st.spinner("Re-embedding with the corrected ranges."):
            chunks = shelf.reindex(
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
        st.success(f"Re-indexed, {chunks} chunks.")
        st.rerun()


def _page(value: object, fallback: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return fallback


def lectures_section(shelf: Library, course: str) -> None:
    st.subheader("Lectures")
    lectures = shelf.lectures(course)
    chapters = {c.title or c.chapter: c.chapter for c in shelf.catalogue.chapters(course)}

    for lecture in lectures:
        _lecture_box(shelf, course, lecture)

    with st.form("new_lecture", clear_on_submit=True):
        st.markdown("**New lecture**")
        left, right = st.columns([2, 2])
        name = left.text_input("Name", placeholder="Week 3 — Induction")
        chapter = right.selectbox("Textbook chapter", ["(none)"] + list(chapters))
        if st.form_submit_button("Create lecture") and name.strip():
            shelf.add_lecture(course, name.strip(), chapters.get(chapter, ""))
            st.rerun()


def _lecture_box(shelf: Library, course: str, lecture: Lecture) -> None:
    history = shelf.history(course, lecture.id)
    with st.container(border=True):
        head, action = st.columns([4, 1])
        head.markdown(f"### {lecture.title}")
        head.caption(
            f"{lecture.slides_name or 'no slides'} · "
            f"{lecture.note_count} note page{'s' if lecture.note_count != 1 else ''} · "
            f"{len(history)} run{'s' if len(history) != 1 else ''}"
        )
        if action.button("Open", key=f"open-{lecture.id}", use_container_width=True):
            go(course=course, lecture=lecture.id)


# --- lecture ----------------------------------------------------------------


def lecture_page(shelf: Library, course: str, lecture: Lecture) -> None:
    if st.button("← Back to the course"):
        go(course=course)
    st.title(lecture.title)

    run_id = st.session_state.get("run_id", "")
    if run_id:
        watch(shelf, run_id)
        return

    uploads_section(shelf, course, lecture)
    st.divider()
    start_section(shelf, course, lecture)
    st.divider()
    history_section(shelf, course, lecture)


def uploads_section(shelf: Library, course: str, lecture: Lecture) -> None:
    st.subheader("Sources")
    left, right = st.columns(2)

    with left:
        st.markdown("**Professor's lecture PDF**")
        if lecture.slides_name:
            st.success(lecture.slides_name)
        deck = st.file_uploader(
            "Slides", type=["pdf", "pptx"], key=f"deck-{lecture.id}", label_visibility="collapsed"
        )
        if deck is not None and st.button("Save slides", key=f"save-deck-{lecture.id}"):
            shelf.replace_slides(course, lecture.id, deck.name, deck.getbuffer().tobytes())
            st.rerun()

    with right:
        st.markdown("**Your handwritten notes**")
        if lecture.note_count:
            st.success(f"{lecture.note_count} pages saved")
        st.caption("One PDF of any length — a GoodNotes export or a scan. Replacing it removes the old one.")
        notes = st.file_uploader(
            "Notes", type=["pdf"], key=f"notes-{lecture.id}", label_visibility="collapsed",
        )
        if notes is not None and st.button("Save notes", key=f"save-notes-{lecture.id}"):
            try:
                pages = shelf.replace_notes(
                    course, lecture.id, notes.name, notes.getbuffer().tobytes()
                )
            except ServiceError as error:
                st.error(str(error))
            else:
                st.success(f"{pages} pages saved, the previous notes removed.")
                st.rerun()


def start_section(shelf: Library, course: str, lecture: Lecture) -> None:
    if not lecture.ready:
        st.info("Upload both the slides and your note photos to build this lecture.")
        return
    if not lecture.chapter:
        st.warning("No textbook chapter set, so the textbook will not be consulted.")
    if st.button("Build the study document", type="primary"):
        try:
            run_id, stream = shelf.start(course, lecture)
        except ServiceError as error:
            st.error(str(error))
            return
        st.session_state["run_id"] = run_id
        consume(shelf, run_id, stream)
        st.rerun()


def history_section(shelf: Library, course: str, lecture: Lecture) -> None:
    history = shelf.history(course, lecture.id)
    st.subheader(f"Past runs ({len(history)})")
    if not history:
        st.caption("Every run you build is kept here, so you can compare or download any of them.")
        return
    for record in history:
        stamp = record.created_at.strftime("%d %b %Y, %H:%M")
        with st.expander(f"{stamp} — {record.status}", expanded=False):
            if record.error:
                st.error(record.error)
                continue
            markdown = shelf.document(record)
            if not markdown:
                st.caption("No document was produced.")
                continue
            st.download_button(
                "Download this version", markdown,
                file_name=f"{lecture.id}-{record.created_at:%Y%m%d-%H%M}.md",
                mime="text/markdown", key=f"dl-{record.id}",
            )
            st.html(to_html(markdown, title=lecture.title))


# --- a run in flight --------------------------------------------------------


def consume(shelf: Library, run_id: str, stream: Any) -> None:
    with st.status("Running", expanded=True) as status:
        try:
            for node, update in stream:
                if node in STEPS:
                    st.write(f"{STEPS[node]} — {_detail(update)}")
        except Exception as error:  # noqa: BLE001 - shown on the page, run marked failed
            shelf.fail(run_id, str(error))
            status.update(label="Failed", state="error")
            st.exception(error)
            return
        if shelf.awaiting_review(run_id):
            status.update(label="Waiting for you to check the transcript", state="complete")
        else:
            shelf.finish(run_id)
            status.update(label="Done", state="complete")


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


def watch(shelf: Library, run_id: str) -> None:
    try:
        pending = shelf.awaiting_review(run_id)
    except StoreError:
        st.session_state.pop("run_id", None)
        st.rerun()
        return

    if pending:
        review(shelf, run_id)
    else:
        finished(shelf, run_id)

    if st.button("Close this run"):
        st.session_state.pop("run_id", None)
        st.rerun()


def review(shelf: Library, run_id: str) -> None:
    """The interrupt: handwriting OCR is the least reliable input, so it is corrected here."""
    st.subheader("Check the transcript")
    st.caption("Anything wrong here is wrong in the finished document.")

    pages: list[NotePage] = list(shelf.runner.state(run_id).get("notes", []))
    if not pages:
        st.warning("No transcript yet.")
        return

    edited: list[dict[str, Any]] = []
    for tab, page in zip(st.tabs([f"Page {p.page}" for p in pages]), pages):
        with tab:
            left, right = st.columns(2)
            if Path(page.image_path).exists():
                left.image(page.image_path, use_container_width=True)
            text = right.text_area(
                "Transcript", page.markdown, height=520, key=f"note-{run_id}-{page.page}"
            )
            edited.append(
                page.model_copy(
                    update={"markdown": text, "edited_by_student": text != page.markdown}
                ).model_dump()
            )

    if st.button("Looks right, carry on", type="primary"):
        consume(shelf, run_id, shelf.resume(run_id, edited))
        st.rerun()


def finished(shelf: Library, run_id: str) -> None:
    record: RunRecord = shelf.catalogue.run(run_id)
    markdown = shelf.document(record)
    if not markdown:
        st.warning("This run produced no document.")
        return

    st.subheader("Your study document")
    highlight = st.toggle("Show where each passage came from", value=True)
    if highlight:
        st.caption(
            "Highlighted passages came from the textbook and carry a page reference. "
            "**Check this:** marks where your notes and the slides disagree."
        )
    st.download_button(
        "Download", markdown, file_name=f"{record.chapter or record.lecture}.md",
        mime="text/markdown",
    )
    st.html(to_html(markdown, title=record.chapter, highlight=highlight))

    outcome = shelf.runner.state(run_id).get("retrieval") or RetrievalOutcome()
    with st.expander("What the textbook gate rejected"):
        st.markdown(provenance_report(record.course, record.chapter, outcome))


# --- routing ----------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Study Note Synthesizer", layout="wide")
    _settings, shelf = library()

    course = st.query_params.get("course", "")
    lecture_id = st.query_params.get("lecture", "")

    if not course:
        home(shelf)
        return
    if not lecture_id:
        course_page(shelf, course)
        return
    try:
        lecture = shelf.catalogue.lecture(course, lecture_id)
    except StoreError:
        go(course=course)
        return
    lecture_page(shelf, course, lecture)


main()
