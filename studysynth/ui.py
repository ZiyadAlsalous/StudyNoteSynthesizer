"""The whole interface: a home page of courses, and a page per course."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import streamlit as st

from studysynth.config import Settings, load
from studysynth.library import Library, ServiceError, build
from studysynth.models import ChapterRange, Lecture, RetrievalOutcome, RunRecord
from studysynth.pipeline.render import provenance_report, to_html
from studysynth.store import IndexBusy, StoreError

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


def home(shelf: Library) -> None:
    st.title("Study Note Synthesizer")
    st.caption(
        "Your handwritten notes, completed against your professor's slides, "
        "with the textbook allowed in only where they leave a gap."
    )

    courses = shelf.courses()
    if courses:
        for row in _rows(courses, per_row=3):
            for column, course in zip(st.columns(3), row, strict=True):
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
        identifier = left.text_input("Course code", placeholder="Add course code")
        title = right.text_input("Course name", placeholder="Add course name")
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
        if st.button("Open", key=f"open-{identifier}", width="stretch"):
            go(course=identifier)


def _rows(items: list[Any], per_row: int) -> list[list[Any]]:
    padded = items + [None] * (-len(items) % per_row)
    return [padded[i : i + per_row] for i in range(0, len(padded), per_row)]


def course_page(shelf: Library, course: str) -> None:
    title = next((c["title"] for c in shelf.courses() if c["id"] == course), course)
    if st.button("← All courses"):
        go()
    st.title(title)

    textbook_section(shelf, course)
    st.divider()
    lectures_section(shelf, course)
    st.divider()
    delete_course_section(shelf, course)


def delete_course_section(shelf: Library, course: str) -> None:
    """Deleting a course cannot be undone, so the id must be typed to confirm."""
    with st.expander("Delete this course"):
        st.warning(
            "This deletes the textbook, every lecture with its slides and notes, and every "
            "study document built for this course. It cannot be undone."
        )
        typed = st.text_input(f"Type {course} to confirm", key=f"confirm-{course}")
        if st.button("Delete course", type="primary", disabled=typed.strip() != course):
            shelf.delete_course(course)
            go()


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
        width="stretch",
        key=f"ranges-{course}",
    )
    if st.button("Save ranges and re-index"):
        with st.spinner("Re-embedding with the corrected ranges."):
            chunks = shelf.reindex(
                course,
                [
                    ChapterRange(
                        chapter=old.chapter,
                        title=old.title,
                        page_start=_page(row["first page"], old.page_start),
                        page_end=_page(row["last page"], old.page_end),
                        manual_override=True,
                    )
                    for old, row in zip(chapters, edited, strict=True)
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
        name = left.text_input("Lecture name", placeholder="Add lecture name")
        chapter = right.selectbox(
            "Textbook chapter",
            ["Find it automatically"] + list(chapters),
            help="Leave this alone unless you want to pin the search to one chapter.",
        )
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
        if action.button("Open", key=f"open-{lecture.id}", width="stretch"):
            go(course=course, lecture=lecture.id)


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
    files = shelf.sources(course, lecture.id)
    left, right = st.columns(2)
    with left:
        _source_column(
            shelf,
            course,
            lecture,
            kind="slides",
            files=files["slides"],
            heading="**Professor's lecture slides**",
            types=["pdf", "pptx"],
            caption="PDF or PowerPoint. Add more decks at any time; each is read once.",
        )
    with right:
        _source_column(
            shelf,
            course,
            lecture,
            kind="notes",
            files=files["notes"],
            heading="**Your handwritten notes**",
            types=["pdf"],
            caption="PDFs of any length. Only a new or changed file goes through OCR.",
        )


def _source_column(
    shelf: Library,
    course: str,
    lecture: Lecture,
    *,
    kind: str,
    files: list[str],
    heading: str,
    types: list[str],
    caption: str,
) -> None:
    """One kind of source: what is saved, a remove button each, and an uploader."""
    st.markdown(heading)
    if not files:
        st.info("Nothing uploaded yet")
    for name in files:
        label, view, remove = st.columns([4, 1, 1])
        label.success(name)
        shown = f"view-{kind}-{lecture.id}-{name}"
        if view.button("Hide" if st.session_state.get(shown) else "View", key=f"b{shown}"):
            st.session_state[shown] = not st.session_state.get(shown, False)
            st.rerun()
        if remove.button("Remove", key=f"rm-{kind}-{lecture.id}-{name}"):
            shelf.remove_source(course, lecture.id, kind, name)
            st.session_state.pop(shown, None)
            st.rerun()
        if st.session_state.get(shown):
            _source_view(shelf, course, lecture, kind, name)

    # A new key after each save empties the uploader.
    round_key = f"round-{kind}-{lecture.id}"
    uploads = st.file_uploader(
        kind.title(),
        type=types,
        accept_multiple_files=True,
        key=f"{kind}-{lecture.id}-{st.session_state.get(round_key, 0)}",
        label_visibility="collapsed",
    )
    st.caption(f"{caption} A file with the same name replaces the old one.")
    if uploads and st.button(f"Save {kind}", key=f"save-{kind}-{lecture.id}"):
        try:
            for upload in uploads:
                data = upload.getbuffer().tobytes()
                if kind == "slides":
                    shelf.add_slides(course, lecture.id, upload.name, data)
                else:
                    shelf.add_notes(course, lecture.id, upload.name, data)
        except ServiceError as error:
            st.error(str(error))
        else:
            st.session_state[round_key] = st.session_state.get(round_key, 0) + 1
            st.rerun()


def _source_view(shelf: Library, course: str, lecture: Lecture, kind: str, name: str) -> None:
    """Which file this is, whether it has been read, and what was read from each page."""
    view = shelf.source_details(course, lecture.id, kind, name)
    with st.container(border=True):
        if view.stored:
            st.caption(f"{len(view.pages)} pages read and stored. The next build reuses them.")
        else:
            st.caption("Not read yet. It is read once, on the next build.")
        st.download_button(
            "Download the original",
            view.path.read_bytes(),
            file_name=view.path.name,
            key=f"dl-{kind}-{lecture.id}-{name}",
        )
        if not view.pages:
            return
        number = 1
        if len(view.pages) > 1:
            number = st.slider("Page", 1, len(view.pages), 1, key=f"pg-{kind}-{lecture.id}-{name}")
        page = view.pages[number - 1]
        if kind == "notes" and number <= len(view.images):
            picture, text = st.columns(2)
            picture.image(view.images[number - 1], width="stretch")
            text.markdown(page.markdown)
            if page.corrected:
                text.caption("Corrected by you at review.")
        else:
            st.markdown(page.markdown or "_No text on this slide._")


def start_section(shelf: Library, course: str, lecture: Lecture) -> None:
    if not lecture.ready:
        st.info("Upload the slides and at least one notes PDF to build this lecture.")
        return
    if not lecture.chapter:
        st.caption(
            "No chapter pinned, so the textbook chapters closest to your notes are "
            "found automatically."
        )
    if st.button("Build the study document", type="primary"):
        try:
            run_id, stream = shelf.start(course, lecture)
        except ServiceError as error:
            st.error(str(error))
            return
        st.session_state["run_id"] = run_id
        consume(shelf, run_id, stream)
        st.rerun()


def _downloads(
    shelf: Library,
    record: RunRecord,
    markdown: str,
    *,
    stem: str,
    key: str = "",
    side_by_side: bool = False,
) -> None:
    """The PDF and Markdown buttons for one run, wherever a run is shown."""
    typeset = shelf.pdf(record)
    suffix = f"-{key}" if key else ""
    left, right = st.columns(2) if side_by_side else (nullcontext(), nullcontext())

    with left:
        if typeset:
            st.download_button(
                "Download PDF",
                typeset,
                file_name=f"{stem}.pdf",
                mime="application/pdf",
                type="primary",
                key=f"pdf{suffix}",
            )
        elif side_by_side:
            st.caption(shelf.pdf_error or "No typeset PDF for this run.")
    with right:
        st.download_button(
            "Download Markdown",
            markdown,
            file_name=f"{stem}.md",
            mime="text/markdown",
            key=f"md{suffix}",
        )


def history_section(shelf: Library, course: str, lecture: Lecture) -> None:
    history = shelf.history(course, lecture.id)
    st.subheader(f"Past runs ({len(history)})")
    if not history:
        st.caption("Every run you build is kept here, so you can compare or download any of them.")
        return
    for record in history:
        stamp = record.created_at.strftime("%d %b %Y, %H:%M")
        with st.expander(f"{stamp} · {record.status}", expanded=False):
            if record.error:
                st.error(record.error)
                continue
            markdown = shelf.document(record)
            if not markdown:
                st.caption("No document was produced.")
                continue
            _downloads(
                shelf,
                record,
                markdown,
                stem=f"{lecture.id}-{record.created_at:%Y%m%d-%H%M}",
                key=record.id,
            )
            st.html(to_html(markdown, title=lecture.title))


def consume(shelf: Library, run_id: str, stream: Any) -> None:
    with st.status("Running", expanded=True) as status:
        try:
            for node, update in stream:
                if node in STEPS:
                    st.write(f"{STEPS[node]}: {_detail(update)}")
        except Exception as error:
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

    pages = shelf.notes(run_id)
    if not pages:
        st.warning("No transcript yet.")
        return

    edited: list[dict[str, Any]] = []
    for tab, page in zip(st.tabs([f"Page {p.page}" for p in pages]), pages, strict=True):
        with tab:
            left, right = st.columns(2)
            if Path(page.image_path).exists():
                left.image(page.image_path, width="stretch")
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
        if record.error:
            st.error(record.error)
        return

    st.subheader("Your study document")
    highlight = st.toggle("Show where each passage came from", value=True)
    if highlight:
        st.caption(
            "Highlighted passages came from the textbook and carry a page reference; blue "
            "ones are model explanations, used only when the course has no textbook. "
            "**Check this:** marks where your notes and the slides disagree."
        )
    _downloads(shelf, record, markdown, stem=record.chapter or record.lecture, side_by_side=True)
    st.caption("The PDF typesets the mathematics. The Markdown keeps it as LaTeX source.")
    st.html(to_html(markdown, title=record.chapter, highlight=highlight))

    outcome = shelf.runner.state(run_id).get("retrieval") or RetrievalOutcome()
    if outcome.auto_scoped and outcome.chapters:
        st.caption(f"Textbook chapters matched to your notes: {', '.join(outcome.chapters)}")
    with st.expander("What the textbook gate rejected"):
        st.markdown(provenance_report(record.course, record.chapter, outcome))


def main() -> None:
    st.set_page_config(page_title="Study Note Synthesizer", layout="wide")
    try:
        _settings, shelf = library()
    except IndexBusy as error:
        st.title("Study Note Synthesizer")
        st.error(str(error))
        st.caption(
            "The textbook index is a single-process store, so only one copy of "
            "the app can run at a time. Everything you have saved is intact."
        )
        if st.button("Try again"):
            library.clear()
            st.rerun()
        return

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
