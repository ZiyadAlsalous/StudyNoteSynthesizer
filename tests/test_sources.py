"""Uploaded slides and notes are read once and stored.

A later run reads them back from the catalogue, so only a new or changed file
goes through PyMuPDF or OCR, and removing a file removes everything taken from it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studysynth.library import build
from studysynth.models import Reason, Rejection
from studysynth.pipeline.ingest import NoteIngestor, SlideIngestor
from studysynth.store import Catalogue, Places

from .conftest import StubLlm, mock_settings
from .samples import write_pdf


@pytest.fixture
def parts(tmp_path: Path) -> dict[str, object]:
    settings = mock_settings(tmp_path)
    llm = StubLlm({"ocr_notes": {}})
    catalogue = Catalogue(settings.paths.catalogue)
    return {
        "llm": llm,
        "catalogue": catalogue,
        "notes": NoteIngestor(settings, llm, Places(settings), catalogue),
        "slides": SlideIngestor(llm, catalogue),
        "folder": tmp_path / "notes",
        "decks": tmp_path / "slides",
    }


def ocr_calls(llm: StubLlm) -> list[str]:
    return [image for job, image in llm.jobs if job == "ocr_notes"]


def test_a_notes_file_is_transcribed_once(parts):
    llm, reader, folder = parts["llm"], parts["notes"], parts["folder"]
    write_pdf(folder / "a.pdf", [["first"], ["second"]])

    first = reader.ingest(folder, "cs3340", "week-3")
    assert len(ocr_calls(llm)) == 2
    second = reader.ingest(folder, "cs3340", "week-3")
    assert len(ocr_calls(llm)) == 2, "a stored file went through OCR again"
    assert [p.markdown for p in second] == [p.markdown for p in first]


def test_adding_a_file_transcribes_only_the_new_one(parts):
    llm, reader, folder = parts["llm"], parts["notes"], parts["folder"]
    write_pdf(folder / "a.pdf", [["first"], ["second"]])
    reader.ingest(folder, "cs3340", "week-3")
    llm.jobs.clear()

    write_pdf(folder / "b.pdf", [["third"]])
    pages = reader.ingest(folder, "cs3340", "week-3")
    assert ocr_calls(llm) == ["b-page0001.png"]
    assert [p.page for p in pages] == [1, 2, 3], "pages are numbered across files"


def test_a_changed_file_is_read_again(parts):
    llm, reader, folder = parts["llm"], parts["notes"], parts["folder"]
    write_pdf(folder / "a.pdf", [["first"]])
    reader.ingest(folder, "cs3340", "week-3")
    llm.jobs.clear()

    for image in (folder / "pages").glob("a-page*.png"):
        image.unlink()
    write_pdf(folder / "a.pdf", [["rewritten"], ["and longer"]])
    assert len(reader.ingest(folder, "cs3340", "week-3")) == 2
    assert len(ocr_calls(llm)) == 2


def test_a_correction_is_kept_for_the_next_run(parts):
    catalogue, reader, folder = parts["catalogue"], parts["notes"], parts["folder"]
    write_pdf(folder / "a.pdf", [["first"]])
    page = reader.ingest(folder, "cs3340", "week-3")[0]

    catalogue.save_correction("cs3340", "week-3", page.content_hash, "fixed by hand")
    assert reader.ingest(folder, "cs3340", "week-3")[0].markdown == "fixed by hand"


def test_slides_are_read_once_and_numbered_across_decks(parts, monkeypatch):
    reader, decks = parts["slides"], parts["decks"]
    write_pdf(decks / "a.pdf", [["Strong induction"], ["Base case"]])
    write_pdf(decks / "b.pdf", [["Loop invariants"]])

    first = reader.ingest(decks, "cs3340", "week-3")
    assert [p.page for p in first] == [1, 2, 3]
    assert first[2].markdown.startswith("## Page 3")

    def fail(deck: Path) -> list[str]:
        raise AssertionError(f"{deck.name} was parsed again")

    monkeypatch.setattr(SlideIngestor, "_pdf", staticmethod(fail))
    assert reader.ingest(decks, "cs3340", "week-3") == first


@pytest.fixture
def shelf(tmp_path):
    settings = mock_settings(tmp_path)
    settings.qdrant.backend = "memory"
    library = build(settings)
    library.add_course("cs3340", "Analysis of Algorithms")
    library.add_lecture("cs3340", "Week 3", "induction")
    return library


def test_removing_a_file_deletes_its_pages_and_stored_text(shelf, tmp_path):
    pdf = write_pdf(tmp_path / "a.pdf", [["one"], ["two"]])
    shelf.add_notes("cs3340", "week-3", "a.pdf", pdf.read_bytes())
    folder = shelf.places.notes_dir("cs3340", "week-3")
    NoteIngestor(shelf.settings, StubLlm(), shelf.places, shelf.catalogue).ingest(
        folder, "cs3340", "week-3"
    )
    assert list((folder / "pages").glob("a-page*.png"))

    shelf.remove_source("cs3340", "week-3", "notes", "a.pdf")
    assert not (folder / "a.pdf").exists()
    assert not list((folder / "pages").glob("a-page*.png")), "rendered pages survived"
    assert shelf.catalogue.stored_pages("cs3340", "week-3", "notes", "a.pdf", "any") is None
    assert shelf.catalogue.lecture("cs3340", "week-3").note_count == 0


def test_view_shows_whether_a_file_has_been_read(shelf, tmp_path):
    pdf = write_pdf(tmp_path / "a.pdf", [["one"], ["two"]])
    shelf.add_notes("cs3340", "week-3", "a.pdf", pdf.read_bytes())
    before = shelf.source_details("cs3340", "week-3", "notes", "a.pdf")
    assert not before.stored and before.pages == []

    folder = shelf.places.notes_dir("cs3340", "week-3")
    NoteIngestor(
        shelf.settings,
        StubLlm({"ocr_notes": {"a-page0001.png": "one"}}),
        shelf.places,
        shelf.catalogue,
    ).ingest(folder, "cs3340", "week-3")
    after = shelf.source_details("cs3340", "week-3", "notes", "a.pdf")
    assert after.stored and len(after.pages) == 2 and len(after.images) == 2
    assert after.pages[0].markdown == "one"


def test_upload_names_are_kept_and_cleaned(shelf, tmp_path):
    deck = write_pdf(tmp_path / "d.pdf", [["slide"]]).read_bytes()
    shelf.add_slides("cs3340", "week-3", "Lecture 1.pdf", deck)
    shelf.add_slides("cs3340", "week-3", "../../escape.pdf", deck)
    assert shelf.sources("cs3340", "week-3")["slides"] == ["Lecture 1.pdf", "escape.pdf"]


def test_review_corrections_are_saved_for_the_next_run(shelf, tmp_path, monkeypatch):
    pdf = write_pdf(tmp_path / "a.pdf", [["one"]])
    shelf.add_notes("cs3340", "week-3", "a.pdf", pdf.read_bytes())
    folder = shelf.places.notes_dir("cs3340", "week-3")
    page = NoteIngestor(shelf.settings, StubLlm(), shelf.places, shelf.catalogue).ingest(
        folder, "cs3340", "week-3"
    )[0]
    shelf.catalogue.start_run("r1", "cs3340", "induction", "week-3")
    monkeypatch.setattr(shelf.runner, "approve_notes", lambda *args: None)
    monkeypatch.setattr(shelf.runner, "stream", lambda run_id: iter(()))

    edited = page.model_copy(update={"markdown": "fixed", "edited_by_student": True})
    list(shelf.resume("r1", [edited.model_dump()]))
    view = shelf.source_details("cs3340", "week-3", "notes", "a.pdf")
    assert view.pages[0].markdown == "fixed" and view.pages[0].corrected


def test_deleting_a_course_removes_everything_it_owns(shelf, tmp_path):
    from .test_pipeline import SLIDES, TEXTBOOK

    shelf.index_textbook("cs3340", write_pdf(tmp_path / "book.pdf", TEXTBOOK), "book.pdf")
    shelf.add_slides(
        "cs3340", "week-3", "deck.pdf", write_pdf(tmp_path / "s.pdf", SLIDES).read_bytes()
    )
    shelf.add_notes(
        "cs3340", "week-3", "notes.pdf", write_pdf(tmp_path / "n.pdf", [["a"], ["b"]]).read_bytes()
    )
    run_id, stream = shelf.start("cs3340", shelf.catalogue.lecture("cs3340", "week-3"))
    list(stream)
    list(shelf.resume(run_id, None))
    shelf.finish(run_id)
    shelf.catalogue.log_rejections(
        run_id,
        [Rejection(candidate_id="c", gap_id="g", mechanism="7.3", reason=Reason.BELOW_RELEVANCE)],
    )
    assert shelf.places.run(run_id).exists() and shelf.catalogue.rejections(run_id)
    assert shelf.runner.state(run_id), "the run left no saved state to delete"

    shelf.add_course("math1600", "Linear Algebra")
    shelf.add_lecture("math1600", "Week 1", "")

    shelf.delete_course("cs3340")
    assert [c["id"] for c in shelf.courses()] == ["math1600"]
    assert not shelf.places.course("cs3340").exists(), "uploads survived"
    assert not shelf.places.run(run_id).exists(), "documents survived"
    assert shelf.catalogue.rejections(run_id) == []
    assert shelf.runner.state(run_id) == {}, "the saved run state survived"
    assert not shelf.vectors.exists("cs3340"), "the textbook index survived"
    assert shelf.catalogue.stored_pages("cs3340", "week-3", "notes", "notes.pdf", "x") is None
    assert shelf.catalogue.textbook("cs3340") is None
    assert [lec.id for lec in shelf.lectures("math1600")] == ["week-1"], (
        "another course was touched"
    )


def test_a_lecture_builds_without_a_textbook(shelf, tmp_path):
    """Regression: a course with no textbook failed with 'Collection does not exist'."""
    from .test_pipeline import SLIDES

    shelf.add_slides(
        "cs3340", "week-3", "deck.pdf", write_pdf(tmp_path / "s.pdf", SLIDES).read_bytes()
    )
    shelf.add_notes(
        "cs3340", "week-3", "notes.pdf", write_pdf(tmp_path / "n.pdf", [["a"], ["b"]]).read_bytes()
    )
    run_id, stream = shelf.start("cs3340", shelf.catalogue.lecture("cs3340", "week-3"))
    list(stream)
    list(shelf.resume(run_id, None))
    record = shelf.finish(run_id)
    assert record.status == "done" and shelf.document(record)


def test_slides_stored_by_an_older_reader_are_read_again(parts):
    """Text from a replaced reader is not trusted: each deck is re-read once."""
    from studysynth.models import SourcePage
    from studysynth.pipeline.ingest import content_hash

    catalogue, reader, decks = parts["catalogue"], parts["slides"], parts["decks"]
    deck = write_pdf(decks / "a.pdf", [["Strong induction"]])
    catalogue.store_pages(
        "cs3340",
        "week-3",
        "slides",
        "a.pdf",
        content_hash(deck.read_bytes()),
        [SourcePage(page=1, markdown="scrambled by the old reader")],
    )
    pages = reader.ingest(decks, "cs3340", "week-3")
    assert "Strong induction" in pages[0].markdown
