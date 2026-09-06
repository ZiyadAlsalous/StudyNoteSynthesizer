"""A full run on mocks: real files in, real document out, no API key.

Also covers the two things a checkpointed graph is for: stopping at the review
interrupt, and resuming after a node fails mid-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studysynth.clients import embeddings as embedding_backends
from studysynth.clients import llm as llm_backends
from studysynth.config import Settings
from studysynth.pipeline.graph import EXTRACT_CONCEPTS, Nodes, Runner
from studysynth.pipeline.ingest import TextbookIngestor
from studysynth.models import ChapterRange, Reason
from studysynth.pipeline.retrieval import TextbookGate
from studysynth.services import ServiceError, build
from studysynth.store import Catalogue, Places, VectorStore

from .conftest import mock_settings
from .samples import write_pdf

TEXTBOOK = [
    [
        "# The inductive step",
        "Once the base case is fixed the inductive step carries the entire proof",
        "obligation, so no separate base case is needed for each residue class.",
        "The step must hold for every n above the base, and that is the whole of it.",
    ],
    [
        # Near-verbatim restatement of what the slides already gave the student.
        # This is exactly what the novelty filter (spec 7.5) exists to reject.
        "# Strong induction restated",
        "Strong induction proves a property for every natural number at or above the",
        "base case by assuming the property holds for every value strictly below n,",
        "rather than only for n minus one. It is a convenience and not a stronger axiom:",
        "any proof written with strong induction can be rewritten as ordinary induction",
        "over a modified predicate. The base case is not a formality. A proof with a false",
        "base case establishes nothing at all, however clean the inductive step looks on",
        "the page. Use the strong form when the recursion in the problem reaches back more",
        "than a single step, which is the usual situation for divide and conquer",
        "recurrences. Ordinary induction hands you one previous case; strong induction",
        "hands you all of them.",
    ],
    [
        "# The well-ordering principle",
        "Every non-empty set of natural numbers has a least element. This principle is",
        "logically equivalent to induction and is often taken as the primitive instead,",
        "with induction derived from it as a theorem about minimal counterexamples.",
    ],
    [
        "# Recurrence relations in detail",
        "A recurrence relation expresses a term of a sequence using earlier terms, and",
        "solving one means finding a closed form. The inductive step in a divide and",
        "conquer proof reaches back more than a single previous case.",
    ],
    [
        "# Unrelated graph traversal",
        "Breadth first traversal visits every vertex at distance k before any vertex at",
        "distance k plus one, using a queue to hold the frontier between the two layers.",
    ],
    [
        "# A long historical digression",
        "The modern statement of induction is usually credited to Pascal and Fermat, though",
        "arguments of the same shape appear far earlier in Euclid and in the work of the",
        "Indian mathematician Bhaskara. Formal treatment waited for Dedekind and Peano, who",
        "made induction an axiom of arithmetic rather than a technique of proof. The history",
        "matters less than the practice, but the vocabulary of the field still carries traces",
        "of the older debate about whether induction is discovered or stipulated. Students",
        "encountering the axiom for the first time often ask which it is, and the honest",
        "answer is that for arithmetic as we use it the question has no operational content.",
    ],
]

SLIDES = [
    ["Strong induction", "Assume P(k) for all k < n, then prove P(n).", "The professor sets the exam."],
    ["Strong induction, continued", "Base case must be verified for each residue class."],
    ["Loop invariants", "Holds before, during and after the loop."],
]


@pytest.fixture
def project(tmp_path: Path) -> dict[str, object]:
    settings = mock_settings(tmp_path)

    catalogue = Catalogue(settings.paths.catalogue)
    places = Places(settings)
    vectors = VectorStore(settings)
    llm = llm_backends.build(settings)
    embed = embedding_backends.build(settings)

    catalogue.add_course("cs3340", "Analysis of Algorithms")
    span = ChapterRange(
        chapter="induction", title="Induction", page_start=1, page_end=6, manual_override=True
    )
    catalogue.set_chapters("cs3340", [span])

    textbook = write_pdf(tmp_path / "textbook.pdf", TEXTBOOK)
    parents, chunks = TextbookIngestor(settings).ingest(textbook, "cs3340", [span])
    catalogue.put_parents(parents)
    vectors.create("cs3340", embed.dimensions)
    vectors.upsert("cs3340", chunks, embed.embed_documents([c.text for c in chunks]))

    slides_dir = tmp_path / "slides"
    write_pdf(slides_dir / "lecture01.pdf", SLIDES)
    notes_dir = tmp_path / "notes"
    write_pdf(notes_dir / "notes.pdf", [
        ["Inductive hypothesis = what you may assume", "Inductive step = what you still owe"],
        ["Loop invariants", "not sure I follow this one - ask in office hours"],
    ])

    gate = TextbookGate(settings, llm, embed, vectors, catalogue)
    nodes = Nodes(settings, llm, embed, gate, catalogue, places)
    return {
        "settings": settings,
        "catalogue": catalogue,
        "runner": Runner(settings, nodes),
        "slides_dir": slides_dir,
        "notes_dir": notes_dir,
        "parents": parents,
    }


def test_textbook_chunks_into_the_expected_sections(project):
    headings = [p.heading for p in project["parents"]]
    assert "The inductive step" in headings
    assert "The well-ordering principle" in headings
    assert all(p.chapter == "induction" for p in project["parents"])


def test_full_run_on_mocks_produces_a_document(project):
    runner, catalogue = project["runner"], project["catalogue"]
    catalogue.start_run("run-1", "cs3340", "induction")

    visited = [node for node, _ in runner.stream(
        "run-1", course="cs3340", chapter="induction",
        slides_dir=str(project["slides_dir"]), notes_dir=str(project["notes_dir"]),
    )]

    # The graph stops for review before it looks at any concept.
    assert runner.pending("run-1") == (EXTRACT_CONCEPTS,)
    assert "extract_concepts" not in visited
    state = runner.state("run-1")
    assert [n.page for n in state["notes"]] == [1, 2]
    assert "Inductive hypothesis" in state["notes"][0].markdown

    runner.approve_notes("run-1")
    rest = [node for node, _ in runner.stream("run-1")]
    assert rest[-1] == "verify"

    final = runner.state("run-1")
    assert final["document"].startswith("# Induction")
    assert "## Topics I have no notes on" in final["document"]
    assert final["unverified"] == []
    catalogue.finish_run("run-1", "done")


def test_every_anti_bloat_mechanism_fires_in_a_real_run(project):
    runner, catalogue = project["runner"], project["catalogue"]
    catalogue.start_run("run-2", "cs3340", "induction")
    list(runner.stream(
        "run-2", course="cs3340", chapter="induction",
        slides_dir=str(project["slides_dir"]), notes_dir=str(project["notes_dir"]),
    ))
    runner.approve_notes("run-2")
    list(runner.stream("run-2"))

    reasons = {r.reason for r in catalogue.rejections("run-2")}
    assert Reason.BELOW_RELEVANCE in reasons, "7.3 never fired"
    assert Reason.NOT_NECESSARY in reasons, "7.4 never fired"
    assert Reason.DUPLICATE_OF in reasons, "7.5 never fired"
    assert Reason.INTRODUCES_CONCEPT in reasons, "7.6 never fired"
    assert Reason.OVER_BUDGET in reasons, "7.7 never fired"

    outcome = runner.state("run-2")["retrieval"]
    assert outcome.tokens_admitted <= outcome.budget
    assert outcome.admitted, "the gate rejected everything, which is also a bug"


def test_the_rejection_log_names_the_mechanism_and_the_score(project):
    runner, catalogue = project["runner"], project["catalogue"]
    catalogue.start_run("run-3", "cs3340", "induction")
    list(runner.stream(
        "run-3", course="cs3340", chapter="induction",
        slides_dir=str(project["slides_dir"]), notes_dir=str(project["notes_dir"]),
    ))
    runner.approve_notes("run-3")
    list(runner.stream("run-3"))

    guard = [r for r in catalogue.rejections("run-3") if r.reason is Reason.INTRODUCES_CONCEPT]
    assert guard and "well-ordering principle" in guard[0].detail
    assert guard[0].mechanism.startswith("7.6")


def test_a_run_resumes_from_the_node_that_failed(project, monkeypatch):
    """Kill the run inside synthesis, then restart it. Nothing before the failed
    node runs a second time."""
    runner, catalogue = project["runner"], project["catalogue"]
    catalogue.start_run("run-4", "cs3340", "induction")
    list(runner.stream(
        "run-4", course="cs3340", chapter="induction",
        slides_dir=str(project["slides_dir"]), notes_dir=str(project["notes_dir"]),
    ))
    runner.approve_notes("run-4")

    nodes = runner._nodes
    original = nodes.synthesize
    calls = {"n": 0}

    def flaky(state):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash inside synthesis")
        return original(state)

    monkeypatch.setattr(nodes, "synthesize", flaky)
    with pytest.raises(RuntimeError, match="simulated crash"):
        list(runner.stream("run-4"))

    # Retrieval already ran and is checkpointed; the gate is not re-graded.
    mid = runner.state("run-4")
    assert mid["retrieval"].admitted
    assert "document" not in mid

    list(runner.stream("run-4"))
    assert runner.state("run-4")["document"].startswith("# Induction")
    assert calls["n"] == 2


# --- persistence and replacement -------------------------------------------


@pytest.fixture
def shelf(tmp_path):
    settings = mock_settings(tmp_path)
    settings.qdrant.backend = "memory"
    library = build(settings)
    library.add_course("cs3340", "Analysis of Algorithms")
    return library


def test_a_textbook_is_indexed_once_and_then_reused(shelf, tmp_path):
    """The whole reason the index is persisted: embedding a book is slow, and
    it must not happen again on every run."""
    assert shelf.textbook_status("cs3340") is None

    pdf = write_pdf(tmp_path / "book.pdf", TEXTBOOK)
    chapters, chunks = shelf.index_textbook("cs3340", pdf, "book.pdf")
    assert chapters >= 1 and chunks > 0

    status = shelf.textbook_status("cs3340")
    assert status is not None
    assert status["filename"] == "book.pdf"
    assert status["chunks"] == chunks
    # Asking again reports the stored index rather than rebuilding it.
    assert shelf.textbook_status("cs3340") == status


def test_replacing_notes_removes_the_previous_pdf_and_its_pages(shelf, tmp_path):
    """A student re-exports their notes often. A run must never mix two
    versions of the same page."""
    from studysynth.pipeline.ingest import NoteIngestor

    shelf.add_lecture("cs3340", "Week 3", "induction")
    long_notes = write_pdf(tmp_path / "a.pdf", [[f"page {n}"] for n in range(1, 13)])
    assert shelf.replace_notes("cs3340", "week-3", "a.pdf", long_notes.read_bytes()) == 12

    folder = shelf.places.notes_dir("cs3340", "week-3")
    reader = NoteIngestor(shelf.settings, shelf.llm, shelf.places)
    assert len(reader._images(folder)) == 12

    short_notes = write_pdf(tmp_path / "b.pdf", [["only page"]])
    assert shelf.replace_notes("cs3340", "week-3", "b.pdf", short_notes.read_bytes()) == 1
    assert len(list(folder.glob("*.pdf"))) == 1, "the previous PDF survived"
    assert len(reader._images(folder)) == 1, "stale rendered pages survived"
    assert shelf.catalogue.lecture("cs3340", "week-3").note_count == 1


@pytest.mark.parametrize("pages", [1, 7, 30, 64])
def test_a_notes_pdf_of_any_length_is_accepted(shelf, tmp_path, pages):
    """One page or thirty, depending on the topic. There is no cap."""
    from studysynth.pipeline.ingest import NoteIngestor

    shelf.add_lecture("cs3340", f"Week {pages}", "induction")
    lecture_id = f"week-{pages}"
    pdf = write_pdf(tmp_path / f"n{pages}.pdf", [[f"page {n}"] for n in range(1, pages + 1)])
    assert shelf.replace_notes("cs3340", lecture_id, "notes.pdf", pdf.read_bytes()) == pages

    folder = shelf.places.notes_dir("cs3340", lecture_id)
    images = NoteIngestor(shelf.settings, shelf.llm, shelf.places)._images(folder)
    assert len(images) == pages, "a page was dropped"
    assert all(p.suffix == ".png" and p.stat().st_size > 0 for p in images)


def test_notes_reject_anything_that_is_not_a_pdf(shelf):
    shelf.add_lecture("cs3340", "Week 3", "induction")
    with pytest.raises(ServiceError, match="must be a PDF"):
        shelf.replace_notes("cs3340", "week-3", "notes.docx", b"PK")


def test_an_unreadable_pdf_gives_a_clear_error(shelf):
    """A raw PyMuPDF stack trace tells a student nothing."""
    shelf.add_lecture("cs3340", "Week 3", "induction")
    with pytest.raises(ServiceError, match="could not be read as a PDF"):
        shelf.replace_notes("cs3340", "week-3", "notes.pdf", b"not really a pdf")


def test_rendered_pages_are_reused_on_a_second_read(shelf, tmp_path):
    """Re-rasterising an unchanged export is wasted work on every rerun."""
    from studysynth.pipeline.ingest import NoteIngestor

    shelf.add_lecture("cs3340", "Week 3", "induction")
    pdf = write_pdf(tmp_path / "notes.pdf", [["one"], ["two"], ["three"]])
    shelf.replace_notes("cs3340", "week-3", "notes.pdf", pdf.read_bytes())
    folder = shelf.places.notes_dir("cs3340", "week-3")
    reader = NoteIngestor(shelf.settings, shelf.llm, shelf.places)

    first = reader._images(folder)
    stamps = [p.stat().st_mtime_ns for p in first]
    second = reader._images(folder)
    assert [p.stat().st_mtime_ns for p in second] == stamps, "pages were re-rendered"


def test_a_lecture_is_not_runnable_until_both_sources_exist(shelf, tmp_path):
    shelf.add_lecture("cs3340", "Week 3", "induction")
    lecture = shelf.catalogue.lecture("cs3340", "week-3")
    assert not lecture.ready
    with pytest.raises(ServiceError, match="Upload both"):
        shelf.start("cs3340", lecture)

    shelf.replace_slides("cs3340", "week-3", "deck.pdf",
                         write_pdf(tmp_path / "d.pdf", SLIDES).read_bytes())
    shelf.replace_notes("cs3340", "week-3", "notes.pdf",
                        write_pdf(tmp_path / "n.pdf", [["a page"]]).read_bytes())
    assert shelf.catalogue.lecture("cs3340", "week-3").ready


def test_run_history_is_kept_per_lecture(shelf):
    shelf.add_lecture("cs3340", "Week 3", "induction")
    assert shelf.history("cs3340", "week-3") == []
    shelf.catalogue.start_run("r1", "cs3340", "induction", "week-3")
    shelf.catalogue.finish_run("r1", "done", document_path="/tmp/one.md")
    shelf.catalogue.start_run("r2", "cs3340", "induction", "week-3")
    shelf.catalogue.finish_run("r2", "done", document_path="/tmp/two.md")

    history = shelf.history("cs3340", "week-3")
    assert len(history) == 2, "every run is kept, not just the latest"
    assert {r.id for r in history} == {"r1", "r2"}


def test_deleting_a_lecture_keeps_its_finished_documents(shelf, tmp_path):
    """Sources go, history stays: the documents are the point."""
    shelf.add_lecture("cs3340", "Week 3", "induction")
    shelf.replace_notes("cs3340", "week-3", "notes.pdf",
                        write_pdf(tmp_path / "n.pdf", [["a page"]]).read_bytes())
    shelf.catalogue.start_run("r1", "cs3340", "induction", "week-3")
    shelf.catalogue.finish_run("r1", "done", document_path="/tmp/one.md")

    shelf.delete_lecture("cs3340", "week-3")
    assert not shelf.places.lecture("cs3340", "week-3").exists()
    assert shelf.catalogue.run("r1").document_path == "/tmp/one.md"


def test_notes_survive_the_review_interrupt_as_objects(project):
    """The edited transcript goes back into the graph as plain dicts, and every
    reader after the interrupt expects NotePage. Regression for a crash that hit
    the review screen and both downstream nodes."""
    runner, catalogue = project["runner"], project["catalogue"]
    catalogue.start_run("run-notes", "cs3340", "induction")
    list(runner.stream(
        "run-notes", course="cs3340", chapter="induction",
        slides_dir=str(project["slides_dir"]), notes_dir=str(project["notes_dir"]),
    ))

    edited = [p.model_copy(update={"markdown": "corrected"}).model_dump()
              for p in runner.state("run-notes")["notes"]]
    runner.approve_notes("run-notes", edited)

    pages = runner.state("run-notes")["notes"]
    assert all(hasattr(p, "page") for p in pages), "notes came back as dicts"
    assert pages[0].markdown == "corrected"

    list(runner.stream("run-notes"))
    assert runner.state("run-notes")["document"]
