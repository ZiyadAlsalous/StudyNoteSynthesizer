"""A full run on mocks: real files in, real document out, no API key.

Also covers the two things a checkpointed graph is for: stopping at the review
interrupt, and resuming after a node fails mid-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studysynth import embeddings as embedding_backends
from studysynth import llm as llm_backends
from studysynth.config import Settings
from studysynth.graph import EXTRACT_CONCEPTS, Nodes, Runner
from studysynth.ingest import TextbookIngestor
from studysynth.models import ChapterRange, Reason
from studysynth.retrieval import TextbookGate
from studysynth.store import Catalogue, Places, VectorStore

from .conftest import mock_settings
from .samples import write_pdf, write_png

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
    write_png(notes_dir / "note1.png", 1)
    write_png(notes_dir / "note2.png", 2)

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
