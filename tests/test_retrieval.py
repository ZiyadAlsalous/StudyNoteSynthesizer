"""One test per anti-bloat mechanism, each exercised alone.

Spec section 7 says a run where nothing is ever rejected is a bug report. These
tests are what stops that from happening quietly.
"""

from __future__ import annotations

import pytest

from studysynth.clients.embeddings import MockEmbeddings
from studysynth.config import Settings
from studysynth.models import Reason
from studysynth.pipeline.retrieval import TextbookGate

from .conftest import StubLlm, make_candidate, make_concept, make_draft, make_gap


def gate(settings: Settings, llm: StubLlm, embeddings: MockEmbeddings) -> TextbookGate:
    return TextbookGate(settings, llm, embeddings, vectors=None, catalogue=None)


def test_no_gaps_means_the_textbook_is_never_read(settings, embeddings):
    llm = StubLlm()
    assert gate(settings, llm, embeddings).retrieve("cs3340", "induction", []) == []
    assert llm.jobs == []


def test_scope_rejects_a_passage_from_another_chapter(settings, embeddings):
    candidates = [make_candidate("a", chapter="induction"), make_candidate("b", chapter="graphs")]
    kept, rejected = gate(settings, StubLlm(), embeddings).check_scope(candidates, ["induction"])
    assert [c.id for c in kept] == ["a"]
    assert rejected[0].reason is Reason.OUT_OF_CHAPTER
    assert rejected[0].mechanism.startswith("7.2")


def test_scope_is_a_single_chapter_unless_adjacency_is_enabled(settings, embeddings):
    assert gate(settings, StubLlm(), embeddings)._scope("cs3340", "induction") == ["induction"]


class StubVectors:
    """Returns chapter hits for the auto-scoping probe."""

    def __init__(self, hits):
        self.hits = hits
        self.filters: list[list[str]] = []

    def search(self, course, vector, chapters, limit):
        self.filters.append(list(chapters))
        return self.hits


def auto_gate(settings, vectors, embeddings):
    return TextbookGate(settings, StubLlm(), embeddings, vectors, catalogue=None)


def test_chapters_are_detected_from_the_gaps_when_none_is_pinned(settings, embeddings):
    """A student should not have to know their induction lecture is chapter 4."""
    vectors = StubVectors([
        (0.91, {"chapter": "4-divide-and-conquer"}),
        (0.88, {"chapter": "4-divide-and-conquer"}),
        (0.42, {"chapter": "22-elementary-graph-algorithms"}),
    ])
    found = auto_gate(settings, vectors, embeddings).detect_chapters("cs3340", [make_gap()])
    assert found[0] == "4-divide-and-conquer", "the strongest chapter should win"
    assert vectors.filters == [[]], "the probe must search the whole book, unfiltered"


def test_detection_returns_at_most_the_configured_number_of_chapters(settings, embeddings):
    settings.retrieval.auto_scope_chapters = 2
    vectors = StubVectors([(0.9, {"chapter": f"ch{n}"}) for n in range(6)])
    found = auto_gate(settings, vectors, embeddings).detect_chapters("cs3340", [make_gap()])
    assert len(found) == 2


def test_detection_does_nothing_without_a_gap(settings, embeddings):
    """7.1 still holds: no gap, no textbook query, not even a probe."""
    vectors = StubVectors([(0.9, {"chapter": "anything"})])
    assert auto_gate(settings, vectors, embeddings).detect_chapters("cs3340", []) == []
    assert vectors.filters == []


def test_relevance_drops_a_topical_near_miss(settings, embeddings):
    llm = StubLlm({"grade_relevance": {
        "a": {"score": 0.91, "reason": "answers it"},
        "b": {"score": 0.40, "reason": "same words, different question"},
    }})
    kept, rejected = gate(settings, llm, embeddings).grade_relevance(
        [make_candidate("a"), make_candidate("b")], [make_gap()]
    )
    assert [c.id for c in kept] == ["a"]
    assert kept[0].relevance == pytest.approx(0.91)
    assert rejected[0].reason is Reason.BELOW_RELEVANCE
    assert rejected[0].score == pytest.approx(0.40)


def test_relevance_threshold_is_inclusive_at_the_configured_value(settings, embeddings):
    llm = StubLlm({"grade_relevance": {"a": {"score": settings.retrieval.min_relevance}}})
    kept, rejected = gate(settings, llm, embeddings).grade_relevance(
        [make_candidate("a")], [make_gap()]
    )
    assert len(kept) == 1 and not rejected


def test_necessity_rejects_relevant_prose_the_student_already_has(settings, embeddings):
    llm = StubLlm({"grade_necessity": {
        "a": {"score": 0.2, "reason": "the drafted slide section already resolves this"},
    }})
    kept, rejected = gate(settings, llm, embeddings).grade_necessity(
        [make_candidate("a", relevance=0.95)], [make_gap()], [make_draft()]
    )
    assert kept == []
    assert rejected[0].reason is Reason.NOT_NECESSARY
    assert rejected[0].mechanism.startswith("7.4")


def test_necessity_is_graded_against_the_draft_not_the_gap_alone(settings, embeddings):
    llm = StubLlm({"grade_necessity": {"a": {"score": 0.9}}})
    draft = make_draft("A long drafted section about the inductive step.")
    gate(settings, llm, embeddings).grade_necessity([make_candidate("a")], [make_gap()], [draft])
    assert ("grade_necessity", "a") in llm.jobs


def test_necessity_bar_cannot_be_looser_than_relevance():
    from studysynth.config import ConfigError

    loose = Settings(_env_file=None)
    loose.retrieval.min_necessity = 0.1
    loose.retrieval.min_relevance = 0.5
    with pytest.raises(ConfigError):
        loose.validated()


def test_novelty_drops_a_restatement_of_drafted_content(settings, embeddings):
    body = "Strong induction assumes the property for every value below n."
    kept, rejected = gate(settings, StubLlm(), embeddings).filter_novel(
        [make_candidate("a", text=body)], [make_draft(body)]
    )
    assert kept == []
    assert rejected[0].reason is Reason.DUPLICATE_OF
    assert rejected[0].score > settings.retrieval.max_similarity_to_draft


def test_novelty_keeps_genuinely_new_material(settings, embeddings):
    kept, rejected = gate(settings, StubLlm(), embeddings).filter_novel(
        [make_candidate("a", text="Amortized analysis charges each operation a fixed credit.")],
        [make_draft("Strong induction assumes every smaller case holds.")],
    )
    assert [c.id for c in kept] == ["a"]
    assert not rejected


def test_novelty_is_a_no_op_before_anything_is_drafted(settings, embeddings):
    kept, rejected = gate(settings, StubLlm(), embeddings).filter_novel([make_candidate("a")], [])
    assert len(kept) == 1 and not rejected


def test_guard_rejects_a_passage_that_introduces_an_off_syllabus_term(settings, embeddings):
    llm = StubLlm({"new_concept_guard": {"a": {"terms": ["well-ordering principle"]}}})
    kept, rejected = gate(settings, llm, embeddings).guard_new_concepts(
        [make_candidate("a")], [make_concept()]
    )
    assert kept == []
    assert rejected[0].reason is Reason.INTRODUCES_CONCEPT
    assert "well-ordering principle" in rejected[0].detail


def test_guard_admits_a_passage_that_stays_within_slide_scope(settings, embeddings):
    llm = StubLlm({"new_concept_guard": {"a": {"terms": []}}})
    kept, rejected = gate(settings, llm, embeddings).guard_new_concepts(
        [make_candidate("a")], [make_concept()]
    )
    assert len(kept) == 1 and not rejected


def test_guard_can_be_disabled_only_from_config(settings, embeddings):
    settings.retrieval.new_concept_guard = False
    llm = StubLlm({"new_concept_guard": {"a": {"terms": ["anything"]}}})
    kept, rejected = gate(settings, llm, embeddings).guard_new_concepts(
        [make_candidate("a")], [make_concept()]
    )
    assert len(kept) == 1 and not rejected and llm.jobs == []


def test_budget_is_the_tighter_of_the_absolute_cap_and_the_fraction(settings, embeddings):
    engine = gate(settings, StubLlm(), embeddings)
    assert engine.budget_for(1000) == int(0.15 * 1000 / 0.85)
    assert engine.budget_for(100_000) == settings.retrieval.textbook_token_budget


def test_budget_admits_highest_necessity_first_and_rejects_the_rest(settings, embeddings):
    settings.retrieval.textbook_token_budget = 250
    candidates = [
        make_candidate("low", tokens=200, necessity=0.65),
        make_candidate("high", tokens=200, necessity=0.95),
    ]
    admitted, rejected, spent, budget = gate(settings, StubLlm(), embeddings).enforce_budget(
        candidates, document_tokens=100_000
    )
    assert [a.candidate_id for a in admitted] == ["high"]
    assert rejected[0].candidate_id == "low"
    assert rejected[0].reason is Reason.OVER_BUDGET
    assert spent == 200 and budget == 250


def test_budget_rejection_records_the_budget_state(settings, embeddings):
    settings.retrieval.textbook_token_budget = 50
    admitted, rejected, _, _ = gate(settings, StubLlm(), embeddings).enforce_budget(
        [make_candidate("a", tokens=200, necessity=0.9)], document_tokens=100_000
    )
    assert not admitted
    assert "0/50 spent" in rejected[0].detail


def test_admitted_passages_carry_a_provenance_citation(settings, embeddings):
    settings.retrieval.textbook_token_budget = 500
    admitted, _, _, _ = gate(settings, StubLlm(), embeddings).enforce_budget(
        [make_candidate("a", tokens=100, necessity=0.9)], document_tokens=100_000
    )
    assert admitted[0].citation == "[C: pages 10-14]"


def test_the_vector_floor_is_off_by_default(settings):
    """It steals work from 7.3 and 7.4, so it must be switched on deliberately."""
    assert settings.retrieval.min_vector_score == 0.0


def test_the_vector_floor_logs_what_it_drops(settings, embeddings):
    """Spec 7.8: nothing disappears without a reason and a score, however cheap
    the mechanism that removed it."""
    settings.retrieval.min_vector_score = 0.5
    llm = StubLlm({"grade_relevance": {"keep": {"score": 0.9}}})
    weak = make_candidate("weak")
    weak = weak.model_copy(update={"retrieval_score": 0.2})
    strong = make_candidate("keep")

    kept, rejected = gate(settings, llm, embeddings).grade_relevance([weak, strong], [make_gap()])
    assert [c.id for c in kept] == ["keep"]
    assert rejected[0].candidate_id == "weak"
    assert rejected[0].reason is Reason.BELOW_RELEVANCE
    assert rejected[0].score == pytest.approx(0.2)
    assert "vector floor" in rejected[0].mechanism
    assert ("grade_relevance", "weak") not in llm.jobs, "the floor should run before the model"
