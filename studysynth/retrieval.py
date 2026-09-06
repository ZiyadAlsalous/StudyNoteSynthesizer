"""Spec section 7. The seven mechanisms that stand between textbook and output.

Each mechanism is a separate method that takes candidates and returns the
survivors plus a rejection for everything it dropped. Nothing here fails
silently: a candidate either survives with a recorded score or appears in the
rejection log with the mechanism that killed it and why.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from .config import Settings
from .embeddings import EmbeddingBackend, cosine
from .ingest import estimate_tokens
from .llm import LlmClient, PromptLibrary
from .models import (
    Admitted,
    Candidate,
    Concept,
    DraftedConcept,
    Gap,
    Reason,
    Rejection,
    RetrievalOutcome,
)
from .store import Catalogue, VectorStore


class RetrievalError(RuntimeError):
    pass


class RelevanceVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class NecessityVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class NewConceptVerdict(BaseModel):
    terms: list[str] = Field(default_factory=list)


Survivors = tuple[list[Candidate], list[Rejection]]


class TextbookGate:
    """The only path by which textbook content reaches the document."""

    def __init__(
        self,
        settings: Settings,
        llm: LlmClient,
        embeddings: EmbeddingBackend,
        vectors: VectorStore,
        catalogue: Catalogue,
    ) -> None:
        self._settings = settings
        self._config = settings.retrieval
        self._llm = llm
        self._embeddings = embeddings
        self._vectors = vectors
        self._catalogue = catalogue
        self._prompts = PromptLibrary(settings.prompts_dir)

    # 7.1 gap-triggered querying ------------------------------------------------

    def retrieve(self, course: str, chapter: str, gaps: Sequence[Gap]) -> list[Candidate]:
        """The textbook is queried once per gap and never otherwise.

        There is no entry point here that takes free text. If the concept and
        gap extraction found nothing unresolved, the textbook is not read at all.
        """
        if not gaps:
            return []
        chapters = self._scope(course, chapter)
        candidates: list[Candidate] = []
        for gap in gaps:
            vector = self._embeddings.embed_query(gap.query)
            hits = self._vectors.search(course, vector.tolist(), chapters, self._config.top_k)
            for rank, (score, payload) in enumerate(hits):
                parent = self._catalogue.parent(str(payload["parent_id"]))
                candidates.append(
                    Candidate(
                        id=f"{gap.id}:{rank}",
                        gap_id=gap.id,
                        parent_id=parent.id,
                        text=parent.text,
                        section_path=parent.section_path,
                        page_start=parent.page_start,
                        page_end=parent.page_end,
                        chapter=parent.chapter,
                        retrieval_score=float(score),
                        token_estimate=parent.token_estimate or estimate_tokens(parent.text),
                    )
                )
        return self._dedupe(candidates)

    @staticmethod
    def _dedupe(candidates: Sequence[Candidate]) -> list[Candidate]:
        """Two gaps often reach the same section. Grade it once."""
        best: dict[str, Candidate] = {}
        for candidate in candidates:
            existing = best.get(candidate.parent_id)
            if existing is None or candidate.retrieval_score > existing.retrieval_score:
                best[candidate.parent_id] = candidate
        return sorted(best.values(), key=lambda c: -c.retrieval_score)

    # 7.2 chapter scoping -------------------------------------------------------

    def _scope(self, course: str, chapter: str) -> list[str]:
        """Which chapters the filter admits. Passed to Qdrant, not applied after."""
        if not self._config.allow_adjacent_chapters:
            return [chapter]
        ordered = [span.chapter for span in self._catalogue.chapters(course)]
        if chapter not in ordered:
            return [chapter]
        index = ordered.index(chapter)
        window = ordered[max(0, index - 1) : index + 2]
        return window

    def check_scope(self, candidates: Sequence[Candidate], allowed: Sequence[str]) -> Survivors:
        """Belt and braces: the index filter should make this a no-op. If it
        ever rejects something, the payload index is missing or misdeclared."""
        kept, rejected = [], []
        for candidate in candidates:
            if candidate.chapter in allowed:
                kept.append(candidate)
            else:
                rejected.append(
                    Rejection(
                        candidate_id=candidate.id,
                        gap_id=candidate.gap_id,
                        mechanism="7.2 chapter scoping",
                        reason=Reason.OUT_OF_CHAPTER,
                        detail=f"chapter {candidate.chapter} not in {list(allowed)}",
                    )
                )
        return kept, rejected

    # 7.3 relevance grading -----------------------------------------------------

    def grade_relevance(self, candidates: Sequence[Candidate], gaps: Sequence[Gap]) -> Survivors:
        questions = {gap.id: gap.question for gap in gaps}
        kept, rejected = [], []
        for candidate in candidates:
            prompt = self._prompts.render(
                "grade_relevance",
                {
                    "question": questions.get(candidate.gap_id, ""),
                    "passage": candidate.text,
                    "page_start": candidate.page_start,
                    "page_end": candidate.page_end,
                },
            )
            verdict = self._llm.structured(
                prompt, RelevanceVerdict, job="grade_relevance", key=candidate.parent_id
            )
            graded = candidate.model_copy(update={"relevance": verdict.score})
            if verdict.score >= self._config.min_relevance:
                kept.append(graded)
            else:
                rejected.append(
                    Rejection(
                        candidate_id=candidate.id,
                        gap_id=candidate.gap_id,
                        mechanism="7.3 relevance grading",
                        reason=Reason.BELOW_RELEVANCE,
                        score=verdict.score,
                        detail=verdict.reason,
                    )
                )
        return kept, rejected

    # 7.4 necessity grading -----------------------------------------------------

    def grade_necessity(
        self,
        candidates: Sequence[Candidate],
        gaps: Sequence[Gap],
        drafts: Sequence[DraftedConcept],
    ) -> Survivors:
        """A different question from relevance: does the student still need it?

        Graded against what has already been drafted from slides and notes, not
        against the gap alone. This is where correct, on-topic, unnecessary prose
        dies.
        """
        questions = {gap.id: gap.question for gap in gaps}
        draft_text = "\n\n".join(f"### {d.heading}\n{d.body}" for d in drafts)
        kept, rejected = [], []
        for candidate in candidates:
            prompt = self._prompts.render(
                "grade_necessity",
                {
                    "question": questions.get(candidate.gap_id, ""),
                    "passage": candidate.text,
                    "draft": draft_text,
                },
            )
            verdict = self._llm.structured(
                prompt, NecessityVerdict, job="grade_necessity", key=candidate.parent_id
            )
            graded = candidate.model_copy(update={"necessity": verdict.score})
            if verdict.score >= self._config.min_necessity:
                kept.append(graded)
            else:
                rejected.append(
                    Rejection(
                        candidate_id=candidate.id,
                        gap_id=candidate.gap_id,
                        mechanism="7.4 necessity grading",
                        reason=Reason.NOT_NECESSARY,
                        score=verdict.score,
                        detail=verdict.reason,
                    )
                )
        return kept, rejected

    # 7.5 novelty filter --------------------------------------------------------

    def filter_novel(
        self, candidates: Sequence[Candidate], drafts: Sequence[DraftedConcept]
    ) -> Survivors:
        """Drop the textbook saying in 200 words what the slide said in 20."""
        if not drafts:
            return list(candidates), []
        labels = [d.concept_id for d in drafts]
        matrix = self._embeddings.embed_documents([f"{d.heading}\n{d.body}" for d in drafts])
        kept, rejected = [], []
        for candidate in candidates:
            vector = self._embeddings.embed_documents([candidate.text])[0]
            scores = matrix @ vector
            best = int(np.argmax(scores))
            similarity = float(scores[best])
            graded = candidate.model_copy(update={"draft_similarity": similarity})
            if similarity <= self._config.max_similarity_to_draft:
                kept.append(graded)
            else:
                rejected.append(
                    Rejection(
                        candidate_id=candidate.id,
                        gap_id=candidate.gap_id,
                        mechanism="7.5 novelty filter",
                        reason=Reason.DUPLICATE_OF,
                        score=similarity,
                        detail=labels[best],
                    )
                )
        return kept, rejected

    # 7.6 new-concept guard -----------------------------------------------------

    def guard_new_concepts(
        self, candidates: Sequence[Candidate], concepts: Sequence[Concept]
    ) -> Survivors:
        """The slides define the examinable surface. The textbook may explain
        what is on them; it may never add to them."""
        if not self._config.new_concept_guard:
            return list(candidates), []
        names = ", ".join(concept.name for concept in concepts)
        kept, rejected = [], []
        for candidate in candidates:
            prompt = self._prompts.render(
                "new_concept_guard", {"concept_names": names, "passage": candidate.text}
            )
            verdict = self._llm.structured(
                prompt, NewConceptVerdict, job="new_concept_guard", key=candidate.parent_id
            )
            if not verdict.terms:
                kept.append(candidate)
            else:
                rejected.append(
                    Rejection(
                        candidate_id=candidate.id,
                        gap_id=candidate.gap_id,
                        mechanism="7.6 new-concept guard",
                        reason=Reason.INTRODUCES_CONCEPT,
                        detail=", ".join(verdict.terms),
                    )
                )
        return kept, rejected

    # 7.7 budget enforcement ----------------------------------------------------

    def budget_for(self, document_tokens: int) -> int:
        """The binding constraint, whichever of the two ceilings is tighter.

        The fraction is of the finished document, so a chapter with a thin draft
        gets a small budget however generous the absolute cap is.
        """
        fraction = self._config.max_textbook_fraction
        if fraction >= 1.0:
            proportional = self._config.textbook_token_budget
        else:
            proportional = int(fraction * document_tokens / (1.0 - fraction))
        return max(0, min(self._config.textbook_token_budget, proportional))

    def enforce_budget(self, candidates: Sequence[Candidate], document_tokens: int) -> tuple[
        list[Admitted], list[Rejection], int, int
    ]:
        budget = self.budget_for(document_tokens)
        ordered = sorted(candidates, key=lambda c: -(c.necessity or 0.0))
        admitted: list[Admitted] = []
        rejected: list[Rejection] = []
        spent = 0
        for candidate in ordered:
            if spent + candidate.token_estimate > budget:
                rejected.append(
                    Rejection(
                        candidate_id=candidate.id,
                        gap_id=candidate.gap_id,
                        mechanism="7.7 budget enforcement",
                        reason=Reason.OVER_BUDGET,
                        score=candidate.necessity,
                        detail=f"{spent}/{budget} spent, passage needs {candidate.token_estimate}",
                    )
                )
                continue
            spent += candidate.token_estimate
            admitted.append(
                Admitted(
                    candidate_id=candidate.id,
                    gap_id=candidate.gap_id,
                    text=candidate.text,
                    citation=f"[C: pages {candidate.page_start}-{candidate.page_end}]",
                    page_start=candidate.page_start,
                    page_end=candidate.page_end,
                    token_estimate=candidate.token_estimate,
                    necessity=candidate.necessity or 0.0,
                )
            )
        return admitted, rejected, spent, budget

    # the whole gate ------------------------------------------------------------

    def run(
        self,
        course: str,
        chapter: str,
        gaps: Sequence[Gap],
        concepts: Sequence[Concept],
        drafts: Sequence[DraftedConcept],
    ) -> RetrievalOutcome:
        rejections: list[Rejection] = []
        candidates = self.retrieve(course, chapter, gaps)

        kept, dropped = self.check_scope(candidates, self._scope(course, chapter))
        rejections += dropped
        kept, dropped = self.grade_relevance(kept, gaps)
        rejections += dropped
        kept, dropped = self.grade_necessity(kept, gaps, drafts)
        rejections += dropped
        kept, dropped = self.filter_novel(kept, drafts)
        rejections += dropped
        kept, dropped = self.guard_new_concepts(kept, concepts)
        rejections += dropped

        document_tokens = sum(estimate_tokens(d.body) for d in drafts)
        admitted, dropped, spent, budget = self.enforce_budget(kept, document_tokens)
        rejections += dropped

        return RetrievalOutcome(
            admitted=admitted, rejections=rejections, tokens_admitted=spent, budget=budget
        )
