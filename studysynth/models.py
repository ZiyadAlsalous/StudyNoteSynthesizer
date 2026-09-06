"""Domain types. No logic lives here."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class Source(StrEnum):
    """Spec section 2: the authority ranking, in order."""

    SLIDES = "A"
    NOTES = "B"
    TEXTBOOK = "C"


class ChunkType(StrEnum):
    PROSE = "prose"
    SLIDE = "slide"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"


class GapKind(StrEnum):
    UNDEFINED = "undefined"
    UNSUPPORTED = "unsupported"
    STUDENT_FLAGGED = "student_flagged"


class Reason(StrEnum):
    """Why a textbook candidate was rejected."""

    OUT_OF_CHAPTER = "out_of_chapter"
    BELOW_RELEVANCE = "below_relevance"
    NOT_NECESSARY = "not_necessary"
    DUPLICATE_OF = "duplicate_of"
    INTRODUCES_CONCEPT = "introduces_concept"
    OVER_BUDGET = "over_budget"


class ChapterRange(BaseModel):
    """Textbook page span for one chapter, from the PDF outline or overridden."""

    chapter: str
    title: str = ""
    page_start: int
    page_end: int
    manual_override: bool = False


class Lecture(BaseModel):
    """One folder inside a course: a slide deck, note photos, and its runs."""

    id: str
    course: str
    title: str
    chapter: str = ""
    created_at: datetime
    slides_name: str = ""
    note_count: int = 0

    @property
    def ready(self) -> bool:
        return bool(self.slides_name) and self.note_count > 0


class Chunk(BaseModel):
    """A child chunk: what gets embedded."""

    id: str
    course: str
    chapter: str
    text: str
    section_path: str
    page_start: int
    page_end: int
    parent_id: str
    chunk_type: ChunkType = ChunkType.PROSE
    token_estimate: int = 0


class Parent(BaseModel):
    """A parent section: what gets handed to the model."""

    id: str
    course: str
    chapter: str
    heading: str
    section_path: str
    text: str
    page_start: int
    page_end: int
    token_estimate: int = 0


class SlidePage(BaseModel):
    page: int
    deck: str
    markdown: str
    is_figure_only: bool = False
    figure_description: str = ""


class NotePage(BaseModel):
    page: int
    image_path: str
    content_hash: str
    markdown: str
    edited_by_student: bool = False


class Concept(BaseModel):
    """One distinct idea the slides raise."""

    id: str
    name: str
    slide_pages: list[int] = Field(default_factory=list)
    slide_definition: str = ""
    covered_by_notes: bool = False
    note_pages: list[int] = Field(default_factory=list)


class Gap(BaseModel):
    """A concept the slides raise but do not resolve."""

    id: str
    concept_id: str
    kind: GapKind
    question: str
    query: str


class Candidate(BaseModel):
    """A retrieved textbook passage under consideration."""

    id: str
    gap_id: str
    parent_id: str
    text: str
    section_path: str
    page_start: int
    page_end: int
    chapter: str
    retrieval_score: float = 0.0
    relevance: float | None = None
    necessity: float | None = None
    draft_similarity: float | None = None
    introduced_terms: list[str] = Field(default_factory=list)
    token_estimate: int = 0


class Admitted(BaseModel):
    """A passage that survived all seven mechanisms."""

    candidate_id: str
    gap_id: str
    text: str
    citation: str
    page_start: int
    page_end: int
    token_estimate: int
    necessity: float


class Rejection(BaseModel):
    """Spec 7.8: why a textbook candidate was rejected, with its score."""

    candidate_id: str
    gap_id: str
    mechanism: str
    reason: Reason
    score: float | None = None
    detail: str = ""


class RetrievalOutcome(BaseModel):
    admitted: list[Admitted] = Field(default_factory=list)
    rejections: list[Rejection] = Field(default_factory=list)
    tokens_admitted: int = 0
    budget: int = 0


class DraftedConcept(BaseModel):
    concept_id: str
    heading: str
    body: str
    slide_pages: list[int] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


class RunRecord(BaseModel):
    id: str
    course: str
    chapter: str
    lecture: str = ""
    status: str = "pending"
    created_at: datetime
    updated_at: datetime
    document_path: str | None = None
    error: str | None = None


class GraphState(TypedDict, total=False):
    """LangGraph channel state: lists accumulate, scalars overwrite."""

    run_id: str
    course: str
    chapter: str
    lecture: str
    slides_dir: str
    notes_dir: str

    # Written by one node each and replaced wholesale at the review interrupt, so these must no.
    slides: list[SlidePage]
    notes: list[NotePage]
    notes_approved: bool

    concepts: list[Concept]
    gaps: list[Gap]
    drafts: list[DraftedConcept]

    retrieval: RetrievalOutcome
    document: str
    verify_rounds: int
    unverified: list[str]
    artifacts: dict[str, Any]
