"""LangGraph assembly: nodes, edges, checkpointer, interrupt, verify loop.

A node does one thing and fails visibly. Nothing here swallows an exception to
keep a run alive: the checkpointer means a failed run resumes at the node that
failed rather than starting over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Settings
from .embeddings import EmbeddingBackend
from .ingest import NoteIngestor, SlideIngestor
from .llm import LlmClient, PromptLibrary
from .models import (
    Concept,
    DraftedConcept,
    Gap,
    GraphState,
    RetrievalOutcome,
    Source,
)
from .retrieval import TextbookGate
from .store import Catalogue, Places

INGEST_SLIDES = "ingest_slides"
INGEST_NOTES = "ingest_notes"
EXTRACT_CONCEPTS = "extract_concepts"
FIND_GAPS = "find_gaps"
DRAFT = "draft_concepts"
RETRIEVE = "retrieve_textbook"
SYNTHESIZE = "synthesize"
VERIFY = "verify"


class GraphError(RuntimeError):
    pass


class ConceptList(BaseModel):
    concepts: list[Concept] = Field(default_factory=list)


class GapList(BaseModel):
    gaps: list[Gap] = Field(default_factory=list)


class VerifyVerdict(BaseModel):
    unverified: list[str] = Field(default_factory=list)


class Nodes:
    """The node bodies. Kept off the graph so each can be called in a test."""

    def __init__(
        self,
        settings: Settings,
        llm: LlmClient,
        embeddings: EmbeddingBackend,
        gate: TextbookGate,
        catalogue: Catalogue,
        places: Places,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._gate = gate
        self._catalogue = catalogue
        self._places = places
        self._prompts = PromptLibrary(settings.prompts_dir)
        self._slides = SlideIngestor(settings, llm)
        self._notes = NoteIngestor(settings, llm, places)

    def ingest_slides(self, state: GraphState) -> dict[str, Any]:
        return {"slides": self._slides.ingest(Path(state["slides_dir"]))}

    def ingest_notes(self, state: GraphState) -> dict[str, Any]:
        return {"notes": self._notes.ingest(Path(state["notes_dir"]))}

    def extract_concepts(self, state: GraphState) -> dict[str, Any]:
        prompt = self._prompts.render(
            "extract_concepts",
            {
                "chapter": state["chapter"],
                "slides": _join(page.markdown for page in state.get("slides", [])),
                "notes": _join(page.markdown for page in state.get("notes", [])),
            },
        )
        found = self._llm.structured(prompt, ConceptList, job="extract_concepts")
        if not found.concepts:
            raise GraphError("Concept extraction returned nothing; the slides did not parse")
        return {"concepts": found.concepts}

    def find_gaps(self, state: GraphState) -> dict[str, Any]:
        concepts = state.get("concepts", [])
        prompt = self._prompts.render(
            "find_gaps",
            {
                "chapter": state["chapter"],
                "concepts": _join(
                    f"- {c.name}: {c.slide_definition or '(no definition on the slides)'} "
                    f"[notes: {'yes' if c.covered_by_notes else 'no'}]"
                    for c in concepts
                ),
            },
        )
        return {"gaps": self._llm.structured(prompt, GapList, job="find_gaps").gaps}

    def draft_concepts(self, state: GraphState) -> dict[str, Any]:
        """Drafted from slides and notes only. The textbook has not been read yet,
        because necessity (spec 7.4) is graded against this draft."""
        slides = {page.page: page.markdown for page in state.get("slides", [])}
        notes = _join(page.markdown for page in state.get("notes", []))
        drafts: list[DraftedConcept] = []
        for concept in state.get("concepts", []):
            prompt = self._prompts.render(
                "draft_concept",
                {
                    "name": concept.name,
                    "slides": _join(slides.get(page, "") for page in concept.slide_pages),
                    "notes": notes,
                },
            )
            body = self._llm.complete(prompt, job="draft_concept")
            pages = ", ".join(str(page) for page in concept.slide_pages)
            drafts.append(
                DraftedConcept(
                    concept_id=concept.id,
                    heading=f"{concept.name} (Pages {pages})" if pages else concept.name,
                    body=body,
                    slide_pages=concept.slide_pages,
                    sources=[Source.SLIDES] + ([Source.NOTES] if concept.covered_by_notes else []),
                )
            )
        return {"drafts": drafts}

    def retrieve_textbook(self, state: GraphState) -> dict[str, Any]:
        outcome = self._gate.run(
            course=state["course"],
            chapter=state["chapter"],
            gaps=state.get("gaps", []),
            concepts=state.get("concepts", []),
            drafts=state.get("drafts", []),
        )
        self._catalogue.log_rejections(state["run_id"], outcome.rejections)
        return {"retrieval": outcome}

    def synthesize(self, state: GraphState) -> dict[str, Any]:
        drafts = state.get("drafts", [])
        outcome = state.get("retrieval") or RetrievalOutcome()
        by_gap: dict[str, list[str]] = {}
        for passage in outcome.admitted:
            by_gap.setdefault(passage.gap_id, []).append(f"{passage.citation} {passage.text}")

        gap_to_concept = {gap.id: gap.concept_id for gap in state.get("gaps", [])}
        additions: dict[str, list[str]] = {}
        for gap_id, passages in by_gap.items():
            additions.setdefault(gap_to_concept.get(gap_id, ""), []).extend(passages)

        sections = []
        for draft in drafts:
            block = f"### {draft.heading}\n\n{draft.body}"
            extra = additions.get(draft.concept_id)
            if extra:
                block += "\n\n" + "\n\n".join(extra)
            sections.append(block)

        uncovered = [
            concept.name
            for concept in state.get("concepts", [])
            if not concept.covered_by_notes
        ]
        prompt = self._prompts.render(
            "synthesize_chapter",
            {
                "chapter": state["chapter"],
                "sections": _join(sections),
                "uncovered": _join(f"- {name}" for name in uncovered),
            },
        )
        return {"document": self._llm.complete(prompt, job="synthesize_chapter")}

    def verify(self, state: GraphState) -> dict[str, Any]:
        prompt = self._prompts.render(
            "verify_claims",
            {
                "document": state.get("document", ""),
                "sources": _join(page.markdown for page in state.get("slides", [])),
            },
        )
        verdict = self._llm.structured(prompt, VerifyVerdict, job="verify_claims")
        return {
            "unverified": verdict.unverified,
            "verify_rounds": state.get("verify_rounds", 0) + 1,
        }


def _join(parts: Any) -> str:
    return "\n\n".join(part for part in parts if part)


class Runner:
    """Compiles the graph and streams a run. Owns the checkpointer's lifetime."""

    def __init__(self, settings: Settings, nodes: Nodes) -> None:
        self._settings = settings
        self._nodes = nodes

    def _needs_another_round(self, state: GraphState) -> str:
        """Bounded: a verify loop that can spin forever is worse than an
        unverified document, because it never ships either."""
        rounds = state.get("verify_rounds", 0)
        if state.get("unverified") and rounds < self._settings.verify.max_rounds:
            return SYNTHESIZE
        return END

    def _graph(self) -> StateGraph[GraphState, Any, Any, Any]:
        graph: StateGraph[GraphState, Any, Any, Any] = StateGraph(GraphState)
        graph.add_node(INGEST_SLIDES, self._nodes.ingest_slides)
        graph.add_node(INGEST_NOTES, self._nodes.ingest_notes)
        graph.add_node(EXTRACT_CONCEPTS, self._nodes.extract_concepts)
        graph.add_node(FIND_GAPS, self._nodes.find_gaps)
        graph.add_node(DRAFT, self._nodes.draft_concepts)
        graph.add_node(RETRIEVE, self._nodes.retrieve_textbook)
        graph.add_node(SYNTHESIZE, self._nodes.synthesize)
        graph.add_node(VERIFY, self._nodes.verify)

        graph.add_edge(START, INGEST_SLIDES)
        graph.add_edge(START, INGEST_NOTES)
        graph.add_edge(INGEST_SLIDES, EXTRACT_CONCEPTS)
        graph.add_edge(INGEST_NOTES, EXTRACT_CONCEPTS)
        graph.add_edge(EXTRACT_CONCEPTS, FIND_GAPS)
        graph.add_edge(FIND_GAPS, DRAFT)
        graph.add_edge(DRAFT, RETRIEVE)
        graph.add_edge(RETRIEVE, SYNTHESIZE)
        graph.add_edge(SYNTHESIZE, VERIFY)
        graph.add_conditional_edges(VERIFY, self._needs_another_round, [SYNTHESIZE, END])
        return graph

    def _saver(self) -> Any:
        return SqliteSaver.from_conn_string(str(self._settings.paths.checkpoints))

    def stream(self, run_id: str, **inputs: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yields (node_name, update) as the run progresses.

        Interrupts before concept extraction so the student can correct the OCR
        transcript, which is the least reliable input in the system.
        """
        self._settings.paths.checkpoints.parent.mkdir(parents=True, exist_ok=True)
        with self._saver() as saver:
            app = self._graph().compile(checkpointer=saver, interrupt_before=[EXTRACT_CONCEPTS])
            config: RunnableConfig = {"configurable": {"thread_id": run_id}}
            pending = app.get_state(config).next
            payload: dict[str, Any] | None = None if pending else {"run_id": run_id, **inputs}
            for event in app.stream(payload, config, stream_mode="updates"):
                for node, update in event.items():
                    yield node, update

    def state(self, run_id: str) -> GraphState:
        with self._saver() as saver:
            app = self._graph().compile(checkpointer=saver, interrupt_before=[EXTRACT_CONCEPTS])
            config: RunnableConfig = {"configurable": {"thread_id": run_id}}
            return cast(GraphState, app.get_state(config).values)

    def pending(self, run_id: str) -> tuple[str, ...]:
        with self._saver() as saver:
            app = self._graph().compile(checkpointer=saver, interrupt_before=[EXTRACT_CONCEPTS])
            config: RunnableConfig = {"configurable": {"thread_id": run_id}}
            return tuple(app.get_state(config).next)

    def approve_notes(self, run_id: str, edited: list[dict[str, Any]] | None = None) -> None:
        """Resume past the review interrupt, optionally with corrected OCR."""
        with self._saver() as saver:
            app = self._graph().compile(checkpointer=saver, interrupt_before=[EXTRACT_CONCEPTS])
            config: RunnableConfig = {"configurable": {"thread_id": run_id}}
            update: dict[str, Any] = {"notes_approved": True}
            if edited is not None:
                update["notes"] = edited
            # as_node is required: slides and notes ingest in parallel, so
            # LangGraph cannot infer which branch this update belongs to.
            app.update_state(config, update, as_node=INGEST_NOTES)
