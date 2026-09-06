from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from studysynth.config import Settings
from studysynth.clients.embeddings import MockEmbeddings
from studysynth.clients.llm import LlmClient
from studysynth.models import Candidate, Concept, DraftedConcept, Gap, GapKind, Source

Model = TypeVar("Model", bound=BaseModel)


class StubLlm(LlmClient):
    """Returns a programmed verdict per candidate id, so a mechanism can be
    tested against exact scores rather than against a model's mood."""

    def __init__(self, verdicts: dict[str, dict[str, Any]] | None = None) -> None:
        self.verdicts = verdicts or {}
        self.jobs: list[tuple[str, str]] = []

    def complete(self, prompt: str, *, job: str, effort: str | None = None) -> str:
        self.jobs.append((job, ""))
        return self.verdicts.get(job, {}).get("text", "")

    def structured(
        self, prompt: str, schema: type[Model], *, job: str, key: str = "",
        effort: str | None = None,
    ) -> Model:
        self.jobs.append((job, key))
        payload = self.verdicts.get(job, {}).get(key)
        if payload is None:
            payload = self.verdicts.get(job, {}).get("default", {})
        return schema.model_validate(payload)

    def vision(self, prompt: str, image: Path, *, job: str) -> str:
        self.jobs.append((job, image.name))
        return self.verdicts.get(job, {}).get(image.name, "")


def mock_settings(tmp_path: Path) -> Settings:
    """Settings with .env ignored and both backends pinned to mock.

    Without `_env_file=None` a developer's real .env selects the Claude
    backend and the suite starts making paid API calls.
    """
    loaded = Settings(_env_file=None)
    loaded.paths.root = tmp_path
    loaded.llm.backend = "mock"
    loaded.llm.fixtures = Path("fixtures/llm")
    loaded.embeddings.backend = "mock"
    loaded.anthropic_api_key = None
    return loaded.validated()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return mock_settings(tmp_path)


@pytest.fixture
def embeddings() -> MockEmbeddings:
    return MockEmbeddings(256)


def make_candidate(
    identifier: str,
    text: str = "A textbook passage about the inductive step.",
    chapter: str = "induction",
    tokens: int = 100,
    **extra: Any,
) -> Candidate:
    return Candidate(
        id=identifier,
        gap_id="gap-1",
        parent_id=identifier,
        text=text,
        section_path="Induction > Strong induction",
        page_start=10,
        page_end=14,
        chapter=chapter,
        retrieval_score=0.7,
        token_estimate=tokens,
        **extra,
    )


def make_gap(identifier: str = "gap-1") -> Gap:
    return Gap(
        id=identifier,
        concept_id="concept-1",
        kind=GapKind.UNSUPPORTED,
        question="Why does strong induction need no separate base case here?",
        query="strong induction base case requirement",
    )


def make_concept(name: str = "Strong induction") -> Concept:
    return Concept(id="concept-1", name=name, slide_pages=[3, 4], slide_definition="...")


def make_draft(body: str = "Strong induction assumes every smaller case.") -> DraftedConcept:
    return DraftedConcept(
        concept_id="concept-1",
        heading="Strong induction (Pages 3-4)",
        body=body,
        slide_pages=[3, 4],
        sources=[Source.SLIDES, Source.NOTES],
    )
