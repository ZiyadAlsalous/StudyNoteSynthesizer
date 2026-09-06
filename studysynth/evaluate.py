"""Offline evaluation against a labeled chapter.

Run: python -m studysynth.evaluate --labels labels.json --document out.md

Bloat rate is the metric this project exists to keep down; the others exist so
that keeping it down cannot be achieved by producing a worse document.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .ingest import estimate_tokens
from .models import RetrievalOutcome
from .render import TEXTBOOK_TAG


class EvaluationError(RuntimeError):
    pass


@dataclass
class Labels:
    """Hand-verified ground truth for one chapter. Building this is tedious and
    skipping it turns the eval harness into decoration."""

    course: str
    chapter: str
    concepts: list[str]
    relevant_pages: dict[str, list[int]]

    @staticmethod
    def load(path: Path) -> "Labels":
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = {"course", "chapter", "concepts"} - payload.keys()
        if missing:
            raise EvaluationError(f"{path} is missing {sorted(missing)}")
        return Labels(
            course=payload["course"],
            chapter=payload["chapter"],
            concepts=payload["concepts"],
            relevant_pages={k: list(v) for k, v in payload.get("relevant_pages", {}).items()},
        )


@dataclass
class Scores:
    coverage: float
    bloat_rate: float
    bloat_ceiling: float
    within_budget: bool
    citation_validity: float
    context_precision: float
    context_recall: float


def coverage(document: str, concepts: Sequence[str]) -> float:
    """Share of slide concepts that reached the document at all."""
    if not concepts:
        return 1.0
    lowered = document.lower()
    return sum(1 for concept in concepts if concept.lower() in lowered) / len(concepts)


def bloat_rate(document: str) -> float:
    """Textbook tokens over total document tokens, measured on the output rather
    than on what the gate thought it admitted."""
    total = estimate_tokens(document)
    if total == 0:
        return 0.0
    textbook = 0
    for match in TEXTBOOK_TAG.finditer(document):
        tail = document[match.end() :]
        passage = tail.split("\n\n", 1)[0]
        textbook += estimate_tokens(passage)
    return textbook / total


def citation_validity(document: str, page_count: int) -> float:
    """Share of page citations that could resolve. A citation past the end of
    the source is the cheap half of the check; the model-graded half is in the
    verify node."""
    citations = re.findall(r"\(Pages? ([\d,\s-]+)\)", document) + [
        f"{a}-{b}" for a, b in TEXTBOOK_TAG.findall(document)
    ]
    if not citations:
        return 0.0
    valid = 0
    for citation in citations:
        numbers = [int(n) for n in re.findall(r"\d+", citation)]
        if numbers and all(1 <= n <= page_count for n in numbers):
            valid += 1
    return valid / len(citations)


def context_scores(outcome: RetrievalOutcome, relevant: dict[str, list[int]]) -> tuple[float, float]:
    """Precision and recall over gap-to-page pairs."""
    if not relevant:
        return 0.0, 0.0
    wanted = {(gap, page) for gap, pages in relevant.items() for page in pages}
    got = {
        (passage.gap_id, page)
        for passage in outcome.admitted
        for page in range(passage.page_start, passage.page_end + 1)
    }
    hits = len(wanted & got)
    precision = hits / len(got) if got else 0.0
    recall = hits / len(wanted)
    return precision, recall


def score(
    document: str, outcome: RetrievalOutcome, labels: Labels, page_count: int, ceiling: float
) -> Scores:
    precision, recall = context_scores(outcome, labels.relevant_pages)
    rate = bloat_rate(document)
    return Scores(
        coverage=coverage(document, labels.concepts),
        bloat_rate=rate,
        bloat_ceiling=ceiling,
        within_budget=rate <= ceiling,
        citation_validity=citation_validity(document, page_count),
        context_precision=precision,
        context_recall=recall,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="studysynth.evaluate")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, help="retrieval outcome JSON from the run")
    parser.add_argument("--pages", type=int, default=999, help="page count of the source material")
    parser.add_argument("--ceiling", type=float, default=0.15)
    arguments = parser.parse_args(argv)

    labels = Labels.load(arguments.labels)
    document = arguments.document.read_text(encoding="utf-8")
    outcome = (
        RetrievalOutcome.model_validate_json(arguments.outcome.read_text(encoding="utf-8"))
        if arguments.outcome
        else RetrievalOutcome()
    )
    scores = score(document, outcome, labels, arguments.pages, arguments.ceiling)
    print(json.dumps(asdict(scores), indent=2))
    return 0 if scores.within_budget else 1


if __name__ == "__main__":
    sys.exit(main())
