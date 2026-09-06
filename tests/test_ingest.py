"""Regression tests for two bugs found against a real 1677-page textbook.

Both were invisible against generated fixtures and only appeared on a book with
a nested outline and no blank lines in its extracted text.
"""

from __future__ import annotations

from pathlib import Path

import pypdf
import pytest

from studysynth.config import Settings
from studysynth.ingest import TextbookIngestor, chapter_ranges, estimate_tokens

from .samples import write_pdf


@pytest.fixture
def nested_textbook(tmp_path: Path) -> Path:
    """A book shaped like a real one: parts, chapters, and subsections."""
    source = write_pdf(tmp_path / "flat.pdf", [[f"page {n} body text"] for n in range(1, 21)])
    reader = pypdf.PdfReader(str(source))
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    part = writer.add_outline_item("I Foundations", 0)
    chapter_one = writer.add_outline_item("1 The Role of Algorithms", 1, parent=part)
    writer.add_outline_item("1.1 Algorithms", 2, parent=chapter_one)
    writer.add_outline_item("1.2 Algorithms as a technology", 4, parent=chapter_one)
    chapter_two = writer.add_outline_item("2 Getting Started", 9, parent=part)
    writer.add_outline_item("2.1 Insertion sort", 10, parent=chapter_two)
    writer.add_outline_item("2.2 Analyzing algorithms", 13, parent=chapter_two)
    chapter_three = writer.add_outline_item("3 Divide-and-Conquer", 15, parent=part)
    writer.add_outline_item("3.1 Multiplying matrices", 16, parent=chapter_three)

    target = tmp_path / "textbook.pdf"
    with target.open("wb") as handle:
        writer.write(handle)
    return target


def test_chapters_come_from_the_chapter_level_not_the_whole_outline(nested_textbook):
    ranges = chapter_ranges(nested_textbook)
    assert [r.title for r in ranges] == [
        "1 The Role of Algorithms",
        "2 Getting Started",
        "3 Divide-and-Conquer",
    ]


def test_a_chapter_does_not_end_where_its_first_subsection_begins(nested_textbook):
    """The original bug: `4 Divide-and-Conquer` came back as 6 pages instead of
    65, because 4.1 was treated as the next chapter."""
    ranges = {r.title: r for r in chapter_ranges(nested_textbook)}
    chapter_one = ranges["1 The Role of Algorithms"]
    assert chapter_one.page_start == 2
    assert chapter_one.page_end == 9, "chapter truncated at its first subsection"

    last = ranges["3 Divide-and-Conquer"]
    assert last.page_end == 20, "the final chapter should run to the end of the book"


def test_an_explicit_level_overrides_the_detected_one(nested_textbook):
    assert [r.title for r in chapter_ranges(nested_textbook, level=0)] == ["I Foundations"]


def test_splitter_breaks_text_that_contains_no_blank_lines():
    """Extracted PDF text often has zero blank lines. Breaking only on
    paragraphs meant never breaking, and one chapter became one chunk."""
    dense = "\n".join(f"line {n} of continuous prose with no paragraph breaks" for n in range(200))
    assert "\n\n" not in dense
    pieces = TextbookIngestor._split_to_size(dense, 100)
    assert len(pieces) > 1
    assert all(estimate_tokens(piece) <= 100 for piece in pieces)


def test_splitter_stays_within_the_limit_across_many_lines():
    """Per-line token estimates floor, so summing them undercounts the joined
    string and pieces drifted over the configured limit."""
    text = "\n".join("x" * 37 for _ in range(400))
    for limit in (50, 137, 320, 1800):
        pieces = TextbookIngestor._split_to_size(text, limit)
        assert max(estimate_tokens(p) for p in pieces) <= limit, f"overflow at limit {limit}"


def test_splitter_never_breaks_inside_a_display_formula():
    body = "prose line\n" * 10 + "$$\n" + "a + b = c\n" * 40 + "$$\n" + "more prose\n" * 10
    pieces = TextbookIngestor._split_to_size(body, 40)
    for piece in pieces:
        assert piece.count("$$") % 2 == 0, "a formula was split across two chunks"


def test_a_single_over_long_line_splits_at_sentence_ends():
    line = " ".join(f"Sentence number {n} says something." for n in range(120))
    pieces = TextbookIngestor._split_to_size(line, 60)
    assert len(pieces) > 1
    assert all(estimate_tokens(piece) <= 60 for piece in pieces)


def test_parents_and_children_respect_the_configured_limits(nested_textbook):
    settings = Settings(_env_file=None)
    parents, chunks = TextbookIngestor(settings).ingest(
        nested_textbook, "cs3340", chapter_ranges(nested_textbook)
    )
    assert parents and chunks
    assert max(p.token_estimate for p in parents) <= settings.chunks.parent_tokens
    assert max(c.token_estimate for c in chunks) <= settings.chunks.child_tokens
    assert all(c.parent_id in {p.id for p in parents} for c in chunks)
