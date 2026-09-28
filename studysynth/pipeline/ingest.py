"""Turning three kinds of source into chunks the rest of the system can use."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..clients.llm import LlmClient, PromptLibrary
from ..config import Settings
from ..models import ChapterRange, Chunk, NotePage, Parent, SlidePage, SourcePage
from ..store import Catalogue


class IngestError(RuntimeError):
    pass


class OutlineMissing(IngestError):
    """The textbook PDF has no usable outline; the UI must supply page ranges."""


_HEADING = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\s*\|", re.MULTILINE)
_CHAPTER_NUMBER = re.compile(r"^\d+\s+\S")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_FORMULA_FENCE = re.compile(r"\$\$")


_CHARS_PER_TOKEN = 4
_OVERSHOOT = 2


def estimate_tokens(text: str) -> int:
    """Four characters per token. Close enough to enforce a budget against."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def content_hash(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def chapter_ranges(pdf: Path, level: int | None = None) -> list[ChapterRange]:
    """Read chapter boundaries from the PDF's bookmarks (its outline)."""
    import pymupdf

    with pymupdf.open(str(pdf)) as document:  # type: ignore[no-untyped-call]
        total = int(document.page_count)
        # get_toc gives [level, title, page]; levels start at 1, pages at 1, and a
        # bookmark with no target page is -1.
        entries = [
            (int(depth) - 1, str(title), int(page))
            for depth, title, page in document.get_toc()
            if int(page) > 0
        ]
    if not entries:
        raise OutlineMissing(f"{pdf.name} has no outline; set page ranges manually")

    depth = level if level is not None else _chapter_depth(entries)
    ranges: list[ChapterRange] = []
    used: set[str] = set()
    for index, (own_depth, title, start) in enumerate(entries):
        if own_depth != depth:
            continue
        end = total
        for later_depth, _, later_start in entries[index + 1 :]:
            if later_depth <= depth:
                end = later_start - 1
                break
        # Titles repeat (every part of a book may open with "Introduction"), so a
        # repeated id gets its start page to keep each chapter distinct.
        identifier = slug(title) or "chapter"
        if identifier in used:
            identifier = f"{identifier}-p{start}"
        used.add(identifier)
        ranges.append(
            ChapterRange(
                chapter=identifier,
                title=title.strip(),
                page_start=start,
                page_end=max(start, end),
            )
        )
    return ranges


def _chapter_depth(entries: list[tuple[int, str, int]]) -> int:
    """The shallowest outline level that looks like numbered chapters."""
    by_depth: dict[int, int] = {}
    for depth, title, _ in entries:
        if _CHAPTER_NUMBER.match(title.strip()):
            by_depth[depth] = by_depth.get(depth, 0) + 1
    for depth in sorted(by_depth):
        if by_depth[depth] >= 3:
            return depth
    return min((depth for depth, _, _ in entries), default=0)


def slug(text: str) -> str:
    """A filesystem and URL safe identifier, empty when nothing survives."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class TextbookIngestor:
    """Structural chunking into parent sections and the child chunks that index them."""

    def __init__(self, settings: Settings) -> None:
        self._chunks = settings.chunks

    def ingest(
        self, pdf: Path, course: str, ranges: Sequence[ChapterRange]
    ) -> tuple[list[Parent], list[Chunk]]:
        pages = self._pages(pdf)
        parents: list[Parent] = []
        chunks: list[Chunk] = []
        for span in ranges:
            body = "\n".join(pages[span.page_start - 1 : span.page_end])
            for parent in self._sections(body, course, span):
                parents.append(parent)
                chunks.extend(self._children(parent))
        return parents, chunks

    @staticmethod
    def _pages(pdf: Path) -> list[str]:
        import pymupdf

        with pymupdf.open(str(pdf)) as document:  # type: ignore[no-untyped-call]
            return [page.get_text() for page in document]

    def _sections(self, body: str, course: str, span: ChapterRange) -> list[Parent]:
        """Split on headings first, then size, never inside a table or formula.

        Whatever precedes the first heading is a block of its own under the
        chapter title. It is usually the chapter introduction, where a textbook
        tends to put the definitions the rest of the chapter assumes.
        """
        marks = list(_HEADING.finditer(body))
        title = span.title or span.chapter
        if not marks:
            blocks = [(title, body)]
        else:
            blocks = []
            lead = body[: marks[0].start()]
            if lead.strip():
                blocks.append((title, lead))
            for index, mark in enumerate(marks):
                start = mark.end()
                stop = marks[index + 1].start() if index + 1 < len(marks) else len(body)
                blocks.append((mark.group(2).strip(), body[start:stop]))

        parents: list[Parent] = []
        for heading, text in blocks:
            for part, piece in enumerate(self._split_to_size(text, self._chunks.parent_tokens)):
                if len(piece.strip()) < self._chunks.min_chunk_chars:
                    continue
                identifier = f"{course}:{span.chapter}:{slug(heading) or 'chapter'}:{part}"
                parents.append(
                    Parent(
                        id=identifier,
                        course=course,
                        chapter=span.chapter,
                        heading=heading,
                        section_path=f"{span.title or span.chapter} > {heading}",
                        text=piece.strip(),
                        page_start=span.page_start,
                        page_end=span.page_end,
                        token_estimate=estimate_tokens(piece),
                    )
                )
        return parents

    @staticmethod
    def _split_to_size(text: str, limit: int) -> list[str]:
        """Split text into pieces of at most `limit` tokens.

        A formula or a table is kept whole even when that overshoots, because a
        half formula is worse than a long chunk. `_OVERSHOOT` bounds how far
        that may go, so malformed input such as a `$$` the extractor never
        closed cannot swallow the rest of the text into one piece.
        """
        budget = limit * _CHARS_PER_TOKEN
        ceiling = budget * _OVERSHOOT
        if len(text) <= budget:
            return [text]

        pieces: list[str] = []
        current: list[str] = []
        size = 0
        in_formula = False

        def flush() -> None:
            nonlocal current, size
            if current:
                pieces.append("".join(current))
                current, size = [], 0

        for line in text.splitlines(keepends=True):
            fences = len(_FORMULA_FENCE.findall(line))
            formula = in_formula or fences > 0
            if fences % 2:
                in_formula = not in_formula
            elif not line.strip():
                in_formula = False
            breakable = not formula and not _TABLE_ROW.match(line)
            splittable = not formula or len(line) > ceiling
            parts = (
                TextbookIngestor._sentences(line, budget)
                if len(line) > budget and splittable
                else [line]
            )
            for part in parts:
                over = size + len(part)
                if current and (over > ceiling or (over > budget and breakable)):
                    flush()
                current.append(part)
                size += len(part)
        flush()
        return pieces

    @staticmethod
    def _sentences(text: str, budget: int) -> list[str]:
        """Break one over-long line at sentence ends."""
        parts: list[str] = []
        current = ""
        for piece in _SENTENCE.split(text):
            if current and len(current) + len(piece) > budget:
                parts.append(current)
                current = ""
            current += piece + " "
        if current.strip():
            parts.append(current)
        return parts or [text]

    def _children(self, parent: Parent) -> list[Chunk]:
        pieces = self._split_to_size(parent.text, self._chunks.child_tokens)
        children: list[Chunk] = []
        for index, piece in enumerate(pieces):
            if len(piece.strip()) < self._chunks.min_chunk_chars:
                continue
            children.append(
                Chunk(
                    id=f"{parent.id}#{index}",
                    course=parent.course,
                    chapter=parent.chapter,
                    text=piece.strip(),
                    section_path=parent.section_path,
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    parent_id=parent.id,
                    token_estimate=estimate_tokens(piece),
                )
            )
        return children


class SlideIngestor:
    """One slide is one chunk, with a `## Page N` marker kept for citation.

    Each deck is read once. Its text is stored against a hash of the file, so a
    later run reads it back and only a new or changed deck is parsed. Pages are
    numbered straight through the decks in name order.
    """

    KIND = "slides"
    # Stored text is tagged with the reader that produced it, so changing the
    # reader re-reads every deck once instead of trusting the old text.
    READER = "pymupdf"

    def __init__(self, client: LlmClient, catalogue: Catalogue) -> None:
        self._client = client
        self._catalogue = catalogue

    def ingest(self, source: Path, course: str = "", lecture: str = "") -> list[SlidePage]:
        if source.is_dir():
            decks = sorted(
                path for path in source.iterdir() if path.suffix.lower() in {".pdf", ".pptx"}
            )
        else:
            decks = [source]
        if not decks:
            raise IngestError(f"No slide decks found at {source}")
        pages: list[SlidePage] = []
        for deck in decks:
            for text in self._texts(deck, course, lecture):
                number = len(pages) + 1
                pages.append(
                    SlidePage(
                        page=number, deck=deck.stem, markdown=f"## Page {number}\n\n{text.strip()}"
                    )
                )
        return pages

    def _texts(self, deck: Path, course: str, lecture: str) -> list[str]:
        digest = f"{self.READER}:{content_hash(deck.read_bytes())}"
        stored = self._catalogue.stored_pages(course, lecture, self.KIND, deck.name, digest)
        if stored is not None:
            return [page.markdown for page in stored]
        texts = self._pptx(deck) if deck.suffix.lower() == ".pptx" else self._pdf(deck)
        self._catalogue.store_pages(
            course,
            lecture,
            self.KIND,
            deck.name,
            digest,
            [SourcePage(page=number, markdown=text) for number, text in enumerate(texts, 1)],
        )
        return texts

    @staticmethod
    def _pdf(deck: Path) -> list[str]:
        """PyMuPDF, not pypdf: pypdf scrambles text drawn in custom fonts."""
        import pymupdf

        with pymupdf.open(str(deck)) as document:  # type: ignore[no-untyped-call]
            return [page.get_text() for page in document]

    @staticmethod
    def _pptx(deck: Path) -> list[str]:
        from pptx import Presentation

        texts: list[str] = []
        for slide in Presentation(str(deck)).slides:
            body = "\n".join(
                shape.text_frame.text for shape in slide.shapes if shape.has_text_frame
            )
            notes = ""
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text
            if notes.strip():
                body = f"{body}\n\n_Speaker notes:_ {notes.strip()}"
            texts.append(body)
        return texts


class NoteIngestor:
    """Handwriting to Markdown, each page transcribed once.

    Transcripts are stored against a hash of each notes PDF, so a later run reads
    them back and only a new or changed PDF goes through OCR. A page already seen
    in any file reuses its transcript, preferring the student's correction.
    """

    KIND = "notes"

    def __init__(self, settings: Settings, client: LlmClient, catalogue: Catalogue) -> None:
        self._settings = settings
        self._client = client
        self._catalogue = catalogue
        self._prompts = PromptLibrary(settings.prompts_dir)

    def ingest(self, source: Path, course: str = "", lecture: str = "") -> list[NotePage]:
        pdfs = self._pdfs(source)
        if not pdfs:
            raise IngestError(f"No notes PDF found at {source}")

        files: list[tuple[Path, str, list[Path], list[SourcePage] | None]] = []
        for pdf in pdfs:
            images = self.rasterise(pdf, pdf.parent / "pages")
            digest = content_hash(pdf.read_bytes())
            stored = self._catalogue.stored_pages(course, lecture, self.KIND, pdf.name, digest)
            if stored is not None and len(stored) != len(images):
                stored = None
            files.append((pdf, digest, images, stored))

        new_images = [image for _, _, images, stored in files if stored is None for image in images]
        transcripts = dict(zip(new_images, self._transcribe(new_images), strict=True))

        pages: list[NotePage] = []
        for pdf, digest, images, stored in files:
            if stored is None:
                stored = [
                    SourcePage(page=number, content_hash=hashed, markdown=text)
                    for number, (hashed, text) in enumerate(
                        (transcripts[image] for image in images), 1
                    )
                ]
                self._catalogue.store_pages(course, lecture, self.KIND, pdf.name, digest, stored)
            for image, page in zip(images, stored, strict=True):
                pages.append(
                    NotePage(
                        page=len(pages) + 1,
                        image_path=str(image),
                        content_hash=page.content_hash,
                        markdown=page.markdown,
                    )
                )
        return pages

    def _transcribe(self, images: list[Path]) -> list[tuple[str, str]]:
        """Only pages never seen before reach the vision model."""
        if not images:
            return []
        prompt = self._prompts.render("ocr_notes")

        def transcribe(image: Path) -> tuple[str, str]:
            digest = content_hash(image.read_bytes())
            known = self._catalogue.known_transcript(digest)
            if known is not None:
                return digest, known
            return digest, self._client.vision(prompt, image, job="ocr_notes")

        workers = min(self._settings.notes.max_parallel_ocr, len(images))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(transcribe, images))

    @staticmethod
    def _pdfs(source: Path) -> list[Path]:
        """Notes are PDFs: GoodNotes exports or scans, of any length."""
        if source.is_dir():
            return sorted(p for p in source.iterdir() if p.suffix.lower() == ".pdf")
        return [source]

    def rasterise(self, pdf: Path, target: Path) -> list[Path]:
        """A vision model needs pixels, so each page is rendered to a PNG."""
        import pymupdf

        target.mkdir(parents=True, exist_ok=True)
        zoom = self._settings.notes.render_dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)  # type: ignore[no-untyped-call]
        rendered: list[Path] = []
        with pymupdf.open(str(pdf)) as document:  # type: ignore[no-untyped-call]
            for number, page in enumerate(document, start=1):
                out = target / f"{pdf.stem}-page{number:04d}.png"
                if not out.exists():
                    page.get_pixmap(matrix=matrix).save(str(out))
                rendered.append(out)
        return rendered
