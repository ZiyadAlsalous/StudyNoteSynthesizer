"""Markdown to HTML and PDF, provenance tagging, and the rejection report."""

from __future__ import annotations

import html
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markdown_it import MarkdownIt

from ..config import PdfSettings
from ..models import RetrievalOutcome

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"

TEXTBOOK_TAG = re.compile(r"\[C: pages (\d+)-(\d+)\]")
CHECK_THIS = re.compile(r"\*\*Check this:\*\*")


class RenderError(RuntimeError):
    """Typesetting failed. The Markdown is still the document of record."""


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES),
        undefined=StrictUndefined,
        autoescape=False,
    )


def tag_provenance(markdown: str) -> str:
    """Wrap textbook passages and recorded disagreements so CSS can colour them."""

    def wrap(match: re.Match[str]) -> str:
        return (
            f'<span class="src-c"><span class="ref">[textbook pp. '
            f"{match.group(1)}-{match.group(2)}]</span> "
        )

    tagged = TEXTBOOK_TAG.sub(wrap, markdown)
    lines = []
    for line in tagged.splitlines():
        if '<span class="src-c">' in line:
            line = line + "</span>"
        if CHECK_THIS.search(line):
            line = f'<span class="check">{line}</span>'
        lines.append(line)
    return "\n".join(lines)


def to_html(markdown: str, *, title: str, highlight: bool = True) -> str:
    """The on-screen preview, where `$...$` stays literal because no KaTeX is
    loaded. `to_pdf` is what turns it into typeset mathematics."""
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    return _environment().get_template("document.html").render(
        title=html.escape(title),
        css=(TEMPLATES / "document.css").read_text(encoding="utf-8"),
        body=parser.render(tag_provenance(markdown)),
        highlight=highlight,
    )


def to_pdf(markdown: str, target: Path, settings: PdfSettings) -> Path:
    """Typeset the document, turning its LaTeX into real mathematics.

    Pandoc drives a TeX engine, so `$\\sum_{i=1}^{n}$` reaches the page as a
    summation rather than as the source the Markdown shows. Provenance tags
    print as written; the coloured version of them lives in `to_html`.
    """
    for binary, variable in ((settings.pandoc, "PANDOC"), (settings.engine, "ENGINE")):
        if shutil.which(binary) is None:
            raise RenderError(
                f"'{binary}' is not on PATH. Install it, or point "
                f"STUDYSYNTH_PDF__{variable} at the executable."
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        settings.pandoc,
        "--from=markdown",
        f"--pdf-engine={settings.engine}",
        "--variable", f"geometry:{settings.paper}",
        "--variable", f"geometry:margin={settings.margin}",
        "--output", str(target),
    ]
    if settings.main_font:
        command += ["--variable", f"mainfont={settings.main_font}"]

    try:
        finished = subprocess.run(
            command,
            input=markdown.encode("utf-8"),
            capture_output=True,
            timeout=settings.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RenderError(
            f"{settings.pandoc} did not finish within {settings.timeout_seconds}s"
        ) from error
    if finished.returncode != 0:
        detail = finished.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RenderError(f"{settings.pandoc} failed: {detail[-1] if detail else 'no output'}")
    return target


def provenance_report(course: str, chapter: str, outcome: RetrievalOutcome) -> str:
    counts = Counter(rejection.mechanism for rejection in outcome.rejections)
    return _environment().get_template("provenance.md").render(
        course=course,
        chapter=chapter,
        outcome=outcome,
        by_mechanism=sorted(counts.items()),
    )
