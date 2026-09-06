"""Markdown to HTML and PDF, provenance tagging, and the rejection report.

Provenance is the point of this module. The student must be able to see at a
glance which sentences came from the textbook, because those are the ones the
professor never said.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from markdown_it import MarkdownIt

from .models import Rejection, RetrievalOutcome, Source

TEXTBOOK_TAG = re.compile(r"\[C: pages (\d+)-(\d+)\]")
CHECK_THIS = re.compile(r"\*\*Check this:\*\*")


class RenderError(RuntimeError):
    pass


PAGE_CSS = """
@page { size: A4; margin: 22mm 18mm; @bottom-center { content: counter(page); } }
body { font: 11pt/1.5 Georgia, serif; color: #1a1a1a; }
h1 { font-size: 20pt; border-bottom: 2px solid #1a1a1a; padding-bottom: 4pt; }
h2 { font-size: 14pt; margin-top: 18pt; }
h3 { font-size: 12pt; }
blockquote { border-left: 3px solid #999; margin-left: 0; padding-left: 12pt; color: #444; }
.src-c { background: #fff6e0; border-left: 3px solid #d99b00; padding: 2pt 6pt; display: inline-block; }
.src-c .ref { font-size: 8pt; color: #8a6400; letter-spacing: .04em; }
.check { background: #ffecec; border-left: 3px solid #c0392b; padding: 2pt 6pt; }
.plain .src-c, .plain .check { background: none; border-left: none; padding: 0; }
table { border-collapse: collapse; } td, th { border: 1px solid #bbb; padding: 3pt 6pt; }
"""

DOCUMENT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>{{ title }}</title>
<style>{{ css }}</style></head>
<body class="{{ 'plain' if not highlight else '' }}">{{ body }}</body></html>
"""

REPORT_MD = """# Provenance report — {{ course }} / {{ chapter }}

Textbook tokens admitted: **{{ outcome.tokens_admitted }}** of a **{{ outcome.budget }}** budget.
Passages admitted: **{{ outcome.admitted | length }}**. Rejected: **{{ outcome.rejections | length }}**.

## Rejections by mechanism

| Mechanism | Rejected |
|---|---|
{% for mechanism, count in by_mechanism -%}
| {{ mechanism }} | {{ count }} |
{% endfor %}
## Admitted passages

{% for passage in outcome.admitted -%}
- pages {{ passage.page_start }}-{{ passage.page_end }}, necessity {{ '%.2f' % passage.necessity }}, {{ passage.token_estimate }} tokens
{% endfor %}
{% if not outcome.rejections %}
> Nothing was rejected. Spec 7.8: that is a bug report, not a success.
{% endif %}
"""


def _environment() -> Environment:
    return Environment(undefined=StrictUndefined, autoescape=False)


def tag_provenance(markdown: str) -> str:
    """Wrap textbook passages and recorded disagreements so CSS can color them.

    Runs before the Markdown parser, on the source, because the tags are written
    into the text by the synthesis step and would otherwise be invisible.
    """

    def wrap(match: re.Match[str]) -> str:
        return (
            f'<span class="src-c"><span class="ref">[textbook pp. '
            f"{match.group(1)}-{match.group(2)}]</span> "
        )

    tagged = TEXTBOOK_TAG.sub(wrap, markdown)
    # Close each opened span at the end of its paragraph.
    lines = []
    for line in tagged.splitlines():
        if '<span class="src-c">' in line:
            line = line + "</span>"
        if CHECK_THIS.search(line):
            line = f'<span class="check">{line}</span>'
        lines.append(line)
    return "\n".join(lines)


def to_html(markdown: str, *, title: str, highlight: bool = True) -> str:
    """LaTeX survives: `$...$` and `$$...$$` pass through untouched for KaTeX."""
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = parser.render(tag_provenance(markdown))
    template = _environment().from_string(DOCUMENT_HTML)
    return template.render(title=html.escape(title), css=PAGE_CSS, body=body, highlight=highlight)


def to_pdf(markdown: str, destination: Path, *, title: str, highlight: bool = True) -> Path:
    try:
        from weasyprint import HTML
    except ImportError as error:
        raise RenderError(
            "PDF output needs weasyprint: pip install 'studysynth[render]'"
        ) from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=to_html(markdown, title=title, highlight=highlight)).write_pdf(str(destination))
    return destination


def provenance_report(course: str, chapter: str, outcome: RetrievalOutcome) -> str:
    counts = Counter(rejection.mechanism for rejection in outcome.rejections)
    template = _environment().from_string(REPORT_MD)
    return template.render(
        course=course,
        chapter=chapter,
        outcome=outcome,
        by_mechanism=sorted(counts.items()),
    )


def source_counts(markdown: str) -> dict[Source, int]:
    """Rough share of the document by source, for the bloat metric."""
    textbook_chars = sum(
        len(block) for block in re.findall(r"\[C: pages \d+-\d+\](.*?)(?:\n\n|$)", markdown, re.S)
    )
    return {
        Source.TEXTBOOK: textbook_chars,
        Source.SLIDES: max(0, len(markdown) - textbook_chars),
    }


def rejection_table(rejections: list[Rejection]) -> str:
    rows = ["| candidate | mechanism | reason | score | detail |", "|---|---|---|---|---|"]
    for item in rejections:
        score = "" if item.score is None else f"{item.score:.2f}"
        rows.append(
            f"| {item.candidate_id} | {item.mechanism} | {item.reason.value} | {score} | {item.detail} |"
        )
    return "\n".join(rows)
