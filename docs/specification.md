# Study Note Synthesizer: Specification

Every threshold in section 7 lives in `studysynth/config.py` and is read from
there, never inlined at a call site. Change a number here first, then in config.

## 1. Purpose

A student photographs their handwritten notes for a lecture. The professor's
slides for that lecture exist as PDF or PPTX. The course textbook exists as a
PDF. The system produces one exam-ready study document per chapter that:

1. contains everything on the slides,
2. folds in the student's own notes as intuition and worked reasoning,
3. surfaces disagreements between the two rather than silently resolving them,
4. and reaches into the textbook only where the first two sources leave a real
   gap.

The output is what the student revises from. It is the only artifact that
matters.

## 2. Authority model

Three sources, strictly ranked.

| Source | Role | Authority |
|---|---|---|
| A — professor's slides | Defines scope and notation | Authoritative on all facts. The professor sets the exam. |
| B — student's handwritten notes | Adds intuition, worked steps, caveats | Subordinate. May contain errors. Never silently corrected. |
| C — course textbook | Gap filler only | Admitted only through Section 7. Never sets scope. |

Where A and B disagree on a fact, both appear inline under a **Check this:**
marker showing what the slides say and what the student wrote.

Source C carries a visible provenance tag on every admitted passage. The student
must always be able to see which sentences came from the textbook.

## 3. Units of work

- **Course** — owns one textbook, one embedding collection, many chapters.
- **Chapter** — the unit a run produces a document for. Maps to a textbook page
  range, from the PDF outline where present, manually overridden in the UI where
  the outline is missing or wrong.
- **Lecture** — one slide deck plus one set of note images. A chapter has one or
  more lectures.

A run takes one chapter and produces one document.

## 4. Ingestion

**Textbook.** Parse the PDF outline for chapter boundaries. Chunk
structurally: split on headings first, then size. Never split mid-table,
mid-formula, or mid-code-block. Parent-child: embed child chunks of 250-400
tokens for retrieval precision, return the parent section of 1500-2000 tokens
to the model for context. Every chunk carries `chapter`, `section_path`,
`page_start`, `page_end`, `parent_id`, `chunk_type`.

**Slides.** Extract per page to Markdown with a `## Page N` marker preserved for
citation. pypdf for PDF, python-pptx for PPTX including speaker notes. Pages that
are diagram-only get a vision pass describing what the diagram shows, appended as
text to that page's chunk.

**Notes.** One PDF of any length: a GoodNotes export or a scan. Each page is
rendered to an image for the vision pass, cached so a re-run never re-renders.
There is no page limit; a lecture may be one page or fifty. Deskew, contrast-normalize, then a vision OCR pass
to Markdown. Cache by content hash so a re-run never re-transcribes an unchanged
page. The OCR transcript is editable by the student at the review interrupt
(Section 8) because OCR on handwriting is the least reliable step in the system.

## 5. Concept and gap extraction

After slides and notes are ingested, one pass extracts a `Concept` per distinct
idea on the slides: name, the slide pages it appears on, the definition as the
slides give it, and whether the student's notes cover it.

A `Gap` is a Concept the slides raise but do not adequately resolve. Three kinds:

- `undefined` — the slides name a term and never define it.
- `unsupported` — the slides state a result with no derivation, proof, or worked
  example, and the notes do not supply one.
- `student_flagged` — the student marked the concept as not understood.

Gaps are the only thing that may trigger a textbook query. This is mechanism 7.1.

## 6. Generation

Multi-stage, never one giant prompt.

1. **Per-concept drafting** — each concept drafted from A and B, with page refs.
2. **Chapter synthesis** — order concepts logically, dedupe, write connective
   prose, produce the "Topics I have no notes on" tail section.
3. **Verify** — a bounded loop, at most `verify.max_rounds`, checking that every
   claim traces to a source and every page citation resolves to a real page.
   Unverified claims are dropped or marked, never silently kept.

All mathematics stays LaTeX: `$...$` inline, `$$...$$` display.

## 7. The anti-bloat controls

**This section is the product.** The textbook is a reference work written to be
read linearly over a term. The study document is read the night before an exam.
Admitting textbook prose freely destroys the document. Seven mechanisms stand
between the textbook and the output. Each one rejects, each rejection is logged
with a reason, and no mechanism may be weakened to make a phase pass.

### 7.1 Gap-triggered querying

The textbook is never queried speculatively. A query is issued only when a `Gap`
from Section 5 exists, and the query text is generated from that gap's concept,
not from free text typed by the student.

*Rejects:* the entire class of "retrieve some relevant context and hope."
*Config:* none. This is structural.
*Log:* every query records the gap id that caused it.

### 7.2 Chapter scoping

Every query filters to a chapter before the vector search runs, using a payload
index declared at collection creation. Filtering after the search is a bug, not a
slower equivalent.

Which chapter is normally decided by the system, not the student. Nobody should
have to know that their induction lecture maps to chapter 4. One unscoped probe
runs the gap queries across the whole book, the chapters carrying the strongest
hits are taken, and the real per-gap queries are scoped to those. A chapter may
still be pinned by hand when the automatic choice is wrong.

*Rejects:* material from chapters the lecture has not reached, and from chapters
already covered, both of which read as off-syllabus to the student.
*Config:* `retrieval.allow_adjacent_chapters` (default `false`),
`retrieval.auto_scope_chapters` (default 2), `retrieval.auto_scope_probe`
(default 40).
*Log:* rejected candidates record `out_of_chapter`.

### 7.3 Relevance grading

Retrieved candidates are scored against the gap. Anything below
`retrieval.min_relevance` is dropped before any model sees it.

*Rejects:* topical near-misses that share vocabulary with the gap but do not
address it.
*Config:* `retrieval.top_k` (default 20 retrieved), `retrieval.min_relevance`
(default 0.55).
*Log:* `below_relevance` with the score.

### 7.4 Necessity grading

A surviving candidate is asked a different question from relevance: *is this
needed?* A passage can be perfectly relevant and still unnecessary because the
slides and notes already resolve the gap. Necessity is graded against the drafted
concept, not against the gap in isolation.

*Rejects:* correct, on-topic textbook prose that adds nothing the student does
not already have. This is the single largest source of bloat and the mechanism
most likely to be quietly dropped during implementation.
*Config:* `retrieval.min_necessity` (default 0.6).
*Log:* `not_necessary`.

### 7.5 Novelty filter

Each surviving passage is embedded and compared against the already-drafted
content for that chapter. Anything above `retrieval.max_similarity_to_draft` is
dropped as a restatement.

*Rejects:* the textbook saying in 200 words what the slide said in 20.
*Config:* `retrieval.max_similarity_to_draft` (default 0.82).
*Log:* `duplicate_of` with the id of the content it duplicates.

### 7.6 New-concept guard

A passage that introduces a concept, term, or notation absent from the slides is
rejected outright, even when relevant, necessary, and novel. The slides define
the examinable surface. The textbook may explain what is on the slides; it may
never add to it.

Exception: a term used purely instrumentally inside an explanation of a slide
concept is admitted if it does not become a heading or a defined term. The guard
checks headings and definitions, not incidental vocabulary.

*Rejects:* scope creep, which is the failure mode that makes a study document
untrustworthy rather than merely long.
*Config:* `retrieval.new_concept_guard` (default `true`).
*Log:* `introduces_concept` with the offending term.

### 7.7 Budget enforcement

A hard ceiling on admitted textbook content per chapter, enforced as the binding
constraint after all other mechanisms pass. Measured in tokens, and also as a
fraction of the total document so that a thin chapter cannot become mostly
textbook.

When the budget is exhausted, remaining candidates are rejected regardless of
score, highest-necessity first having already been admitted.

*Config:* `retrieval.textbook_token_budget` (default 1200 tokens per chapter),
`retrieval.max_textbook_fraction` (default 0.15 of the document).
*Log:* `over_budget`, with the budget state at the time of rejection.

### 7.8 The rejection log

Every rejection from 7.2 through 7.7 is written to the run record with the
candidate id, the mechanism, the reason code, and the numeric score involved.
The provenance report (Section 6 of the render output) shows counts per
mechanism. A run where nothing is ever rejected is a bug report, not a success.

## 8. Human review interrupt

The graph interrupts once, after OCR and before drafting. The student sees the
transcribed notes, edits them, and resumes. The interrupt is a real LangGraph
interrupt with a SQLite checkpointer, so a run survives process death and resumes
where it stopped rather than re-transcribing.

## 9. Frontend — four screens

1. **Courses** — create a course, upload a textbook, watch ingestion progress,
   review and correct detected chapter page ranges.
2. **Chapter run** — pick a chapter, upload slides and note images, start a run,
   watch node-by-node progress over SSE.
3. **Review** — the interrupt. Editable OCR transcript beside the source image,
   then Resume.
4. **Library** — finished documents, provenance highlighting toggle that colors
   each passage by source A/B/C, and the rejection report per run.

## 10. Evaluation

Not built yet. When it is, run from the command line against a labeled chapter:

- **Coverage** — fraction of slide concepts present in the document.
- **Bloat rate** — admitted textbook tokens over total document tokens, against
  the Section 7.7 ceiling.
- **Faithfulness** — claims traceable to a source.
- **Citation validity** — fraction of page citations that resolve to a page
  actually containing the cited content.
- **Context precision / recall** — over the gap-to-passage pairs.
