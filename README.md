# Study Note Synthesizer

Turns your handwritten lecture notes into an exam-ready study document per
chapter. The slides define what is examinable, your notes add the reasoning, and
the textbook is allowed in only where the first two leave a real gap.

`study-synthesizer-spec.md` is the source of truth for behaviour. This file is
how to run it.

## Run it now, with no API key

Mock mode is the default. Every LLM call is replayed from `fixtures/llm/*.json`
and embeddings come from a deterministic local backend, so the whole pipeline
runs offline.

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    cp .env.example .env

    python -m studysynth demo \
        --textbook demo/textbook.pdf --slides demo/slides --notes demo/notes \
        --out demo/out

That runs a full chapter: slides read, notes transcribed, review interrupt,
concepts, gaps, drafting, the textbook gate, synthesis, verification. It writes
`demo/out/runs/<id>/document.md` and a provenance report.

## The textbook gate

Spec section 7. The textbook is a reference work written to be read over a term;
the study document is read the night before an exam. Seven mechanisms stand
between them, and a run where nothing is rejected is a bug report:

| | Mechanism | Rejects |
|---|---|---|
| 7.1 | Gap-triggered querying | Speculative retrieval. There is no code path that queries the textbook without a gap. |
| 7.2 | Chapter scoping | Off-syllabus chapters, filtered in the index before the vector search. |
| 7.3 | Relevance grading | Topical near-misses that share vocabulary but answer nothing. |
| 7.4 | Necessity grading | Correct, relevant prose you already have from the slides. |
| 7.5 | Novelty filter | The textbook restating a slide at ten times the length. |
| 7.6 | New-concept guard | Anything that would make you think a new topic is examinable. |
| 7.7 | Budget enforcement | Everything else, once the chapter's token ceiling is reached. |

Every rejection is logged with the mechanism, a reason code and the score, and
surfaces in the provenance report. From the demo run:

    Textbook tokens admitted: 55 of a 106 budget.
    Passages admitted: 1. Rejected: 5.

    | Mechanism               | Rejected |
    |-------------------------|----------|
    | 7.3 relevance grading   | 1        |
    | 7.4 necessity grading   | 1        |
    | 7.5 novelty filter      | 1        |
    | 7.6 new-concept guard   | 1        |
    | 7.7 budget enforcement  | 1        |

Every threshold lives in `studysynth/config.py`. Changing the textbook budget is
one line in one file.

## Layout

    studysynth/
      config.py       every tunable threshold, one place
      models.py       domain types, no logic
      store.py        SQLite metadata, disk layout, Qdrant collections
      llm.py          Claude client, vision OCR, structured output, mock mode
      embeddings.py   embedding backend behind one interface
      ingest.py       textbook chunking, slide extraction, note preprocessing
      retrieval.py    the seven anti-bloat mechanisms
      graph.py        LangGraph nodes, checkpointer, review interrupt, verify loop
      render.py       Markdown to HTML/PDF, provenance tags, reports
      api.py          FastAPI routes and SSE, thin
      worker.py       background job runner and service wiring
      evaluate.py     offline eval
      prompts/        every prompt, as Markdown, never inlined in Python
    web/              React + Vite + TypeScript + Tailwind, four screens
    tests/

Dependencies run one way and never back up:

    config → models → {store, llm, embeddings} → {ingest, retrieval} → graph → {api, worker}

## Checks

    pytest              # 23 tests, one per anti-bloat mechanism plus a full mock run
    mypy --strict       # clean across 14 source files

## Real mode

Uncomment the Claude and Qwen settings in `.env`. The API key is read when a
call is made, never at import, so nothing about the mock path depends on it.

    docker compose up          # qdrant + api

Payload indexes have no effect in Qdrant's in-memory mode, so chapter scoping
(7.2) should be exercised against the container before you trust it.

## Frontend

    cd web && npm install && npm run dev

Four screens: courses and chapter ranges, chapter run with live progress, the
review interrupt with an editable transcript beside the source image, and the
library with provenance highlighting and the rejection report.

## Known limits

- `MockEmbeddings` is lexical, not semantic. It gives real cosine geometry for
  tests, but the 7.5 similarity threshold must be re-tuned against Qwen before
  those numbers mean anything.
- Textbook chunking splits on Markdown headings. `pypdf` returns plain text, so
  a layout-aware parser is needed for structural chunking to work on a real
  textbook.
- Diagram-only slides need page images beside the deck; without them the vision
  pass is skipped rather than guessed at.
