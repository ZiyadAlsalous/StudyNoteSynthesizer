# 📚 Study Note Synthesizer — Your Notes, Completed

Your handwritten lecture notes, completed against your professor's slides, with the
course textbook allowed in **only where they leave a real gap**. Everything runs on
your own machine with your own API key.

---

## Overview

Photograph your handwritten notes for a lecture. Point the app at your professor's
slide deck and, optionally, the course textbook. It returns one exam-ready study
document: everything from the slides, your own reasoning folded in beside the point
it explains, and disagreements between the two shown rather than silently resolved.

The hard part is not writing the document. It is **keeping the textbook out of it**.
A textbook is written to be read over a term; a study document is read the night
before an exam. Admit textbook prose freely and the useful part drowns. So seven
separate mechanisms stand between the textbook and the output, every rejection is
logged with a reason and a score, and a run that rejects nothing is treated as a bug.

## ✨ Features

- **Three ranked sources** — the slides are authoritative because the professor sets
  the exam; your notes add intuition and worked reasoning; the textbook is a gap
  filler that never sets scope.
- **Disagreements surfaced, not corrected** — where your notes and the slides conflict
  on a fact, both appear inline under a **Check this:** marker. You need to see what
  you misunderstood.
- **Seven anti-bloat controls** — the product, described below.
- **Human review interrupt** — the graph stops after OCR so you can fix the
  transcription beside the page it came from, because handwriting is the least
  reliable thing the system reads.
- **Crash-resumable runs** — a SQLite checkpointer means a failed run resumes at the
  node that failed instead of re-transcribing everything.
- **The textbook is embedded once** — parsed, chunked and written to an on-disk vector
  index per course. Later runs query that index rather than re-embedding the book.
- **Full run history** — replacing your notes deletes the old images so a run never
  mixes two versions of a page, but every document you ever built stays downloadable.
- **Provenance on every passage** — a toggle colours the document by source, and a
  report lists exactly what the textbook gate rejected and why.
- **Prompts are files, not code** — every prompt is a `.md` you can edit without
  reading Python.
- **Runs offline for development** — a mock backend replays recorded fixtures, so the
  whole pipeline and all 37 tests run with no API key and no cost.

## 🚧 The textbook gate

Spec section 7. Each mechanism rejects, and each rejection is logged.

| | Mechanism | Rejects |
|---|---|---|
| 7.1 | Gap-triggered querying | Speculative retrieval. No code path queries the textbook without a detected gap. |
| 7.2 | Chapter scoping | Off-topic chapters. The chapters matching your notes are found automatically, then filtered inside the index before the vector search. |
| 7.3 | Relevance grading | Topical near-misses that share vocabulary but answer nothing. |
| 7.4 | Necessity grading | Correct, relevant prose you already have from the slides. The largest bloat source. |
| 7.5 | Novelty filter | The textbook restating a slide at ten times the length. |
| 7.6 | New-concept guard | Anything that would make you believe a new topic is examinable. |
| 7.7 | Budget enforcement | Everything left, once the chapter's token ceiling is reached. |

Every threshold lives in one file, `config.py`. Changing the textbook budget is one
line in one place.

## 🏗️ Architecture

```
Slides (PDF/PPTX)          Note photos            Textbook (PDF, optional)
      │                         │                          │
      ▼                         ▼                          ▼
 page extraction          vision OCR              outline → chapters
      │                (cached by content hash)   structural chunking
      │                         │                 parent/child split
      └────────────┬────────────┘                          │
                   ▼                                       ▼
          ══ REVIEW INTERRUPT ══                 Qdrant (on disk, once per course)
        you correct the transcript                         │
                   │                                       │
                   ▼                                       │
          concept extraction                               │
                   │                                       │
                   ▼                                       │
             gap detection ─────── the only thing that ────┘
                   │               may query the textbook
                   ▼                                       │
          drafting from A + B                              ▼
                   │                          ┌── 7.1 gap-triggered
                   │                          │   7.2 chapter scoping
                   ▼                          │   7.3 relevance
          the textbook gate ◀─────────────────┤   7.4 necessity
                   │                          │   7.5 novelty
                   │  admitted passages       │   7.6 new-concept guard
                   ▼  + rejection log         └── 7.7 budget
             synthesis  ⇄  verify (bounded loop)
                   │
                   ▼
      Study document  (Markdown + provenance tags, kept in run history)
```

Eight LangGraph nodes. Slides and notes ingest in parallel, then fan in.
Dependencies run one way and never back up:

```
config → models → {store, llm, embeddings} → {ingest, retrieval} → graph → services → ui
```

## 🛠️ Tech Stack

**Core:** Python 3.12 · Pydantic v2 · `mypy --strict`
**Orchestration:** LangGraph with a SQLite checkpointer and a human-review interrupt
**LLM:** Claude Opus 5 — vision OCR, concept and gap extraction, grading, synthesis
**Embeddings:** Qwen3-Embedding-0.6B, run locally (4B and 8B drop in via config)
**Vector DB:** Qdrant, embedded and on disk, one collection per course
**Storage:** SQLite catalogue · per-course file store · Markdown documents per run
**Interface:** Streamlit — three screens, one file, no build step

## 🚀 Getting Started

```bash
# 1. Clone
git clone https://github.com/<your-username>/studysynth.git
cd studysynth

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[claude,qwen,dev]"

# 3. Configure — copy the example and paste your key
cp .env.example .env
# ANTHROPIC_API_KEY=...                    # the only key needed
# STUDYSYNTH_LLM__BACKEND=claude
# STUDYSYNTH_EMBEDDINGS__BACKEND=qwen      # embeddings run locally, no key

# 4. Run
python -m studysynth ui
```

Open **http://localhost:8501**. Create a course, upload a textbook once, add a
lecture, upload your slides and note photos, and build.

```bash
# Develop with no API key and no cost — replays recorded fixtures
STUDYSYNTH_LLM__BACKEND=mock STUDYSYNTH_EMBEDDINGS__BACKEND=mock python -m studysynth ui

# Checks
pytest              # 37 tests: the seven controls, a full mock run, persistence
mypy --strict       # clean across 14 modules
```

## 📂 Repository Structure

```
studysynth/
├── studysynth/
│   ├── __main__.py          # Entry point — launches the interface
│   ├── config.py            # Every tunable threshold. One place.
│   ├── models.py            # Domain types. No logic.
│   ├── store/               # Persistence
│   │   ├── catalogue.py     #   SQLite: courses, lectures, runs, rejections
│   │   ├── files.py         #   Disk layout — the only module that picks paths
│   │   └── vectors.py       #   Qdrant: one collection per course
│   ├── clients/             # The two external services, each behind an interface
│   │   ├── llm.py           #   Claude: vision OCR, structured output, mock mode
│   │   └── embeddings.py    #   Qwen locally, swappable by config
│   ├── pipeline/            # The work itself
│   │   ├── ingest.py        #   Textbook chunking · slides · notes to images · OCR
│   │   ├── retrieval.py     #   The seven anti-bloat mechanisms
│   │   ├── graph.py         #   LangGraph nodes, checkpointer, review interrupt
│   │   └── render.py        #   Markdown → HTML/PDF, provenance tags
│   ├── services.py          # Everything the interface calls, wired once
│   ├── ui.py                # The whole interface: home · course · lecture
│   └── prompts/             # Every prompt as Markdown. Never inlined in Python.
├── fixtures/llm/            # Recorded responses, so tests run offline and free
├── tests/                   # 37 tests
└── study-synthesizer-spec.md  # Source of truth for behaviour
```

Everything you upload and everything it produces lives under `data/` — the SQLite
catalogue, the vector index, your files, and one folder per run holding its document
and provenance report. `data/` is gitignored, so your notes and your key never reach
GitHub.

## 🗺️ Roadmap

- [ ] Prompt caching on the graders, where the drafted chapter is identical across
      every candidate in a run
- [ ] Tune the vector floor against real Qwen scores and switch it on
- [ ] A layout-aware parser, so structural chunking sees real headings instead of
      the flat text `pypdf` returns
- [ ] Re-tune the novelty threshold against Qwen rather than the lexical mock
- [ ] An offline eval: coverage, bloat rate and citation validity against a
      hand-labelled chapter

## 📫 Contact

Built by Ziyad Salous — ziyadsalous00@gmail.com
