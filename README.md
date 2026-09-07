# 📚 Study Note Synthesizer

**Your handwritten lecture notes, completed against your professor's slides, with the course textbook allowed in only where they leave a real gap.**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Claude](https://img.shields.io/badge/Claude-Sonnet%205-D97757?logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![Claude Haiku](https://img.shields.io/badge/Claude-Haiku%204.5-D97757?logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-checkpointed-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20index-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Qwen3](https://img.shields.io/badge/Qwen3-embeddings-615CED)](https://huggingface.co/Qwen)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2)](https://mypy-lang.org/)
[![ruff](https://img.shields.io/badge/ruff-clean-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)

## Overview

Photograph your handwritten notes for a lecture. Point the app at your professor's slide deck and, optionally, the course textbook. It returns one exam-ready study document, typeset as a PDF: everything from the slides, your own reasoning folded in beside the point it explains, and disagreements between the two shown rather than silently resolved.

The hard part is not writing the document. It is **keeping the textbook out of it**. A textbook is written to be read over a term; a study document is read the night before an exam. Admit textbook prose freely and the useful part drowns.

So seven separate mechanisms stand between the textbook and the output. Every rejection is logged with a reason and a score, and a run that rejects nothing is treated as a bug rather than a success.

Everything runs on your own machine with your own API key. Your notes never leave it except as prompts you can read.

## ✨ Features

- **Three ranked sources.** The slides are authoritative because the professor sets the exam. Your notes add intuition and worked reasoning. The textbook is a gap filler that never sets scope.
- **Disagreements surfaced, not corrected.** Where your notes and the slides conflict on a fact, both appear inline under a **Check this:** marker. You need to see what you misunderstood.
- **Seven anti-bloat controls.** The product, described under Architecture.
- **📝 Human review interrupt.** The graph stops after OCR so you can fix the transcription beside the page it came from, because handwriting is the least reliable thing the system reads.
- **♻️ Crash-resumable runs.** A SQLite checkpointer means a failed run resumes at the node that failed instead of re-transcribing everything.
- **🔢 Typeset mathematics.** LaTeX is preserved from OCR all the way through, then rendered by pandoc and a TeX engine, so the PDF shows real summations and fractions rather than `$` and backslashes.
- **💾 The textbook is embedded once.** Parsed, chunked and written to an on-disk vector index per course. Later runs query that index rather than re-embedding the book.
- **🎨 Provenance on every passage.** A toggle colours the document by source, and every run keeps a report of what the gate rejected and why.
- **📄 Prompts are files, not code.** All nine live in `studysynth/prompts/` as Markdown you can edit without touching Python.
- **🔌 Runs offline for development.** A mock backend replays recorded fixtures, so the whole pipeline and all 54 tests run with no API key and no cost.

## 🏗️ Architecture

```
Slides (PDF/PPTX)          Note photos            Textbook (PDF, optional)
      │                         │                          │
      ▼                         ▼                          ▼
 page extraction          vision OCR              outline to chapters
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
        Study document  (PDF + Markdown, with provenance, kept in run history)
```

Eight LangGraph nodes. Slides and notes ingest in parallel, then fan in. Dependencies run one way and never back up:

```
config → models → {store, clients} → {ingest, retrieval} → graph → library → ui
```

### The textbook gate

Each mechanism rejects, and each rejection is logged with its score.

| | Mechanism | Rejects |
|---|---|---|
| 7.1 | Gap-triggered querying | Speculative retrieval. No code path queries the textbook without a detected gap. |
| 7.2 | Chapter scoping | Off-topic chapters. The chapters matching your notes are found automatically, then filtered inside the index before the vector search. |
| 7.3 | Relevance grading | Topical near-misses that share vocabulary but answer nothing. |
| 7.4 | Necessity grading | Correct, relevant prose you already have from the slides. The largest bloat source. |
| 7.5 | Novelty filter | The textbook restating a slide at ten times the length. |
| 7.6 | New-concept guard | Anything that would make you believe a new topic is examinable. |
| 7.7 | Budget enforcement | Everything left, once the chapter's token ceiling is reached. |

Every threshold lives in `studysynth/config.py`. Changing the textbook budget is one line in one place.

## 🛠️ Tech Stack

**Core:** Python 3.12 · Pydantic v2 · `mypy --strict` · `ruff`

**Orchestration:** LangGraph, with a SQLite checkpointer and a human-review interrupt

**LLMs:** Claude Sonnet 5 (concepts, gaps, drafting, grading, synthesis) · Claude Haiku 4.5 (handwriting OCR)

**Embeddings:** Qwen3-Embedding-0.6B, run locally (4B and 8B drop in via config)

**Vector DB:** Qdrant, embedded and on disk, one collection per course

**Typesetting:** pandoc with a TeX engine, for real mathematics in the PDF

**Storage:** SQLite catalogue · per-course file store · PDF and Markdown per run

**Interface:** Streamlit, three screens, no build step

## 🚀 Getting Started

**1. Clone**

```bash
git clone https://github.com/ZiyadAlsalous/study-note-synthesizer.git
cd study-note-synthesizer
```

**2. Install**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[claude,qwen,dev]"
```

PDF output additionally needs `pandoc` and a TeX engine:

```bash
brew install pandoc && brew install --cask mactex-no-gui   # macOS
```

**3. Configure**

```bash
cp .env.example .env
```

```ini
ANTHROPIC_API_KEY=...                    # the only key needed
STUDYSYNTH_LLM__BACKEND=claude
STUDYSYNTH_LLM__MODEL=claude-sonnet-5
STUDYSYNTH_LLM__VISION_MODEL=claude-haiku-4-5
STUDYSYNTH_EMBEDDINGS__BACKEND=qwen      # embeddings run locally, no key
```

**4. Run**

```bash
python -m studysynth ui
```

Open **http://localhost:8501**. Create a course, upload a textbook once, add a lecture, upload your slides and your notes PDF, and build.

Only one copy can run at a time, because the index is an embedded single-process store. If the app reports a busy index, close the other copy or run `pkill -f 'studysynth ui'`. Nothing saved is lost.

**Development, with no API key and no cost**

```bash
STUDYSYNTH_LLM__BACKEND=mock STUDYSYNTH_EMBEDDINGS__BACKEND=mock python -m studysynth ui

pytest         # 54 tests: the seven controls, a full mock run, persistence
mypy           # strict, 19 modules
ruff check .   # lint
```

## 📂 Repository Structure

```
studysynth/
├── studysynth/
│   ├── __main__.py          # Entry point, launches the interface
│   ├── config.py            # Every tunable threshold. One place.
│   ├── models.py            # Domain types. No logic.
│   ├── library.py           # Everything the interface calls, wired once
│   ├── ui.py                # The whole interface: home, course, lecture
│   ├── store/               # Persistence
│   │   ├── catalogue.py     #   SQLite: courses, lectures, runs, rejections
│   │   ├── files.py         #   Disk layout, the only module that picks paths
│   │   └── vectors.py       #   Qdrant: one collection per course
│   ├── clients/             # The two external services, each behind an interface
│   │   ├── llm.py           #   Claude: vision OCR, structured output, mock mode
│   │   └── embeddings.py    #   Qwen locally, swappable by config
│   ├── pipeline/            # The work itself
│   │   ├── ingest.py        #   Textbook chunking, slides, notes to images, OCR
│   │   ├── retrieval.py     #   The seven anti-bloat mechanisms
│   │   ├── graph.py         #   LangGraph nodes, checkpointer, review interrupt
│   │   └── render.py        #   Markdown to HTML and PDF, provenance tags
│   ├── prompts/             # All nine prompts as Markdown. Never inlined.
│   └── templates/           # Document HTML, print CSS, provenance report
├── docs/
│   └── specification.md     # Source of truth for behaviour
├── fixtures/llm/            # Recorded responses, so tests run offline and free
└── tests/                   # 54 tests
```

Everything you upload and everything the app produces lives under `data/`: the SQLite catalogue, the vector index, your files, and one folder per run holding its PDF, its Markdown and its provenance report. `data/` and `.env` are both gitignored, so your notes and your key never reach GitHub.

## 🗺️ Roadmap

- [ ] Prompt caching on the graders, where the drafted chapter is identical across every candidate in a run
- [ ] Per-job token accounting, so cost per run is measured rather than estimated
- [ ] Tune the vector floor against real Qwen scores and switch it on
- [ ] A layout-aware parser, so structural chunking sees real headings instead of the flat text `pypdf` returns
- [ ] An offline eval: coverage, bloat rate and citation validity against a hand-labelled chapter

## 📫 Contact

Built by **Ziyad Salous**

📧 [ziyadsalous00@gmail.com](mailto:ziyadsalous00@gmail.com)
