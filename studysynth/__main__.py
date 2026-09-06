"""Command line entry point: serve the API, or run the mock demo end to end."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from .config import load
from .ingest import OutlineMissing
from .models import ChapterRange
from .render import provenance_report
from .worker import Worker, build_services


def _demo(argv: argparse.Namespace) -> int:
    """A full chapter run on fixtures, writing real files. No API key needed."""
    settings = load()
    settings.paths.root = Path(argv.out)
    services = build_services(settings.validated())
    jobs = Worker(services)

    services.catalogue.add_course(argv.course, argv.course)
    try:
        chunks = jobs.ingest_textbook(argv.course, Path(argv.textbook), ranges=None)
    except OutlineMissing:
        # Stands in for the UI's manual page-range override, which exists for
        # exactly this case: a textbook PDF with no usable outline.
        import pypdf

        pages = len(pypdf.PdfReader(argv.textbook).pages)
        span = ChapterRange(chapter=argv.chapter, title=argv.chapter, page_start=1,
                            page_end=pages, manual_override=True)
        chunks = jobs.ingest_textbook(argv.course, Path(argv.textbook), ranges=[span])
    print(f"textbook: {chunks} chunks indexed")

    run_id = uuid.uuid4().hex[:12]
    jobs.start(run_id, argv.course, argv.chapter, Path(argv.slides), Path(argv.notes))
    for event in jobs.events(run_id):
        print(f"  {event.node:20s} {event.detail}")
    jobs.resume(run_id)
    for event in jobs.events(run_id):
        print(f"  {event.node:20s} {event.detail}")

    record = services.catalogue.run(run_id)
    print(f"\nrun {run_id}: {record.status} -> {record.document_path}")
    if record.status != "done":
        print(f"error: {record.error}", file=sys.stderr)
        return 1
    state = services.runner.state(run_id)
    print(provenance_report(record.course, record.chapter, state["retrieval"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="studysynth")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ui")
    commands.add_parser("serve")
    demo = commands.add_parser("demo")
    demo.add_argument("--course", default="cs3340")
    demo.add_argument("--chapter", default="induction")
    demo.add_argument("--textbook", required=True)
    demo.add_argument("--slides", required=True)
    demo.add_argument("--notes", required=True)
    demo.add_argument("--out", default="data")

    arguments = parser.parse_args(argv)
    if arguments.command == "demo":
        return _demo(arguments)
    if arguments.command == "ui":
        return _ui()
    from .api import serve

    serve()
    return 0


def _ui() -> int:
    """Streamlit owns its own server, so hand off rather than import it."""
    import subprocess

    page = Path(__file__).parent / "ui.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(page), "--server.headless", "true"]
    )


if __name__ == "__main__":
    sys.exit(main())
