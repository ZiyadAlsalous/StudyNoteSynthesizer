"""Command line entry point.

    python -m studysynth ui

Streamlit owns its own server, so this hands off rather than importing it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="studysynth")
    parser.add_argument(
        "command", nargs="?", default="ui", choices=["ui"], help="what to run"
    )
    parser.add_argument("--port", type=int, default=8501)
    arguments = parser.parse_args(argv)

    page = Path(__file__).parent / "ui.py"
    return subprocess.call(
        [
            sys.executable, "-m", "streamlit", "run", str(page),
            "--server.port", str(arguments.port),
            "--server.headless", "true",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
