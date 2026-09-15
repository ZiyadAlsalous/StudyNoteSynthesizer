"""The entry point: `python -m studysynth ui` must reach Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from studysynth import __main__


def _captured(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def record(command: list[str], **_: Any) -> int:
        calls.append(command)
        return 0

    monkeypatch.setattr(__main__.subprocess, "call", record)
    return calls


def test_default_command_runs_the_ui_on_the_default_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _captured(monkeypatch)

    assert __main__.main([]) == 0

    command = calls[0]
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert Path(command[4]).name == "ui.py"
    assert Path(command[4]).exists()
    assert "8501" in command


def test_port_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _captured(monkeypatch)

    assert __main__.main(["ui", "--port", "9000"]) == 0

    assert "9000" in calls[0]


def test_an_unknown_command_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _captured(monkeypatch)

    with pytest.raises(SystemExit):
        __main__.main(["serve"])

    assert not calls
