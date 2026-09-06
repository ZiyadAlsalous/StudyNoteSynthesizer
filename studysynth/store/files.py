"""Disk layout: the only module that decides where a file goes."""

from __future__ import annotations

from pathlib import Path

from ..config import Settings


class Places:
    """One folder per course, one per lecture, one per run."""

    def __init__(self, settings: Settings) -> None:
        self._paths = settings.paths

    def course(self, course: str) -> Path:
        return self._paths.courses / course

    def textbook(self, course: str) -> Path:
        return self.course(course) / "textbook.pdf"

    def lecture(self, course: str, lecture: str) -> Path:
        return self.course(course) / "lectures" / lecture

    def slides_dir(self, course: str, lecture: str) -> Path:
        target = self.lecture(course, lecture) / "slides"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def notes_dir(self, course: str, lecture: str) -> Path:
        target = self.lecture(course, lecture) / "notes"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def clear(self, folder: Path) -> int:
        """Replacing an upload removes what it replaces."""
        removed = 0
        if folder.is_dir():
            for path in folder.iterdir():
                if path.is_file():
                    path.unlink()
                    removed += 1
        return removed

    def run(self, run_id: str) -> Path:
        return self._paths.runs / run_id

    def artifact(self, run_id: str, name: str) -> Path:
        target = self.run(run_id) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def note_cache(self, content_hash: str) -> Path:
        target = self._paths.root / "cache" / "ocr" / f"{content_hash}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
