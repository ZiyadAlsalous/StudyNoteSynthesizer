"""Claude client and its mock twin, behind one interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from base64 import b64encode
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import Settings

Model = TypeVar("Model", bound=BaseModel)


class LlmError(RuntimeError):
    """Raised when a completion cannot be produced or parsed."""


class MissingCredentials(LlmError):
    """No API key anywhere: not in .env, not exported, no CLI profile."""


class FixtureMissing(LlmError):
    """The mock backend was asked for a prompt it has no recorded answer for."""


class PromptMissing(LlmError):
    pass


class PromptLibrary:
    """Loads `prompts/<name>.md` and fills `{placeholders}`."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def render(self, name: str, variables: dict[str, Any] | None = None) -> str:
        path = self._directory / f"{name}.md"
        if not path.exists():
            raise PromptMissing(f"No prompt template at {path}")
        template = path.read_text(encoding="utf-8")
        for key, value in (variables or {}).items():
            template = template.replace("{" + key + "}", str(value))
        return template


class LlmClient(ABC):
    """The two calls the pipeline makes: free text, and a validated object."""

    @abstractmethod
    def complete(self, prompt: str, *, job: str, effort: str | None = None) -> str: ...

    @abstractmethod
    def structured(
        self, prompt: str, schema: type[Model], *, job: str, key: str = "",
        effort: str | None = None,
    ) -> Model: ...

    @abstractmethod
    def vision(self, prompt: str, image: Path, *, job: str) -> str: ...


class MockLlm(LlmClient):
    """Replays `fixtures/llm/<job>.json`."""

    def __init__(self, fixtures: Path) -> None:
        self._fixtures = fixtures
        self._cache: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []

    def _fixture(self, job: str) -> dict[str, Any]:
        if job not in self._cache:
            path = self._fixtures / f"{job}.json"
            if not path.exists():
                raise FixtureMissing(
                    f"Mock mode has no fixture for job '{job}' (looked in {path})"
                )
            self._cache[job] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[job]

    def _lookup(self, job: str, key: str) -> Any:
        fixture = self._fixture(job)
        by_key = fixture.get("by_key", {})
        if key and key in by_key:
            return by_key[key]
        if "default" not in fixture:
            raise FixtureMissing(f"Fixture '{job}' has no entry for key '{key}' and no default")
        return fixture["default"]

    def complete(self, prompt: str, *, job: str, effort: str | None = None) -> str:
        self.calls.append((job, prompt[:80]))
        value = self._lookup(job, "")
        return value if isinstance(value, str) else json.dumps(value)

    def structured(
        self, prompt: str, schema: type[Model], *, job: str, key: str = "",
        effort: str | None = None,
    ) -> Model:
        self.calls.append((job, key))
        payload = self._lookup(job, key)
        try:
            return schema.model_validate(payload)
        except ValidationError as error:
            raise LlmError(
                f"Fixture '{job}' key '{key}' does not match {schema.__name__}"
            ) from error

    def vision(self, prompt: str, image: Path, *, job: str) -> str:
        self.calls.append((job, image.name))
        value = self._lookup(job, image.name)
        return value if isinstance(value, str) else json.dumps(value)


class ClaudeLlm(LlmClient):
    """Anthropic API. The key is read at call time, never at import."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None

    def _anthropic(self) -> Any:
        """Built on first call, never at import."""
        if self._client is None:
            import anthropic

            key = self._settings.anthropic_api_key
            try:
                self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
            except Exception as error:
                raise MissingCredentials(
                    "No Anthropic credentials found. Put ANTHROPIC_API_KEY in .env, "
                    "export it, or run `ant auth login`."
                ) from error
        return self._client

    def _message(self, content: list[dict[str, Any]], model: str, effort: str | None) -> str:
        resolved = self._settings.llm.effort if effort is None else effort
        options: dict[str, Any] = {"output_config": {"effort": resolved}} if resolved else {}
        response = self._anthropic().messages.create(
            model=model,
            max_tokens=self._settings.llm.max_output_tokens,
            messages=[{"role": "user", "content": content}],
            **options,
        )
        if response.stop_reason == "max_tokens":
            raise LlmError(
                f"Model hit max_tokens ({self._settings.llm.max_output_tokens}) and the "
                "response is truncated. Raise STUDYSYNTH_LLM__MAX_OUTPUT_TOKENS and re-run; "
                "the run resumes from its last checkpoint."
            )
        if response.stop_reason == "refusal":
            raise LlmError(f"Model declined this request: {response.stop_details}")
        parts = [block.text for block in response.content if block.type == "text"]
        if not parts:
            raise LlmError("Claude returned no text content")
        return "".join(parts)

    def complete(self, prompt: str, *, job: str, effort: str | None = None) -> str:
        return self._message(
            [{"type": "text", "text": prompt}], self._settings.llm.model, effort
        )

    def structured(
        self, prompt: str, schema: type[Model], *, job: str, key: str = "",
        effort: str | None = None,
    ) -> Model:
        instruction = (
            f"{prompt}\n\nReturn only JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        attempts = max(1, self._settings.llm.max_attempts)
        last: ValidationError | None = None
        for attempt in range(attempts):
            text = instruction if attempt == 0 else (
                f"{instruction}\n\nYour previous reply did not parse as JSON for this "
                "schema. Return the JSON object alone, with no prose and no code fence."
            )
            raw = self._message(
                [{"type": "text", "text": text}], self._settings.llm.model, effort
            )
            try:
                return schema.model_validate_json(_strip_fence(raw))
            except ValidationError as error:
                last = error
        raise LlmError(
            f"Job '{job}' returned JSON that is not a {schema.__name__} "
            f"after {attempts} attempts"
        ) from last

    def vision(self, prompt: str, image: Path, *, job: str) -> str:
        suffix = image.suffix.lstrip(".").lower()
        media = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media,
                    "data": b64encode(image.read_bytes()).decode("ascii"),
                },
            },
            {"type": "text", "text": prompt},
        ]
        return self._message(
            content, self._settings.llm.vision_model, self._settings.llm.vision_effort
        )


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
    return body.rsplit("```", 1)[0].strip()


def build(settings: Settings) -> LlmClient:
    if settings.llm.backend == "mock":
        return MockLlm(settings.llm.fixtures)
    if settings.llm.backend == "claude":
        return ClaudeLlm(settings)
    raise LlmError(f"Unknown llm backend '{settings.llm.backend}'")
