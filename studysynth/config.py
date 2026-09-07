"""Every tunable threshold in the system."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when settings are internally inconsistent."""


class Paths(BaseModel):
    root: Path = Path("data")

    @property
    def catalogue(self) -> Path:
        return self.root / "catalogue.sqlite"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints.sqlite"

    @property
    def courses(self) -> Path:
        return self.root / "courses"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def vectors(self) -> Path:
        return self.root / "qdrant"


class LlmSettings(BaseModel):
    backend: str = "mock"
    model: str = "claude-sonnet-5"
    vision_model: str = "claude-haiku-4-5"
    vision_effort: str = ""
    max_output_tokens: int = 16000
    effort: str = "high"
    grading_effort: str = "low"
    draft_effort: str = "medium"
    max_attempts: int = 3
    fixtures: Path = Path("fixtures/llm")


class EmbeddingSettings(BaseModel):
    backend: str = "mock"
    model: str = "Qwen/Qwen3-Embedding-8B"
    dimensions: int = 1024
    batch_size: int = 16
    query_prefix: str = (
        "Instruct: Given a gap in a student's lecture notes, retrieve the "
        "textbook passage that resolves it\nQuery: "
    )


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    backend: str = "local"
    distance: str = "Cosine"
    payload_indexes: tuple[str, ...] = ("chapter", "section_path", "page_start")


class NoteSettings(BaseModel):
    render_dpi: int = 200
    max_parallel_ocr: int = 12


class ChunkSettings(BaseModel):
    child_tokens: int = 320
    parent_tokens: int = 1800
    min_chunk_chars: int = 120


class RetrievalSettings(BaseModel):
    """Spec section 7: every threshold standing between the textbook and the output."""

    allow_adjacent_chapters: bool = False
    auto_scope_chapters: int = 2
    auto_scope_probe: int = 40
    max_parallel_grading: int = 8
    min_vector_score: float = 0.0
    top_k: int = 20
    min_relevance: float = Field(default=0.55, ge=0.0, le=1.0)
    min_necessity: float = Field(default=0.60, ge=0.0, le=1.0)
    max_similarity_to_draft: float = Field(default=0.82, ge=0.0, le=1.0)
    new_concept_guard: bool = True
    textbook_token_budget: int = 1200
    max_textbook_fraction: float = Field(default=0.15, ge=0.0, le=1.0)


class VerifySettings(BaseModel):
    max_rounds: int = 1
    drop_unverified: bool = True


class PdfSettings(BaseModel):
    """Typesetting the document. Nothing here is specific to a course or a book.

    LaTeX reaches the page as mathematics rather than as `$` and backslashes,
    which is the whole reason the pipeline keeps it as LaTeX from OCR onward.
    """

    pandoc: str = "pandoc"
    engine: str = "xelatex"
    paper: str = "a4paper"
    margin: str = "22mm"
    main_font: str = ""
    timeout_seconds: int = 120


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="STUDYSYNTH_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    paths: Paths = Paths()
    llm: LlmSettings = LlmSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    qdrant: QdrantSettings = QdrantSettings()
    chunks: ChunkSettings = ChunkSettings()
    notes: NoteSettings = NoteSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    verify: VerifySettings = VerifySettings()
    pdf: PdfSettings = PdfSettings()

    prompts_dir: Path = Path(__file__).parent / "prompts"

    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )

    def validated(self) -> Settings:
        if self.chunks.parent_tokens <= self.chunks.child_tokens:
            raise ConfigError("parent_tokens must exceed child_tokens")
        if self.retrieval.min_necessity < self.retrieval.min_relevance:
            raise ConfigError("min_necessity must be >= min_relevance")
        return self


def load() -> Settings:
    return Settings().validated()
