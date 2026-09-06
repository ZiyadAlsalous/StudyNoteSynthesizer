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
    vision_model: str = "claude-sonnet-5"
    max_output_tokens: int = 8000
    # Sampling parameters were removed on Opus 5; effort replaces them.
    effort: str = "high"
    # Grading is a small judgement returning one number, so it does not need
    # the depth that drafting and synthesis do.
    grading_effort: str = "low"
    max_attempts: int = 3
    fixtures: Path = Path("fixtures/llm")


class EmbeddingSettings(BaseModel):
    backend: str = "mock"
    model: str = "Qwen/Qwen3-Embedding-8B"
    dimensions: int = 1024
    batch_size: int = 16
    # Instruction prefix goes on queries only, never on stored documents.
    query_prefix: str = (
        "Instruct: Given a gap in a student's lecture notes, retrieve the "
        "textbook passage that resolves it\nQuery: "
    )


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    # local: an on-disk collection, so a textbook is embedded once and reused.
    backend: str = "local"
    distance: str = "Cosine"
    # Declared at collection creation so chapter scoping (spec 7.2) filters before the vector s.
    payload_indexes: tuple[str, ...] = ("chapter", "section_path", "page_start")


class NoteSettings(BaseModel):
    # Handwriting needs the detail; 200 dpi is legible without huge payloads.
    render_dpi: int = 200
    max_parallel_ocr: int = 6


class ChunkSettings(BaseModel):
    child_tokens: int = 320
    child_overlap: int = 48
    parent_tokens: int = 1800
    min_chunk_chars: int = 120


class RetrievalSettings(BaseModel):
    """Spec section 7: every threshold standing between the textbook and the output."""

    # 7.2 chapter scoping
    allow_adjacent_chapters: bool = False
    # With no chapter chosen, the gaps themselves say which chapters to read.
    auto_scope_chapters: int = 2
    auto_scope_probe: int = 40
    # Graders are independent per candidate, so they run concurrently.
    max_parallel_grading: int = 8
    # A cheap cosine floor before any model sees a candidate. Off by default:
    # it steals work from 7.3 and 7.4, and a safe value depends on the score
    # distribution of whichever embedding model is configured. Measure first.
    min_vector_score: float = 0.0
    # 7.3 relevance grading
    top_k: int = 20
    min_relevance: float = Field(default=0.55, ge=0.0, le=1.0)
    # 7.4 necessity grading
    min_necessity: float = Field(default=0.60, ge=0.0, le=1.0)
    # 7.5 novelty filter
    max_similarity_to_draft: float = Field(default=0.82, ge=0.0, le=1.0)
    # 7.6 new-concept guard
    new_concept_guard: bool = True
    # 7.7 budget enforcement
    textbook_token_budget: int = 1200
    max_textbook_fraction: float = Field(default=0.15, ge=0.0, le=1.0)


class VerifySettings(BaseModel):
    max_rounds: int = 2
    drop_unverified: bool = True


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


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
    server: ServerSettings = ServerSettings()

    prompts_dir: Path = Path(__file__).parent / "prompts"

    # Read here rather than left to the SDK's own environment lookup: settings loaded from a .e.
    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )

    def validated(self) -> "Settings":
        if self.chunks.parent_tokens <= self.chunks.child_tokens:
            raise ConfigError("parent_tokens must exceed child_tokens")
        if self.retrieval.min_necessity < self.retrieval.min_relevance:
            # Necessity is the stricter question; a looser bar would make 7.4 a no-op behind 7..
            raise ConfigError("min_necessity must be >= min_relevance")
        return self


def load() -> Settings:
    return Settings().validated()
