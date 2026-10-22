from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Document:
    id: str
    content_sha256: str
    title: str
    filename: str
    page_count: int
    created_at: datetime | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    document_title: str
    filename: str
    page: int
    chunk_index: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.document_title}, p.{self.page}"


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class IngestionResult:
    document: Document
    chunks_created: int
    created: bool


@dataclass(frozen=True)
class AnswerResult:
    text: str
    sources: list[SearchHit]
    generated: bool
