from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .models import Chunk, Document


class Repository:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content_sha256 TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL CHECK (page_count > 0),
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL CHECK (page > 0),
                    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                    text TEXT NOT NULL,
                    UNIQUE(document_id, page, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY(chunk_id, model, dimensions)
                );
                """
            )

    def find_document_by_hash(self, content_sha256: str) -> Document | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE content_sha256 = ?", (content_sha256,)
            ).fetchone()
        return self._document_from_row(row) if row else None

    def add_document(self, document: Document, chunks: Iterable[Chunk]) -> bool:
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("文档至少需要一个非空 chunk")

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE content_sha256 = ?", (document.content_sha256,)
            ).fetchone()
            if existing:
                return False

            created_at = (document.created_at or datetime.now(timezone.utc)).isoformat()
            connection.execute(
                """
                INSERT INTO documents(id, content_sha256, title, filename, page_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.content_sha256,
                    document.title,
                    document.filename,
                    document.page_count,
                    created_at,
                ),
            )
            connection.executemany(
                """
                INSERT INTO chunks(id, document_id, page, chunk_index, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (chunk.id, chunk.document_id, chunk.page, chunk.chunk_index, chunk.text)
                    for chunk in chunk_list
                ],
            )
        return True

    def list_documents(self) -> list[Document]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, filename"
            ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def list_chunks(self) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, d.title AS document_title, d.filename
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY d.filename, c.page, c.chunk_index
                """
            ).fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def save_embeddings(
        self,
        model: str,
        dimensions: int,
        vectors: dict[str, list[float]],
    ) -> None:
        if not vectors:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings(chunk_id, model, dimensions, vector_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (chunk_id, model, dimensions, json.dumps(vector, separators=(",", ":")))
                    for chunk_id, vector in vectors.items()
                ],
            )

    def load_embeddings(self, model: str, dimensions: int) -> dict[str, list[float]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, vector_json
                FROM embeddings
                WHERE model = ? AND dimensions = ?
                """,
                (model, dimensions),
            ).fetchall()
        return {row["chunk_id"]: json.loads(row["vector_json"]) for row in rows}

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM documents")

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            content_sha256=row["content_sha256"],
            title=row["title"],
            filename=row["filename"],
            page_count=row["page_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _chunk_from_row(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            document_title=row["document_title"],
            filename=row["filename"],
            page=row["page"],
            chunk_index=row["chunk_index"],
            text=row["text"],
        )
