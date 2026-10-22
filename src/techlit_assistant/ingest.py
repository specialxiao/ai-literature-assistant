from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .database import Repository
from .models import Chunk, Document, IngestionResult


class IngestionError(ValueError):
    """A user-facing document ingestion error."""


@dataclass(frozen=True)
class ExtractedDocument:
    title: str
    pages: list[str]


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_page_text(text: str, max_chars: int = 1000, overlap_chars: int = 150) -> list[str]:
    if max_chars < 100:
        raise ValueError("max_chars 不能小于 100")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars 必须大于等于 0 且小于 max_chars")

    text = normalize_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            minimum_break = start + max_chars // 2
            candidates = [
                text.rfind("\n\n", minimum_break, hard_end),
                text.rfind("。", minimum_break, hard_end),
                text.rfind(". ", minimum_break, hard_end),
                text.rfind("\n", minimum_break, hard_end),
                text.rfind(" ", minimum_break, hard_end),
            ]
            best_break = max(candidates)
            if best_break >= minimum_break:
                end = best_break + (2 if text[best_break : best_break + 2] in {"\n\n", ". "} else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = end - overlap_chars
        start = max(next_start, start + 1)

    return chunks


def extract_document(filename: str, content: bytes) -> ExtractedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(filename, content)
    if suffix in {".md", ".txt"}:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("gb18030")
            except UnicodeDecodeError as exc:
                raise IngestionError("文本文件必须使用 UTF-8 或 GB18030 编码") from exc
        text = normalize_text(text)
        if not text:
            raise IngestionError("文件中没有可用文本")
        return ExtractedDocument(title=Path(filename).stem, pages=[text])
    raise IngestionError("仅支持 PDF、Markdown 和 TXT 文件")


def _extract_pdf(filename: str, content: bytes) -> ExtractedDocument:
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise IngestionError("PDF 已加密，请先移除密码后再上传")
        pages = [normalize_text(page.extract_text() or "") for page in reader.pages]
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"无法解析 PDF：{exc}") from exc

    if not pages:
        raise IngestionError("PDF 没有页面")
    if not any(pages):
        raise IngestionError("没有抽取到文本；扫描版 PDF 暂不支持，请先做 OCR")

    metadata_title = ""
    if reader.metadata:
        metadata_title = str(reader.metadata.title or "").strip()
    return ExtractedDocument(title=metadata_title or Path(filename).stem, pages=pages)


def ingest_bytes(
    repository: Repository,
    filename: str,
    content: bytes,
    max_chars: int = 1000,
    overlap_chars: int = 150,
) -> IngestionResult:
    if not content:
        raise IngestionError("文件为空")

    content_hash = hashlib.sha256(content).hexdigest()
    existing = repository.find_document_by_hash(content_hash)
    if existing:
        return IngestionResult(document=existing, chunks_created=0, created=False)

    extracted = extract_document(filename, content)
    document_id = content_hash[:20]
    document = Document(
        id=document_id,
        content_sha256=content_hash,
        title=extracted.title,
        filename=Path(filename).name,
        page_count=len(extracted.pages),
    )

    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(extracted.pages, start=1):
        for index, chunk_text in enumerate(
            split_page_text(page_text, max_chars=max_chars, overlap_chars=overlap_chars)
        ):
            chunks.append(
                Chunk(
                    id=f"{document_id}:{page_number}:{index}",
                    document_id=document_id,
                    document_title=document.title,
                    filename=document.filename,
                    page=page_number,
                    chunk_index=index,
                    text=chunk_text,
                )
            )

    if not chunks:
        raise IngestionError("文件没有生成任何可检索文本块")
    created = repository.add_document(document, chunks)
    return IngestionResult(
        document=document,
        chunks_created=len(chunks) if created else 0,
        created=created,
    )
