from pathlib import Path

import pytest

from techlit_assistant.database import Repository
from techlit_assistant.ingest import IngestionError, ingest_bytes, split_page_text


def test_split_page_text_respects_size_and_overlap():
    text = "段落一。" * 120 + "\n\n" + "Paragraph two. " * 80
    chunks = split_page_text(text, max_chars=240, overlap_chars=40)

    assert len(chunks) > 2
    assert all(0 < len(chunk) <= 240 for chunk in chunks)
    assert chunks[0][-20:] in chunks[1]


def test_duplicate_content_is_not_inserted(tmp_path: Path):
    repository = Repository(tmp_path / "test.db")
    content = "向量检索使用余弦相似度。".encode()

    first = ingest_bytes(repository, "one.md", content)
    second = ingest_bytes(repository, "renamed.md", content)

    assert first.created is True
    assert second.created is False
    assert len(repository.list_documents()) == 1
    assert len(repository.list_chunks()) == 1


def test_empty_file_has_clear_error(tmp_path: Path):
    repository = Repository(tmp_path / "test.db")
    with pytest.raises(IngestionError, match="文件为空"):
        ingest_bytes(repository, "empty.txt", b"")
