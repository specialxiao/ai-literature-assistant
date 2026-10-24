from pathlib import Path

import pytest

from techlit_assistant import local_embedding
from techlit_assistant.database import Repository
from techlit_assistant.ingest import ingest_bytes
from techlit_assistant.local_embedding import LocalE5Embedder, token_windows
from techlit_assistant.retrieval import AutoRetriever, BM25Retriever, EmbeddingRetriever


@pytest.mark.parametrize("length", [0, 1, 507, 508, 509, 1024, 1560])
def test_windows_preserve_all_tokens_and_tail(length):
    tokens = list(range(length))
    windows = token_windows(tokens, capacity=508)
    assert all(len(window) <= 508 for window in windows)
    assert {token for window in windows for token in window} == set(tokens)
    if length:
        assert windows[-1][-1] == length - 1
        assert all(windows)
    if len(windows) > 1:
        assert windows[0][-64:] == windows[1][:64]


def test_query_and_document_use_different_e5_prefixes(tmp_path, monkeypatch):
    calls = []
    embedder = LocalE5Embedder(tmp_path)

    def spy(texts, prefix):
        calls.append((list(texts), prefix))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(embedder, "_encode", spy)
    embedder.embed_documents(["English evidence", "中文证据"])
    embedder.embed_query("中文问题")
    assert calls == [
        (["English evidence", "中文证据"], "passage: "),
        (["中文问题"], "query: "),
    ]


def test_local_cache_isolated_by_path_and_preprocessing(tmp_path, monkeypatch):
    first = LocalE5Embedder(tmp_path / "one")
    second = LocalE5Embedder(tmp_path / "two")
    original = first.cache_model_key
    assert original != second.cache_model_key
    assert original.startswith("local:")
    monkeypatch.setattr(local_embedding, "PREPROCESS_VERSION", "changed")
    assert first.cache_model_key != original


def test_local_model_profiles_do_not_share_prefix_or_cache(tmp_path, monkeypatch):
    e5 = LocalE5Embedder(tmp_path)
    minilm = LocalE5Embedder(tmp_path, "minilm")
    assert e5.cache_model_key != minilm.cache_model_key
    assert minilm.max_tokens == 128
    assert minilm.overlap == 32
    calls = []
    monkeypatch.setattr(minilm, "_encode", lambda texts, prefix: calls.append(prefix) or [[1.0]])
    minilm.embed_documents(["document"])
    minilm.embed_query("question")
    assert calls == ["", ""]


def test_high_confidence_bm25_does_not_load_local_model(tmp_path, monkeypatch):
    repository = Repository(tmp_path / "test.db")
    ingest_bytes(repository, "rag.md", b"RAG retrieves relevant evidence.")

    def fail_if_loaded(*args):
        pytest.fail("High-confidence lexical queries must not load torch or E5")

    monkeypatch.setattr(local_embedding, "_load_cpu_model", fail_if_loaded)
    semantic = EmbeddingRetriever(repository, LocalE5Embedder(tmp_path / "missing"))
    retriever = AutoRetriever(BM25Retriever(repository), semantic)
    assert retriever.search("RAG")[0].chunk.filename == "rag.md"
    assert not retriever.last_trace.used_embedding


def test_local_failure_returns_bm25_hits(tmp_path, monkeypatch):
    repository = Repository(tmp_path / "test.db")
    ingest_bytes(repository, "rag.md", b"RAG retrieves relevant evidence.")

    def unavailable(*args):
        raise RuntimeError("Local model missing")

    monkeypatch.setattr(local_embedding, "_load_cpu_model", unavailable)
    retriever = AutoRetriever(
        BM25Retriever(repository),
        EmbeddingRetriever(repository, LocalE5Embedder(tmp_path / "missing")),
    )
    assert retriever.search("relevant evidence paraphrased unknown terms")
    assert retriever.last_trace.strategy == "bm25_fallback"
    assert "Local model missing" in retriever.last_trace.error


def test_embedding_retriever_separates_query_and_document_and_reuses_cache(tmp_path: Path):
    repository = Repository(tmp_path / "test.db")
    ingest_bytes(repository, "rag.md", "检索增强生成的证据。".encode())

    class SpyEmbedder:
        cache_model_key = "local:spy"
        cache_dimensions = 2

        def __init__(self):
            self.documents = []
            self.queries = []

        def embed_documents(self, texts):
            self.documents.extend(texts)
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            self.queries.append(text)
            return [1.0, 0.0]

    embedder = SpyEmbedder()
    retriever = EmbeddingRetriever(repository, embedder)
    for _ in range(2):
        hits = retriever.search("evidence")
        assert hits[0].chunk.page == 1
    assert len(embedder.documents) == 1
    assert embedder.queries == ["evidence", "evidence"]
