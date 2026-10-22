from pathlib import Path
from types import SimpleNamespace

from techlit_assistant.database import Repository
from techlit_assistant.ingest import ingest_bytes
from techlit_assistant.models import Chunk, SearchHit
from techlit_assistant.retrieval import (
    AutoRetriever,
    BM25Retriever,
    CompatibleEmbedder,
    LexicalDiagnostics,
    TfidfRetriever,
    reciprocal_rank_fusion,
    tokenize_mixed,
)


def test_tfidf_retrieves_chinese_and_english(tmp_path: Path):
    repository = Repository(tmp_path / "test.db")
    ingest_bytes(repository, "rag.md", "检索增强生成需要召回相关证据。".encode())
    ingest_bytes(
        repository,
        "attention.md",
        b"Scaled dot-product attention applies softmax to attention scores.",
    )
    retriever = TfidfRetriever(repository)

    assert retriever.search("如何召回相关证据？", top_k=1)[0].chunk.filename == "rag.md"
    assert retriever.search("attention softmax", top_k=1)[0].chunk.filename == "attention.md"
    assert retriever.search("quantum banana spaceship", top_k=5) == []


def test_mixed_tokenizer_handles_nfkc_chinese_and_identifiers():
    tokens = tokenize_mixed("ＲＡＧ中的 top-k Recall v2.0")
    assert {"rag", "中", "中的", "top-k", "recall", "v2.0"} <= set(tokens)


def test_bm25_retrieves_chinese_and_exact_acronyms(tmp_path: Path):
    repository = Repository(tmp_path / "test.db")
    ingest_bytes(repository, "rag.md", "RAG 使用检索增强生成召回外部证据。".encode())
    ingest_bytes(repository, "lora.md", b"LoRA is a parameter-efficient tuning method.")
    retriever = BM25Retriever(repository)

    assert retriever.search("如何召回外部证据", top_k=1)[0].chunk.filename == "rag.md"
    assert retriever.search("LoRA", top_k=1)[0].chunk.filename == "lora.md"
    assert retriever.search("量子香蕉宇宙飞船", top_k=5) == []


class FakeEmbeddings:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        data = [
            SimpleNamespace(index=i, embedding=[float(i), 1.0]) for i in range(len(kwargs["input"]))
        ]
        return SimpleNamespace(data=data)


def test_compatible_embedder_omits_unset_dimensions():
    embeddings = FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    embedder = CompatibleEmbedder(
        None,
        "embedding-model",
        dimensions=None,
        base_url="http://127.0.0.1:9999/v1",
        client=client,
    )

    assert len(embedder.embed(["a", "b"])) == 2
    assert "dimensions" not in embeddings.requests[0]
    assert embedder.cache_dimensions == 0


def test_compatible_embedder_batches_at_thirty_two():
    embeddings = FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)
    embedder = CompatibleEmbedder(
        None,
        "embedding-model",
        base_url="http://127.0.0.1:9999/v1",
        client=client,
    )

    assert len(embedder.embed([f"text-{index}" for index in range(33)])) == 33
    assert [len(request["input"]) for request in embeddings.requests] == [32, 1]


def test_embedding_cache_key_isolated_by_endpoint():
    client = SimpleNamespace(embeddings=FakeEmbeddings())
    first = CompatibleEmbedder(
        "key", "same-model", base_url="https://one.example/v1", client=client
    )
    second = CompatibleEmbedder(
        "key", "same-model", base_url="https://two.example/v1", client=client
    )

    assert first.cache_model_key != second.cache_model_key


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id, "d1", "Paper", "paper.md", 1, 0, f"text {chunk_id}")


def make_diagnostics(high_confidence: bool, reason: str) -> LexicalDiagnostics:
    return LexicalDiagnostics(high_confidence, reason, 2.0, 0.5, 0.8, 0.0)


class FakeRetriever:
    def __init__(self, hits, diagnostics=None, error=None):
        self.hits = hits
        self.last_diagnostics = diagnostics
        self.error = error
        self.calls = 0

    def search(self, query: str, top_k: int = 5):
        self.calls += 1
        if self.error:
            raise self.error
        return self.hits[:top_k]


def test_rrf_fuses_rankings_and_removes_duplicates():
    a, b, c = make_chunk("a"), make_chunk("b"), make_chunk("c")
    result = reciprocal_rank_fusion(
        [
            [SearchHit(a, 9.0), SearchHit(b, 8.0)],
            [SearchHit(b, 0.9), SearchHit(c, 0.8)],
        ],
        top_k=3,
    )
    assert [hit.chunk.id for hit in result] == ["b", "a", "c"]


def test_auto_retriever_skips_embedding_when_bm25_is_confident():
    lexical_hit = SearchHit(make_chunk("lexical"), 3.0)
    lexical = FakeRetriever([lexical_hit], make_diagnostics(True, "high_coverage"))
    semantic = FakeRetriever([SearchHit(make_chunk("semantic"), 0.9)])
    retriever = AutoRetriever(lexical, semantic)

    assert retriever.search("RAG", top_k=1) == [lexical_hit]
    assert semantic.calls == 0
    assert retriever.last_trace.used_embedding is False


def test_auto_retriever_fuses_when_bm25_confidence_is_low():
    a, b, c = make_chunk("a"), make_chunk("b"), make_chunk("c")
    lexical = FakeRetriever(
        [SearchHit(a, 1.0), SearchHit(b, 0.9)],
        make_diagnostics(False, "score_tie"),
    )
    semantic = FakeRetriever([SearchHit(b, 0.9), SearchHit(c, 0.8)])
    retriever = AutoRetriever(lexical, semantic)

    result = retriever.search("概念性问题", top_k=3)
    assert [hit.chunk.id for hit in result] == ["b", "a", "c"]
    assert semantic.calls == 1
    assert retriever.last_trace.strategy == "rrf"


def test_auto_retriever_degrades_to_bm25_when_embedding_fails():
    lexical_hit = SearchHit(make_chunk("lexical"), 1.0)
    lexical = FakeRetriever([lexical_hit], make_diagnostics(False, "low_coverage"))
    semantic = FakeRetriever([], error=RuntimeError("service unavailable"))
    retriever = AutoRetriever(lexical, semantic)

    assert retriever.search("query", top_k=1) == [lexical_hit]
    assert retriever.last_trace.strategy == "bm25_fallback"
    assert "RuntimeError" in (retriever.last_trace.error or "")
