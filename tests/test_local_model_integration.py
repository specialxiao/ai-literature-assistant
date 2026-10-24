"""Opt-in real-model checks: TECHLIT_RUN_LOCAL_TESTS=1 python -m pytest tests/test_local_model_integration.py"""

import os
from pathlib import Path

import numpy as np
import pytest

from techlit_assistant import local_embedding
from techlit_assistant.database import Repository
from techlit_assistant.ingest import ingest_bytes, split_page_text
from techlit_assistant.local_embedding import LocalE5Embedder
from techlit_assistant.retrieval import EmbeddingRetriever

pytestmark = pytest.mark.skipif(
    os.getenv("TECHLIT_RUN_LOCAL_TESTS") != "1",
    reason="Requires optional dependencies and downloaded E5",
)
MODEL_PATH = Path(__file__).resolve().parents[1] / "data/models/multilingual-e5-small"


def test_expanded_corpus_contains_real_overflow_and_tail_evidence():
    tokenizer, _ = local_embedding._load_cpu_model(str(MODEL_PATH.resolve()))
    path = Path(__file__).resolve().parents[1] / "eval/expanded/corpus/zh_long_context.md"
    chunks = split_page_text(path.read_text(encoding="utf-8"))
    target = next(chunk for chunk in chunks if "TAIL-TRACE-27" in chunk)
    before_tail = target.split("TAIL-TRACE-27", 1)[0]
    tokens = tokenizer.encode(before_tail, add_special_tokens=False, verbose=False)
    assert len(tokens) > 512  # This would be lost by naive model-level truncation.


def test_real_local_bilingual_cross_language_retrieval(tmp_path):
    repository = Repository(tmp_path / "bilingual.db")
    sources = {
        "attention_en.txt": "Self-attention computes query-key similarities, applies softmax, "
        "and combines value vectors using these attention weights.",
        "rag_zh.txt": "检索增强生成会先检索外部知识库中的相关文档，"
        "再将证据作为上下文交给大语言模型生成回答，减少事实幻觉。",
        "cooking.txt": "To bake a chocolate cake, mix flour, eggs, sugar and cocoa, then bake.",
    }
    for filename, content in sources.items():
        ingest_bytes(repository, filename, content.encode())
    embedder = LocalE5Embedder(MODEL_PATH)
    retriever = EmbeddingRetriever(repository, embedder, min_score=0)
    assert (
        retriever.search("自注意力如何用查询键计算权重并聚合值向量？", 1)[0].chunk.filename
        == "attention_en.txt"
    )
    assert (
        retriever.search(
            "How does retrieval augmented generation use external evidence to reduce hallucinations?",
            1,
        )[0].chunk.filename
        == "rag_zh.txt"
    )
    vectors = repository.load_embeddings(embedder.cache_model_key, 384)
    assert len(vectors) == 3
    for vector in vectors.values():
        assert len(vector) == 384
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-5)
    # A new retriever reuses persisted vectors without re-encoding documents.
    second = LocalE5Embedder(MODEL_PATH)

    def unexpected_document_encoding(texts):
        pytest.fail("Persisted vectors should be reused")

    second.embed_documents = unexpected_document_encoding
    assert EmbeddingRetriever(repository, second).search("attention", 1)
    assert local_embedding._load_cpu_model.cache_info().currsize == 1


def test_real_long_chunk_windows_fit_model_context(monkeypatch):
    tokenizer, model = local_embedding._load_cpu_model(str(MODEL_PATH.resolve()))
    batches = []

    class RecordingModel:
        def __call__(self, **batch):
            batches.append(batch["input_ids"].clone())
            return model(**batch)

    monkeypatch.setattr(local_embedding, "_load_cpu_model", lambda _: (tokenizer, RecordingModel()))
    vector = LocalE5Embedder(MODEL_PATH).embed_documents(
        ["Self-attention aggregates values. 中文检索证据。 " * 100 + "TAILMARKER"]
    )[0]
    assert sum(batch.shape[0] for batch in batches) > 1
    assert all(batch.shape[1] <= 512 for batch in batches)
    last_window = tokenizer.decode(batches[-1][-1], skip_special_tokens=True)
    assert "TAILMARKER" in last_window
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-5)


def test_minilm_encoding_matches_official_short_text_and_respects_128_tokens(monkeypatch):
    path = MODEL_PATH.parent / "paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer, model = local_embedding._load_cpu_model(str(path.resolve()))
    batches = []

    class RecordingModel:
        def __call__(self, **batch):
            batches.append(batch["input_ids"].clone())
            return model(**batch)

    monkeypatch.setattr(local_embedding, "_load_cpu_model", lambda _: (tokenizer, RecordingModel()))
    embedder = LocalE5Embedder(path, "minilm")
    short = "A cache stores previously computed key and value vectors."
    embedder.embed_documents([short])
    expected = tokenizer.encode(short)
    assert batches[0][0].tolist() == expected
    batches.clear()
    vector = embedder.embed_documents([short * 40 + "TAILMARKER"])[0]
    assert all(batch.shape[1] <= 128 for batch in batches)
    assert sum(batch.shape[0] for batch in batches) > 1
    assert "TAILMARKER" in tokenizer.decode(batches[-1][-1], skip_special_tokens=True)
    assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-5)


def test_e5_prefix_is_tokenized_jointly_with_content(monkeypatch):
    tokenizer, model = local_embedding._load_cpu_model(str(MODEL_PATH.resolve()))
    batches = []

    class RecordingModel:
        def __call__(self, **batch):
            batches.append(batch["input_ids"].clone())
            return model(**batch)

    monkeypatch.setattr(local_embedding, "_load_cpu_model", lambda _: (tokenizer, RecordingModel()))
    LocalE5Embedder(MODEL_PATH).embed_query("中文与 English 混合内容")
    assert batches[0][0].tolist() == tokenizer.encode("query: 中文与 English 混合内容")
