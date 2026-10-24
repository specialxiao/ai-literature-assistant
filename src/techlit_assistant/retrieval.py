from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .database import Repository
from .models import Chunk, SearchHit

CJK_RE = re.compile(r"[\u3400-\u9fff]+")
TOKEN_SPAN_RE = re.compile(
    r"[\u3400-\u9fff]+|[a-z0-9]+(?:[._/+:-][a-z0-9]+)*",
    flags=re.IGNORECASE,
)
SELECTIVE_QUERY_RE = re.compile(r"\b[A-Z][A-Z0-9._+-]{1,}\b|[`\"“”][^`\"“”]+[`\"“”]")


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize Chinese as unigrams+bigrams and preserve Latin identifiers."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in TOKEN_SPAN_RE.finditer(normalized):
        span = match.group(0)
        if CJK_RE.fullmatch(span):
            tokens.extend(span)
            tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
        else:
            tokens.append(span)
    return tokens


@dataclass(frozen=True)
class LexicalDiagnostics:
    high_confidence: bool
    reason: str
    top_score: float
    relative_gap: float
    idf_coverage: float
    oov_ratio: float


@dataclass(frozen=True)
class RetrievalTrace:
    strategy: str
    reason: str
    used_embedding: bool
    lexical: LexicalDiagnostics
    error: str | None = None


EMPTY_DIAGNOSTICS = LexicalDiagnostics(
    high_confidence=False,
    reason="no_lexical_hit",
    top_score=0.0,
    relative_gap=0.0,
    idf_coverage=0.0,
    oov_ratio=1.0,
)


class TfidfRetriever:
    """A deterministic multilingual baseline using character n-grams."""

    def __init__(self, repository: Repository, min_score: float = 0.05):
        self.repository = repository
        self.min_score = min_score

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query = query.strip()
        chunks = self.repository.list_chunks()
        if not query or not chunks or top_k <= 0:
            return []

        contains_cjk = bool(CJK_RE.search(query))
        vectorizer = TfidfVectorizer(
            analyzer="char" if contains_cjk else "word",
            ngram_range=(2, 4) if contains_cjk else (1, 2),
            lowercase=True,
            sublinear_tf=True,
            max_features=50_000,
        )
        document_matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        query_vector = vectorizer.transform([query])
        scores = (document_matrix @ query_vector.T).toarray().ravel()
        ranked = np.argsort(scores)[::-1]
        return [
            SearchHit(chunk=chunks[index], score=float(scores[index]))
            for index in ranked[:top_k]
            if scores[index] >= self.min_score
        ]


class BM25Retriever:
    """BM25 with a tokenizer designed for Chinese/English technical text."""

    def __init__(
        self,
        repository: Repository,
        k1: float = 1.5,
        b: float = 0.75,
        min_score: float = 0.0,
    ):
        self.repository = repository
        self.k1 = k1
        self.b = b
        self.min_score = min_score
        self.last_diagnostics = EMPTY_DIAGNOSTICS

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query = query.strip()
        chunks = self.repository.list_chunks()
        if not query or not chunks or top_k <= 0:
            self.last_diagnostics = EMPTY_DIAGNOSTICS
            return []

        tokenized_corpus = [tokenize_mixed(chunk.text) for chunk in chunks]
        query_tokens = list(dict.fromkeys(tokenize_mixed(query)))
        if not query_tokens:
            self.last_diagnostics = EMPTY_DIAGNOSTICS
            return []

        document_frequency: Counter[str] = Counter()
        term_frequencies: list[Counter[str]] = []
        for tokens in tokenized_corpus:
            frequencies = Counter(tokens)
            term_frequencies.append(frequencies)
            document_frequency.update(frequencies.keys())

        document_count = len(chunks)
        average_length = max(
            sum(len(tokens) for tokens in tokenized_corpus) / document_count,
            1.0,
        )
        idf = {
            token: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

        scores: list[float] = []
        for tokens, frequencies in zip(tokenized_corpus, term_frequencies, strict=True):
            length_normalizer = self.k1 * (1.0 - self.b + self.b * len(tokens) / average_length)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1.0)
                score += idf.get(token, 0.0) * numerator / (frequency + length_normalizer)
            scores.append(score)

        ranked = sorted(range(document_count), key=lambda index: (-scores[index], chunks[index].id))
        positive_ranked = [index for index in ranked if scores[index] > self.min_score]
        self.last_diagnostics = self._diagnose(
            query,
            query_tokens,
            tokenized_corpus,
            idf,
            document_count,
            scores,
            positive_ranked,
        )
        if self.last_diagnostics.idf_coverage < 0.05 and self.last_diagnostics.oov_ratio > 0.80:
            return []
        return [
            SearchHit(chunk=chunks[index], score=float(scores[index]))
            for index in positive_ranked[:top_k]
        ]

    @staticmethod
    def _diagnose(
        query: str,
        query_tokens: list[str],
        tokenized_corpus: list[list[str]],
        idf: dict[str, float],
        document_count: int,
        scores: list[float],
        ranked: list[int],
    ) -> LexicalDiagnostics:
        if not ranked:
            return EMPTY_DIAGNOSTICS

        meaningful = [
            token for token in query_tokens if len(token) >= 2 or not CJK_RE.fullmatch(token)
        ] or query_tokens
        meaningful = list(dict.fromkeys(meaningful))
        maximum_idf = math.log(1.0 + (document_count + 0.5) / 0.5)
        weights = {token: idf.get(token, maximum_idf) for token in meaningful}
        total_weight = max(sum(weights.values()), 1e-12)

        top_tokens = set(tokenized_corpus[ranked[0]])
        coverage = (
            sum(weight for token, weight in weights.items() if token in top_tokens) / total_weight
        )
        oov_ratio = sum(token not in idf for token in meaningful) / len(meaningful)

        top_score = scores[ranked[0]]
        second_score = scores[ranked[1]] if len(ranked) > 1 else 0.0
        relative_gap = (top_score - second_score) / max(top_score, 1e-12)
        short_exact_query = len(meaningful) <= 4 and all(
            token in top_tokens for token in meaningful
        )
        selective_match = any(
            bool(selected := tokenize_mixed(match.group(0))) and set(selected) <= top_tokens
            for match in SELECTIVE_QUERY_RE.finditer(query)
        )

        high_confidence = (
            top_score > 0
            and coverage >= 0.70
            and oov_ratio <= 0.40
            and (relative_gap >= 0.12 or short_exact_query or selective_match)
        )
        if high_confidence:
            reason = "high_coverage"
        elif oov_ratio > 0.40:
            reason = "high_oov"
        elif coverage < 0.70:
            reason = "low_coverage"
        else:
            reason = "score_tie"
        return LexicalDiagnostics(
            high_confidence=high_confidence,
            reason=reason,
            top_score=top_score,
            relative_gap=relative_gap,
            idf_coverage=coverage,
            oov_ratio=oov_ratio,
        )


class Embedder(Protocol):
    cache_model_key: str
    cache_dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class CompatibleEmbedder:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        dimensions: int | None = None,
        base_url: str | None = None,
        client=None,
    ):
        if not model:
            raise ValueError("语义向量检索需要配置 TECHLIT_EMBEDDING_MODEL")
        if not api_key and not base_url and client is None:
            raise ValueError("语义向量检索需要 API Key，或配置一个本地兼容服务地址")
        if client is None:
            from openai import OpenAI

            client_kwargs = {"api_key": api_key or "not-required", "timeout": 60.0}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)
        self.client = client
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url

    @property
    def cache_model_key(self) -> str:
        endpoint = self.base_url or "default-api-endpoint"
        endpoint_fingerprint = sha256(endpoint.encode()).hexdigest()[:12]
        return f"{endpoint_fingerprint}:{self.model}"

    @property
    def cache_dimensions(self) -> int:
        return self.dimensions or 0

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            request = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions
            response = self.client.embeddings.create(**request)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class EmbeddingRetriever:
    def __init__(
        self,
        repository: Repository,
        embedder: Embedder,
        min_score: float = 0.20,
    ):
        self.repository = repository
        self.embedder = embedder
        self.min_score = min_score

    def _ensure_index(self, chunks: list[Chunk]) -> dict[str, list[float]]:
        stored = self.repository.load_embeddings(
            self.embedder.cache_model_key, self.embedder.cache_dimensions
        )
        missing = [chunk for chunk in chunks if chunk.id not in stored]
        if missing:
            generated = self.embedder.embed_documents([chunk.text for chunk in missing])
            additions = {chunk.id: vector for chunk, vector in zip(missing, generated, strict=True)}
            self.repository.save_embeddings(
                self.embedder.cache_model_key,
                self.embedder.cache_dimensions,
                additions,
            )
            stored.update(additions)
        return stored

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query = query.strip()
        chunks = self.repository.list_chunks()
        if not query or not chunks or top_k <= 0:
            return []

        stored = self._ensure_index(chunks)
        query_vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        matrix = np.asarray([stored[chunk.id] for chunk in chunks], dtype=np.float32)
        query_norm = np.linalg.norm(query_vector)
        row_norms = np.linalg.norm(matrix, axis=1)
        denominator = np.maximum(row_norms * query_norm, 1e-12)
        scores = (matrix @ query_vector) / denominator
        ranked = np.argsort(scores)[::-1]
        candidates = [
            SearchHit(chunk=chunks[index], score=float(scores[index]))
            for index in ranked
            if scores[index] >= self.min_score
        ]
        return diversify_hits(candidates, top_k)


def diversify_hits(
    hits: Sequence[SearchHit], top_k: int, max_per_source: int = 2
) -> list[SearchHit]:
    """Prefer source-page breadth without discarding long-page evidence when needed."""
    if top_k <= 0:
        return []
    selected: list[SearchHit] = []
    deferred: list[SearchHit] = []
    counts: Counter[tuple[str, int]] = Counter()
    seen: set[str] = set()
    for hit in hits:
        if hit.chunk.id in seen:
            continue
        seen.add(hit.chunk.id)
        source = (hit.chunk.document_id, hit.chunk.page)
        if counts[source] < max_per_source:
            selected.append(hit)
            counts[source] += 1
        else:
            deferred.append(hit)
        if len(selected) >= top_k:
            break
    return (selected + deferred[: max(0, top_k - len(selected))])[:top_k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    top_k: int,
    rrf_k: int = 60,
) -> list[SearchHit]:
    if top_k <= 0:
        return []
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    lexical_rank: dict[str, int] = {}
    for ranking_index, ranking in enumerate(rankings):
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunks[chunk_id] = hit.chunk
            if ranking_index == 0:
                lexical_rank[chunk_id] = rank
    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            lexical_rank.get(chunk_id, 10**9),
            chunk_id,
        ),
    )
    return [
        SearchHit(chunk=chunks[chunk_id], score=scores[chunk_id])
        for chunk_id in ordered_ids[:top_k]
    ]


class AutoRetriever:
    """Use BM25 when confident; otherwise fuse BM25 with embeddings."""

    def __init__(
        self,
        lexical: BM25Retriever,
        semantic: EmbeddingRetriever | None = None,
    ):
        self.lexical = lexical
        self.semantic = semantic
        self.last_trace = RetrievalTrace(
            strategy="bm25",
            reason="not_run",
            used_embedding=False,
            lexical=EMPTY_DIAGNOSTICS,
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not query.strip() or top_k <= 0:
            self.last_trace = RetrievalTrace("bm25", "no_lexical_hit", False, EMPTY_DIAGNOSTICS)
            return []
        candidate_k = max(10, top_k * 3)
        lexical_hits = self.lexical.search(query, top_k=candidate_k)
        diagnostics = self.lexical.last_diagnostics

        if diagnostics.high_confidence:
            self.last_trace = RetrievalTrace(
                strategy="bm25",
                reason=diagnostics.reason,
                used_embedding=False,
                lexical=diagnostics,
            )
            return lexical_hits[:top_k]

        if self.semantic is None:
            self.last_trace = RetrievalTrace(
                strategy="bm25_fallback",
                reason=f"embedding_unavailable:{diagnostics.reason}",
                used_embedding=False,
                lexical=diagnostics,
            )
            return lexical_hits[:top_k]

        try:
            semantic_hits = self.semantic.search(query, top_k=candidate_k)
        except Exception as exc:  # noqa: BLE001 - cascade boundary must degrade offline.
            self.last_trace = RetrievalTrace(
                strategy="bm25_fallback",
                reason=f"embedding_error:{diagnostics.reason}",
                used_embedding=True,
                lexical=diagnostics,
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            return lexical_hits[:top_k]

        if not semantic_hits:
            self.last_trace = RetrievalTrace(
                strategy="no_evidence",
                reason="semantic_below_threshold",
                used_embedding=True,
                lexical=diagnostics,
            )
            return []

        if diagnostics.idf_coverage < 0.35 or diagnostics.oov_ratio > 0.60:
            self.last_trace = RetrievalTrace(
                strategy="semantic",
                reason="weak_lexical_match",
                used_embedding=True,
                lexical=diagnostics,
            )
            return diversify_hits(semantic_hits, top_k)

        self.last_trace = RetrievalTrace(
            strategy="rrf",
            reason=diagnostics.reason,
            used_embedding=True,
            lexical=diagnostics,
        )
        fused = reciprocal_rank_fusion(
            [lexical_hits, semantic_hits], top_k=len(lexical_hits) + len(semantic_hits)
        )
        return diversify_hits(fused, top_k)


# Compatibility alias for code written against v0.1.
OpenAIEmbedder = CompatibleEmbedder
