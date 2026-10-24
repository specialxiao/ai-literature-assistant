from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings
from .database import Repository
from .ingest import ingest_bytes
from .retrieval import (
    AutoRetriever,
    BM25Retriever,
    CompatibleEmbedder,
    EmbeddingRetriever,
    TfidfRetriever,
)
from .retriever_factory import build_semantic_retriever


@dataclass(frozen=True)
class EvaluationMetrics:
    evaluated_answerable: int
    evaluated_unanswerable: int
    recall_at_k: float
    mrr: float
    unanswerable_no_hit_rate: float
    average_latency_ms: float
    p95_latency_ms: float
    embedding_call_rate: float
    embedding_api_query_rate: float


def load_questions(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if "query" not in item or "relevant" not in item:
                raise ValueError(f"{path}:{line_number} 缺少 query 或 relevant")
            items.append(item)
    return items


def evaluate_retrieval(
    retriever, questions: list[dict], top_k: int = 5, *, details: list[dict] | None = None
) -> EvaluationMetrics:
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    unanswerable_no_hits: list[float] = []
    latencies: list[float] = []
    embedding_calls = 0
    api_queries = 0

    for item in questions:
        started = time.perf_counter()
        hits = retriever.search(item["query"], top_k=top_k)
        trace = getattr(retriever, "last_trace", None)
        if isinstance(retriever, EmbeddingRetriever) or (
            trace is not None and trace.used_embedding
        ):
            embedding_calls += 1
            semantic = (
                retriever if isinstance(retriever, EmbeddingRetriever) else retriever.semantic
            )
            if isinstance(getattr(semantic, "embedder", None), CompatibleEmbedder):
                api_queries += 1
        latencies.append((time.perf_counter() - started) * 1000)
        relevant = {(source["filename"], int(source["page"])) for source in item["relevant"]}
        ranks = [
            rank
            for rank, hit in enumerate(hits, start=1)
            if (hit.chunk.filename, hit.chunk.page) in relevant
        ]
        if details is not None:
            details.append(
                {
                    "id": item.get("id"),
                    "query": item["query"],
                    "category": item.get("category", "unspecified"),
                    "split": item.get("split", "unspecified"),
                    "answerable": bool(relevant),
                    "first_relevant_rank": min(ranks) if ranks else None,
                    "evidence_text_hit": bool(item.get("evidence"))
                    and any(item["evidence"] in hit.chunk.text for hit in hits),
                    "latency_ms": latencies[-1],
                    "strategy": trace.strategy if trace else type(retriever).__name__,
                    "fallback": bool(trace and trace.strategy == "bm25_fallback"),
                    "hits": [
                        {
                            "filename": hit.chunk.filename,
                            "page": hit.chunk.page,
                            "chunk_id": hit.chunk.id,
                            "score": hit.score,
                        }
                        for hit in hits
                    ],
                }
            )
        if not relevant:
            unanswerable_no_hits.append(float(not hits))
            continue

        recalls.append(float(bool(ranks)))
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)

    p95 = 0.0
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[max(0, round(0.95 * len(ordered)) - 1)]
    return EvaluationMetrics(
        evaluated_answerable=len(recalls),
        evaluated_unanswerable=len(unanswerable_no_hits),
        recall_at_k=statistics.fmean(recalls) if recalls else 0.0,
        mrr=statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        unanswerable_no_hit_rate=(
            statistics.fmean(unanswerable_no_hits) if unanswerable_no_hits else 0.0
        ),
        average_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
        p95_latency_ms=p95,
        embedding_call_rate=embedding_calls / len(questions) if questions else 0.0,
        embedding_api_query_rate=api_queries / len(questions) if questions else 0.0,
    )


def seed_directory(repository: Repository, directory: Path) -> None:
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in {".pdf", ".md", ".txt"}:
            result = ingest_bytes(repository, path.name, path.read_bytes())
            status = "imported" if result.created else "skipped duplicate"
            print(f"{status}: {path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate literature retrieval.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/evaluation.db"),
        help="Isolated evaluation database; defaults to data/evaluation.db, not the app library",
    )
    parser.add_argument(
        "--mode",
        choices=["tfidf", "bm25", "auto", "local", "api", "openai"],
        default="tfidf",
        help="'openai' is retained as a legacy alias for 'api'",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--reset", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env(Path.cwd())
    database_path = args.database.resolve()
    if database_path == settings.database_path.resolve():
        raise SystemExit("评测数据库不能使用应用资料库，请另选 --database 路径")
    repository = Repository(database_path)
    if args.reset:
        repository.clear()
    if args.seed_dir:
        seed_directory(repository, args.seed_dir)

    if args.mode == "tfidf":
        retriever = TfidfRetriever(repository)
    elif args.mode == "bm25":
        retriever = BM25Retriever(repository)
    elif args.mode in {"local", "api", "openai"}:
        provider = "local" if args.mode == "local" else "api"
        retriever = build_semantic_retriever(repository, settings, provider)
        if retriever is None:
            raise SystemExit(
                "Configure TECHLIT_EMBEDDING_MODEL and an API key or base URL for API mode"
            )
    else:
        semantic = build_semantic_retriever(repository, settings)
        retriever = AutoRetriever(BM25Retriever(repository), semantic)

    metrics = evaluate_retrieval(retriever, load_questions(args.questions), args.k)
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
