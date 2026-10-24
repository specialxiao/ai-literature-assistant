"""Compare retrieval modes on the fixed synthetic expansion, without cloud calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from techlit_assistant.config import Settings
from techlit_assistant.database import Repository
from techlit_assistant.evaluation import evaluate_retrieval, load_questions, seed_directory
from techlit_assistant.retrieval import AutoRetriever, BM25Retriever, TfidfRetriever
from techlit_assistant.retriever_factory import build_semantic_retriever

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval/expanded"


def summarize(rows: list[dict]) -> dict:
    positive = [row for row in rows if row["answerable"]]
    negative = [row for row in rows if not row["answerable"]]
    return {
        "questions": len(rows),
        "answerable": len(positive),
        "unanswerable": len(negative),
        "hit_rate_at_k": (
            statistics.fmean(row["first_relevant_rank"] is not None for row in positive)
            if positive
            else None
        ),
        "mrr": (
            statistics.fmean(
                1 / row["first_relevant_rank"] if row["first_relevant_rank"] else 0
                for row in positive
            )
            if positive
            else None
        ),
        "evidence_text_hit_rate": (
            statistics.fmean(row["evidence_text_hit"] for row in positive) if positive else None
        ),
        "unanswerable_no_hit_rate": (
            statistics.fmean(not row["hits"] for row in negative) if negative else None
        ),
        "fallback_queries": sum(row["fallback"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["tfidf", "bm25", "local", "auto"],
        default=["tfidf", "bm25", "local", "auto"],
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("work/expanded-evaluation.json"))
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    questions = load_questions(DATASET / "questions.jsonl")
    fingerprint = hashlib.sha256()
    for path in [DATASET / "questions.jsonl", *sorted((DATASET / "corpus").glob("*.md"))]:
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())
    # Force local embeddings even if the user's .env selects a paid API backend.
    settings = replace(Settings.from_env(ROOT), embedding_provider="local")
    report = {
        "dataset": "synthetic-technical-notes-v1",
        "dataset_sha256": fingerprint.hexdigest(),
        "questions": len(questions),
        "categories": dict(Counter(item["category"] for item in questions)),
        "k": args.k,
        "local_min_score": settings.local_min_score,
        "local_model": settings.local_model_name,
        "notes": "Synthetic page-level retrieval evaluation, not real-paper or answer-quality results. "
        "Metrics exclude one warmup query. First semantic warmup includes cold model load "
        "and indexing; later modes reuse model and document vectors. No cloud APIs.",
        "modes": {},
    }
    (ROOT / "work").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="expanded-eval-", dir=ROOT / "work") as directory:
        repository = Repository(Path(directory) / "evaluation.db")
        seed_directory(repository, DATASET / "corpus")
        report["documents"] = len(repository.list_documents())
        report["chunks"] = len(repository.list_chunks())
        for mode in args.modes:
            if mode == "tfidf":
                retriever = TfidfRetriever(repository)
            elif mode == "bm25":
                retriever = BM25Retriever(repository)
            else:
                semantic = build_semantic_retriever(repository, settings)
                retriever = (
                    semantic
                    if mode == "local"
                    else AutoRetriever(BM25Retriever(repository), semantic)
                )
            started = time.perf_counter()
            retriever.search(questions[0]["query"], top_k=args.k)
            warmup_ms = (time.perf_counter() - started) * 1000
            details: list[dict] = []
            metrics = evaluate_retrieval(retriever, questions, args.k, details=details)
            result = {
                "metrics": asdict(metrics),
                "summary": summarize(details),
                "warmup_ms": warmup_ms,
                "by_category": {
                    group: summarize([row for row in details if row["category"] == group])
                    for group in sorted({row["category"] for row in details})
                },
                "by_split": {
                    split: summarize([row for row in details if row["split"] == split])
                    for split in ("dev", "test")
                },
                "queries": details,
            }
            report["modes"][mode] = result
            print(
                json.dumps(
                    {
                        "mode": mode,
                        **result["summary"],
                        "warm_average_ms": metrics.average_latency_ms,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report saved: {output.name}", flush=True)


if __name__ == "__main__":
    main()
