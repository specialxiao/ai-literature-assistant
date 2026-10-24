"""Select a local model/threshold on development questions only; never opens test queries."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from techlit_assistant.config import Settings
from techlit_assistant.database import Repository
from techlit_assistant.evaluation import load_questions, seed_directory
from techlit_assistant.retrieval import AutoRetriever, BM25Retriever, diversify_hits
from techlit_assistant.retriever_factory import build_semantic_retriever

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval/expanded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["e5", "minilm"], default=["e5", "minilm"])
    parser.add_argument("--output", type=Path, default=Path("work/dev-calibration.json"))
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.90,
        help="Preserve answerable recall first; optimize abstention within this constraint",
    )
    args = parser.parse_args()
    if not 0 <= args.min_hit_rate <= 1:
        parser.error("--min-hit-rate must be between 0 and 1")
    # Filter before inference, scoring, logging or model selection. Never tune on test.
    questions = [q for q in load_questions(DATASET / "questions.jsonl") if q["split"] == "dev"]
    settings = Settings.from_env(ROOT)
    (ROOT / "work").mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="dev-calibration-", dir=ROOT / "work") as directory:
        repository = Repository(Path(directory) / "dev.db")
        seed_directory(repository, DATASET / "corpus")
        for model in args.models:
            path = (
                ROOT
                / "data/models"
                / (
                    "multilingual-e5-small"
                    if model == "e5"
                    else "paraphrase-multilingual-MiniLM-L12-v2"
                )
            )
            config = replace(
                settings,
                embedding_provider="local",
                local_model_name=model,
                local_model_path=path,
                local_min_score=-1,
            )
            semantic = build_semantic_retriever(repository, config)
            candidates = {}
            for item in questions:
                hits = semantic.search(item["query"], top_k=len(repository.list_chunks()))
                candidates[item["query"]] = sorted(hits, key=lambda hit: -hit.score)

            class CachedSemantic:
                def __init__(self, threshold, candidates):
                    self.threshold = threshold
                    self.candidates = candidates

                def search(self, query, top_k=5):
                    return diversify_hits(
                        [hit for hit in self.candidates[query] if hit.score >= self.threshold],
                        top_k,
                    )

            for threshold in [0.0, *[value / 100 for value in range(25, 96, 5)]]:
                retriever = AutoRetriever(
                    BM25Retriever(repository), CachedSemantic(threshold, candidates)
                )
                positive = correct = negatives = rejected = 0
                for item in questions:
                    hits = retriever.search(item["query"], 5)
                    if item["relevant"]:
                        positive += 1
                        expected = {
                            (source["filename"], source["page"]) for source in item["relevant"]
                        }
                        correct += any(
                            (hit.chunk.filename, hit.chunk.page) in expected for hit in hits
                        )
                    else:
                        negatives += 1
                        rejected += not hits
                recall = correct / positive
                specificity = rejected / negatives
                results.append(
                    {
                        "model": model,
                        "threshold": threshold,
                        "hit_rate_at_5": recall,
                        "unanswerable_no_hit_rate": specificity,
                        "balanced_accuracy": (recall + specificity) / 2,
                    }
                )
            model_rows = [r for r in results if r["model"] == model]
            eligible = [r for r in model_rows if r["hit_rate_at_5"] >= args.min_hit_rate]
            best = max(
                eligible or model_rows,
                key=lambda r: (r["balanced_accuracy"], r["hit_rate_at_5"], -r["threshold"]),
            )
            print(json.dumps(best), flush=True)
    eligible = [r for r in results if r["hit_rate_at_5"] >= args.min_hit_rate]
    chosen = max(
        eligible or results,
        key=lambda r: (r["balanced_accuracy"], r["hit_rate_at_5"], -r["threshold"]),
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "split": "dev",
                "questions": len(questions),
                "objective": "maximize mean(Hit@5, no-hit) subject to minimum Hit@5",
                "min_hit_rate": args.min_hit_rate,
                "chosen": chosen,
                "candidates": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Chosen on dev only:", json.dumps(chosen), flush=True)


if __name__ == "__main__":
    main()
