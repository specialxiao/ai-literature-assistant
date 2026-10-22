from pathlib import Path

from techlit_assistant.database import Repository
from techlit_assistant.evaluation import build_parser, evaluate_retrieval
from techlit_assistant.ingest import ingest_bytes
from techlit_assistant.retrieval import TfidfRetriever


def test_evaluation_metrics(tmp_path: Path):
    repository = Repository(tmp_path / "test.db")
    ingest_bytes(repository, "rag.md", "检索增强生成会先检索外部资料。".encode())
    retriever = TfidfRetriever(repository)
    questions = [
        {
            "query": "什么方法会先检索外部资料？",
            "relevant": [{"filename": "rag.md", "page": 1}],
        },
        {"query": "quantum banana spaceship", "relevant": []},
    ]

    metrics = evaluate_retrieval(retriever, questions, top_k=5)
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.unanswerable_no_hit_rate == 1.0
    assert metrics.embedding_api_query_rate == 0.0


def test_legacy_openai_cli_mode_remains_available():
    args = build_parser().parse_args(["--questions", "questions.jsonl", "--mode", "openai"])
    assert args.mode == "openai"


def test_evaluation_defaults_to_separate_database():
    args = build_parser().parse_args(["--questions", "questions.jsonl", "--mode", "local"])
    assert args.database == Path("data/evaluation.db")
