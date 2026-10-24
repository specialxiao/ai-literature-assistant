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


def test_evaluation_records_query_level_failures_without_retrieval_changes(tmp_path):
    repository = Repository(tmp_path / "test.db")
    ingest_bytes(repository, "rag.md", b"retrieval provides external evidence")
    questions = [
        {
            "id": "positive",
            "query": "external evidence",
            "category": "en_to_en",
            "split": "test",
            "relevant": [{"filename": "rag.md", "page": 1}],
            "evidence": "external evidence",
        },
        {
            "id": "negative",
            "query": "quantum banana spaceship",
            "category": "unanswerable",
            "split": "test",
            "relevant": [],
            "evidence": "",
        },
    ]
    details = []
    metrics = evaluate_retrieval(TfidfRetriever(repository), questions, details=details)
    assert metrics.recall_at_k == 1
    assert len(details) == 2
    assert details[0]["first_relevant_rank"] == 1
    assert details[0]["evidence_text_hit"]
    assert details[1]["first_relevant_rank"] is None
    assert details[1]["hits"] == []
