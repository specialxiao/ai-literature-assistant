import unicodedata
from collections import Counter
from pathlib import Path

from techlit_assistant.evaluation import load_questions
from techlit_assistant.ingest import normalize_text, split_page_text

DATASET = Path(__file__).resolve().parents[1] / "eval/expanded"


def test_expanded_dataset_is_balanced_and_unique():
    questions = load_questions(DATASET / "questions.jsonl")
    assert len(questions) == 100
    assert len({item["id"] for item in questions}) == 100
    assert (
        len({unicodedata.normalize("NFKC", item["query"]).casefold() for item in questions}) == 100
    )
    assert Counter(item["category"] for item in questions) == {
        "en_to_en": 20,
        "zh_to_en": 20,
        "zh_to_zh": 20,
        "en_to_zh": 20,
        "unanswerable": 20,
    }
    assert Counter(item["split"] for item in questions) == {"dev": 50, "test": 50}


def test_all_positive_labels_have_real_evidence_in_the_corpus():
    questions = load_questions(DATASET / "questions.jsonl")
    paths = {path.name: path for path in (DATASET / "corpus").glob("*.md")}
    assert len(paths) == 20
    annotated = Counter()
    for item in questions:
        if not item["relevant"]:
            assert item["category"] == "unanswerable"
            assert not item["evidence"]
            continue
        assert item["evidence"]
        for source in item["relevant"]:
            assert source["filename"] in paths
            assert source["page"] == 1
            text = paths[source["filename"]].read_text(encoding="utf-8")
            assert item["evidence"] in normalize_text(text), item["id"]
            annotated[source["filename"]] += 1
    assert set(annotated) == set(paths)
    assert set(annotated.values()) == {4}


def test_positive_evidence_survives_the_application_chunking_policy():
    for item in load_questions(DATASET / "questions.jsonl"):
        if not item["relevant"]:
            continue
        path = DATASET / "corpus" / item["relevant"][0]["filename"]
        chunks = split_page_text(path.read_text(encoding="utf-8"))
        assert any(item["evidence"] in chunk for chunk in chunks), item["id"]
