from types import SimpleNamespace

import pytest

from techlit_assistant.answer import INSUFFICIENT_EVIDENCE, CompatibleAnswerer
from techlit_assistant.models import Chunk, SearchHit


class FakeResponses:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.last_request = kwargs
        return SimpleNamespace(output_text=self.owner.text)


class FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.last_request = kwargs
        message = SimpleNamespace(content=self.owner.text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, text: str):
        self.text = text
        self.last_request = None
        self.responses = FakeResponses(self)
        self.chat = SimpleNamespace(completions=FakeCompletions(self))


def make_hit() -> SearchHit:
    return SearchHit(
        chunk=Chunk("c1", "d1", "RAG Notes", "rag.md", 2, 0, "RAG uses retrieval."),
        score=0.8,
    )


def test_no_evidence_does_not_call_model():
    client = FakeClient("should not appear")
    answerer = CompatibleAnswerer("", "test", client=client)
    result = answerer.answer("question", [])
    assert result.text == INSUFFICIENT_EVIDENCE
    assert result.generated is False
    assert client.last_request is None


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
def test_unknown_source_marker_is_blocked(api_mode: str):
    answerer = CompatibleAnswerer("", "test", api_mode=api_mode, client=FakeClient("Claim [S9]"))
    result = answerer.answer("question", [make_hit()])
    assert "已被拦截" in result.text
    assert result.generated is False


def test_responses_mode_keeps_store_disabled():
    client = FakeClient("Claim [S1]")
    result = CompatibleAnswerer("", "test", client=client).answer("question", [make_hit()])
    assert result.text == "Claim [S1]"
    assert client.last_request["store"] is False
    assert "instructions" in client.last_request


def test_chat_completions_mode_maps_system_and_user_messages():
    client = FakeClient("Claim [S1]")
    result = CompatibleAnswerer("", "test", api_mode="chat_completions", client=client).answer(
        "question", [make_hit()]
    )
    assert result.text == "Claim [S1]"
    assert [message["role"] for message in client.last_request["messages"]] == [
        "system",
        "user",
    ]


def test_unsupported_api_mode_is_clear():
    with pytest.raises(ValueError, match="不支持"):
        CompatibleAnswerer("key", "model", api_mode="unknown")


def test_local_endpoint_constructs_without_real_secret():
    answerer = CompatibleAnswerer(
        None,
        "local-model",
        base_url="http://127.0.0.1:9999/v1",
        api_mode="chat_completions",
    )
    assert answerer.model == "local-model"
