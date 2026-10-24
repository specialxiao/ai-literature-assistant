from pathlib import Path

import pytest

from techlit_assistant.config import Settings

ENV_NAMES = [
    "TECHLIT_LOCAL_MODEL",
    "TECHLIT_EMBEDDING_PROVIDER",
    "TECHLIT_LOCAL_MODEL_PATH",
    "TECHLIT_LOCAL_MIN_SCORE",
    "TECHLIT_DB_PATH",
    "TECHLIT_LLM_API_KEY",
    "TECHLIT_LLM_BASE_URL",
    "TECHLIT_LLM_MODEL",
    "TECHLIT_LLM_API_MODE",
    "TECHLIT_EMBEDDING_API_KEY",
    "TECHLIT_EMBEDDING_BASE_URL",
    "TECHLIT_EMBEDDING_MODEL",
    "TECHLIT_EMBEDDING_DIMENSIONS",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CHAT_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_DIMENSIONS",
]


@pytest.fixture(autouse=True)
def clean_model_environment(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_new_settings_take_precedence_and_share_endpoint_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "legacy-chat")
    monkeypatch.setenv("TECHLIT_LLM_API_KEY", "new-key")
    monkeypatch.setenv("TECHLIT_LLM_BASE_URL", "https://provider.example/v1/")
    monkeypatch.setenv("TECHLIT_LLM_MODEL", "new-chat")
    monkeypatch.setenv("TECHLIT_EMBEDDING_MODEL", "new-embedding")

    settings = Settings.from_env(tmp_path)

    assert settings.llm.api_key == "new-key"
    assert settings.llm.base_url == "https://provider.example/v1"
    assert settings.llm.model == "new-chat"
    assert settings.llm.api_mode == "chat_completions"
    assert settings.embedding.api_key == "new-key"
    assert settings.embedding.base_url == "https://provider.example/v1"
    assert settings.embedding_dimensions is None


def test_legacy_openai_settings_keep_previous_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")

    settings = Settings.from_env(tmp_path)

    assert settings.llm.model == "gpt-5.4-mini"
    assert settings.llm.api_mode == "responses"
    assert settings.llm.base_url == "https://legacy.example/v1"
    assert settings.embedding.model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 512


def test_local_endpoint_can_be_configured_without_secret(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECHLIT_LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("TECHLIT_LLM_MODEL", "local-model")

    settings = Settings.from_env(tmp_path)

    assert settings.llm.api_key is None
    assert settings.llm.configured is True


def test_explicit_blank_new_key_blocks_unrelated_legacy_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-legacy-key")
    monkeypatch.setenv("TECHLIT_LLM_API_KEY", "")
    monkeypatch.setenv("TECHLIT_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("TECHLIT_LLM_MODEL", "deepseek-flash")

    settings = Settings.from_env(tmp_path)

    assert settings.llm.api_key is None
    assert settings.llm.configured is False


def test_invalid_base_url_has_actionable_error(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECHLIT_LLM_BASE_URL", "www.example.com")
    with pytest.raises(ValueError, match="http:// 或 https://"):
        Settings.from_env(tmp_path)


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_invalid_embedding_dimensions_are_rejected(tmp_path: Path, monkeypatch, value: str):
    monkeypatch.setenv("TECHLIT_EMBEDDING_DIMENSIONS", value)
    with pytest.raises(ValueError, match="正整数"):
        Settings.from_env(tmp_path)


def test_local_embedding_configuration(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECHLIT_EMBEDDING_PROVIDER", "local")
    settings = Settings.from_env(tmp_path)
    assert settings.embedding_provider == "local"
    assert settings.local_model_path == tmp_path / "data/models/multilingual-e5-small"
    assert settings.local_min_score == 0
    assert not settings.embedding.configured  # No API credentials are needed.


def test_minilm_profile_has_its_own_path_and_default_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHLIT_LOCAL_MODEL", "minilm")
    settings = Settings.from_env(tmp_path)
    assert settings.local_model_name == "minilm"
    assert settings.local_model_path.name == "paraphrase-multilingual-MiniLM-L12-v2"
    assert settings.local_min_score == 0.35


def test_unknown_local_model_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TECHLIT_LOCAL_MODEL", "unknown")
    with pytest.raises(ValueError, match="e5 或 minilm"):
        Settings.from_env(tmp_path)


@pytest.mark.parametrize("value", ["nan", "inf", "-2", "invalid"])
def test_local_threshold_rejects_invalid_values(tmp_path, monkeypatch, value):
    monkeypatch.setenv("TECHLIT_LOCAL_MIN_SCORE", value)
    with pytest.raises(ValueError, match="-1 到 1"):
        Settings.from_env(tmp_path)
