from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _has_env(*names: str) -> bool:
    return any(name in os.environ for name in names)


def _new_or_legacy(primary: str, *legacy_names: str) -> str | None:
    """An explicitly present new variable, even blank, blocks legacy fallback."""
    if primary in os.environ:
        return os.getenv(primary, "").strip() or None
    return _first_env(*legacy_names)


def _normalize_base_url(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"API Base URL 格式无效：{value}。地址必须以 http:// 或 https:// 开头")
    return normalized


@dataclass(frozen=True)
class EndpointSettings:
    api_key: str | None
    base_url: str | None
    model: str

    @property
    def configured(self) -> bool:
        if not self.model:
            return False
        if not self.base_url:
            return bool(self.api_key)
        hostname = (urlparse(self.base_url).hostname or "").lower()
        is_local = hostname in {"localhost", "127.0.0.1", "::1"}
        return bool(self.api_key or is_local)

    @property
    def display_name(self) -> str:
        if not self.base_url:
            return "默认 API 端点"
        parsed = urlparse(self.base_url)
        return parsed.netloc or self.base_url


@dataclass(frozen=True)
class LLMSettings(EndpointSettings):
    api_mode: str


@dataclass(frozen=True)
class Settings:
    project_root: Path
    database_path: Path
    llm: LLMSettings
    embedding: EndpointSettings
    embedding_dimensions: int | None
    embedding_provider: str = "api"
    local_model_path: Path = Path("data/models/multilingual-e5-small")
    local_min_score: float = 0.0
    local_model_name: str = "e5"

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> Settings:
        root = (project_root or Path.cwd()).resolve()
        # Explicit process variables (for CLI experiments/deployments) win over .env.
        load_dotenv(root / ".env", override=False)

        provider = _first_env("TECHLIT_EMBEDDING_PROVIDER") or "api"
        if provider not in {"local", "api"}:
            raise ValueError("TECHLIT_EMBEDDING_PROVIDER 必须是 local 或 api")
        local_model_name = _first_env("TECHLIT_LOCAL_MODEL") or "e5"
        if local_model_name not in {"e5", "minilm"}:
            raise ValueError("TECHLIT_LOCAL_MODEL 必须是 e5 或 minilm")
        model_path = Path(
            _first_env("TECHLIT_LOCAL_MODEL_PATH")
            or (
                "data/models/paraphrase-multilingual-MiniLM-L12-v2"
                if local_model_name == "minilm"
                else "data/models/multilingual-e5-small"
            )
        )
        model_path = model_path if model_path.is_absolute() else root / model_path
        try:
            local_min_score = float(
                _first_env("TECHLIT_LOCAL_MIN_SCORE")
                or ("0.35" if local_model_name == "minilm" else "0")
            )
        except ValueError as exc:
            raise ValueError("TECHLIT_LOCAL_MIN_SCORE 必须是 -1 到 1 之间的数") from exc
        if not -1 <= local_min_score <= 1:
            raise ValueError("TECHLIT_LOCAL_MIN_SCORE 必须是 -1 到 1 之间的数")

        raw_db_path = Path(os.getenv("TECHLIT_DB_PATH", "data/app.db"))
        db_path = raw_db_path if raw_db_path.is_absolute() else root / raw_db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        legacy_key = _first_env("OPENAI_API_KEY")
        using_new_llm_config = _has_env(
            "TECHLIT_LLM_API_KEY",
            "TECHLIT_LLM_BASE_URL",
            "TECHLIT_LLM_MODEL",
            "TECHLIT_LLM_API_MODE",
        )
        llm_key = _new_or_legacy("TECHLIT_LLM_API_KEY", "OPENAI_API_KEY")
        llm_base_url = _normalize_base_url(
            _new_or_legacy("TECHLIT_LLM_BASE_URL", "OPENAI_BASE_URL")
        )
        llm_model = _new_or_legacy("TECHLIT_LLM_MODEL", "OPENAI_CHAT_MODEL")
        if not llm_model and legacy_key and not using_new_llm_config:
            llm_model = "gpt-5.4-mini"

        explicit_mode = _first_env("TECHLIT_LLM_API_MODE")
        api_mode = explicit_mode or (
            "chat_completions" if _first_env("TECHLIT_LLM_BASE_URL") else "responses"
        )
        if api_mode not in {"responses", "chat_completions"}:
            raise ValueError("TECHLIT_LLM_API_MODE 必须是 responses 或 chat_completions")

        using_new_embedding_config = _has_env(
            "TECHLIT_EMBEDDING_API_KEY",
            "TECHLIT_EMBEDDING_BASE_URL",
            "TECHLIT_EMBEDDING_MODEL",
            "TECHLIT_EMBEDDING_DIMENSIONS",
            "TECHLIT_LLM_API_KEY",
            "TECHLIT_LLM_BASE_URL",
        )
        if "TECHLIT_EMBEDDING_API_KEY" in os.environ:
            embedding_key = os.getenv("TECHLIT_EMBEDDING_API_KEY", "").strip() or llm_key
        else:
            embedding_key = llm_key if using_new_llm_config else legacy_key
        embedding_base_url = _normalize_base_url(
            _new_or_legacy("TECHLIT_EMBEDDING_BASE_URL") or llm_base_url
        )
        embedding_model = _new_or_legacy("TECHLIT_EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL")
        if not embedding_model and legacy_key and not using_new_embedding_config:
            embedding_model = "text-embedding-3-small"

        raw_dimensions = _new_or_legacy(
            "TECHLIT_EMBEDDING_DIMENSIONS", "OPENAI_EMBEDDING_DIMENSIONS"
        )
        if raw_dimensions:
            try:
                dimensions: int | None = int(raw_dimensions)
            except ValueError as exc:
                raise ValueError("TECHLIT_EMBEDDING_DIMENSIONS 必须是正整数") from exc
            if dimensions <= 0:
                raise ValueError("TECHLIT_EMBEDDING_DIMENSIONS 必须是正整数")
        else:
            dimensions = 512 if legacy_key and not using_new_embedding_config else None

        return cls(
            project_root=root,
            database_path=db_path,
            llm=LLMSettings(
                api_key=llm_key,
                base_url=llm_base_url,
                model=llm_model or "",
                api_mode=api_mode,
            ),
            embedding=EndpointSettings(
                api_key=embedding_key,
                base_url=embedding_base_url,
                model=embedding_model or "",
            ),
            embedding_dimensions=dimensions,
            embedding_provider=provider,
            local_model_path=model_path,
            local_min_score=local_min_score,
            local_model_name=local_model_name,
        )
