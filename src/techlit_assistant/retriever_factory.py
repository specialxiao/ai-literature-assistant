from __future__ import annotations

from .config import Settings
from .database import Repository
from .retrieval import CompatibleEmbedder, EmbeddingRetriever


def build_semantic_retriever(
    repository: Repository, settings: Settings, provider: str | None = None
) -> EmbeddingRetriever | None:
    if (provider or settings.embedding_provider) == "local":
        # Importing this module does not load torch, instantiate or download a model.
        from .local_embedding import LocalEmbedder

        return EmbeddingRetriever(
            repository,
            LocalEmbedder(settings.local_model_path, settings.local_model_name),
            settings.local_min_score,
        )
    if not settings.embedding.configured:
        return None
    return EmbeddingRetriever(
        repository,
        CompatibleEmbedder(
            settings.embedding.api_key,
            settings.embedding.model,
            settings.embedding_dimensions,
            settings.embedding.base_url,
        ),
    )
