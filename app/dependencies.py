"""FastAPI dependency wiring. Tests override these with fakes."""

from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.services.embeddings import EmbeddingProvider, build_embeddings
from app.services.llm import LLMProvider, build_llm
from app.services.vector_store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(settings.chroma_persist_dir, settings.chroma_collection)


def get_embeddings() -> EmbeddingProvider:
    return build_embeddings(get_settings())


def get_llm() -> LLMProvider:
    return build_llm(get_settings())


def get_app_settings() -> Settings:
    return get_settings()
