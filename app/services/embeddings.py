"""Embedding provider abstraction.

Two backends:
- ``google``: uses Google's text-embedding-004 via ``google-genai``.
- ``fake``: deterministic hash-based vectors. Lets tests and demos run without API keys.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from app.config import Settings


class EmbeddingProvider(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbeddings:
    """Deterministic embeddings derived from SHA-512 of the input.

    Not semantically meaningful, but stable enough for tests and offline demos.
    """

    dimension = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha512(text.encode("utf-8")).digest()
            vec = [(b - 128) / 128.0 for b in digest[: self.dimension]]
            out.append(vec)
        return out


class GoogleEmbeddings:
    dimension = 768

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.models.embed_content(model=self._model, contents=texts)
        return [e.values for e in result.embeddings]


def build_embeddings(settings: Settings) -> EmbeddingProvider:
    if settings.embeddings_provider == "google":
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required when EMBEDDINGS_PROVIDER=google")
        return GoogleEmbeddings(settings.google_api_key, settings.embeddings_model)
    return FakeEmbeddings()
