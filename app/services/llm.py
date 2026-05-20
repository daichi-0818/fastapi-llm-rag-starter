"""LLM provider abstraction (Anthropic Claude / Google Gemini)."""

from __future__ import annotations

from typing import Protocol

from app.config import Settings

SYSTEM_PROMPT = (
    "You are a concise RAG assistant. "
    "Answer using ONLY the provided context. "
    "If the context is insufficient, say so explicitly."
)


def build_user_prompt(question: str, contexts: list[str]) -> str:
    joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{joined}\n\nQuestion: {question}\n\nAnswer:"


class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicProvider:
    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()


class GoogleProvider:
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=f"{system}\n\n{user}",
        )
        return (response.text or "").strip()


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    if settings.llm_provider == "google":
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=google")
        return GoogleProvider(settings.google_api_key, settings.google_model)
    raise RuntimeError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
