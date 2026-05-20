from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["anthropic", "google"] = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"
    google_api_key: str = ""
    google_model: str = "gemini-2.5-flash"

    embeddings_provider: Literal["google", "fake"] = "fake"
    embeddings_model: str = "text-embedding-004"

    chroma_persist_dir: str = "./.chroma"
    chroma_collection: str = "documents"

    app_host: str = "0.0.0.0"
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
