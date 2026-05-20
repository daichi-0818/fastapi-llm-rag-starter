from __future__ import annotations

from fastapi import FastAPI

from app.routers import chat, ingest, search

app = FastAPI(
    title="fastapi-llm-rag-starter",
    description="Production-ready FastAPI + RAG starter with pluggable LLM providers.",
    version="0.1.0",
)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(chat.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
