# fastapi-llm-rag-starter

Production-ready FastAPI + RAG starter with pluggable LLM providers (Anthropic Claude / Google Gemini), ChromaDB vector store, and a clean test setup that runs without API keys.

> **Why this exists**
> Most RAG examples are notebook-style and tied to a single LLM vendor.
> This starter shows a layout suitable for real services: a provider abstraction so you can swap Claude ⇄ Gemini per environment, a thin vector-store wrapper, FastAPI dependency injection for testability, and a stub-driven test suite that does not call paid APIs in CI.

## Features

- **FastAPI** app with three endpoints: `/ingest`, `/search`, `/chat`
- **Pluggable LLM**: `LLM_PROVIDER=anthropic` or `google` — same interface
- **Pluggable embeddings**: `google` (text-embedding-004) or `fake` (deterministic, no API)
- **ChromaDB** persistent vector store
- **pytest** with `dependency_overrides` — runs offline, no API keys needed
- **GitHub Actions** CI (ruff + pytest on Python 3.11 / 3.12)
- **Docker** image, single-command boot

## Quickstart

```bash
git clone https://github.com/daichi-0818/fastapi-llm-rag-starter.git
cd fastapi-llm-rag-starter

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# default: EMBEDDINGS_PROVIDER=fake → no API key needed for local exploration
# set ANTHROPIC_API_KEY or GOOGLE_API_KEY to enable /chat

uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for Swagger UI.

### End-to-end smoke test

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"text":"FastAPI is a modern Python web framework"},
                    {"text":"ChromaDB stores vector embeddings"}]}'

curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"python web","top_k":2}'
```

### Run tests

```bash
pytest -q
```

All eight tests run offline using `FakeEmbeddings` and a `StubLLM` injected via FastAPI `dependency_overrides`.

## Architecture

```
app/
├── main.py              FastAPI entrypoint, mounts routers
├── config.py            pydantic-settings (.env aware)
├── dependencies.py      DI wiring — overridable from tests
├── routers/
│   ├── ingest.py        POST /ingest    → upsert docs
│   ├── search.py        POST /search    → semantic top-k
│   └── chat.py          POST /chat      → RAG answer + citations
└── services/
    ├── embeddings.py    Google / Fake providers (Protocol)
    ├── vector_store.py  ChromaDB wrapper
    └── llm.py           Claude / Gemini providers (Protocol)
```

### Design choices

- **Provider abstraction via `typing.Protocol`** — no inheritance, no metaclass tricks. Easy to add a new provider.
- **Fake embeddings backend** — deterministic SHA-512-based vectors. Enables CI without paid APIs and offline development.
- **Empty-dict metadata placeholder** — ChromaDB ≥ 1.5 rejects `{}` in `metadatas`; the wrapper injects/strips an internal placeholder so callers stay metadata-agnostic.
- **`dependency_overrides` in tests** — FastAPI's native pattern. No global monkey patching.

## Docker

```bash
docker build -t fastapi-llm-rag-starter .
docker run --rm -p 8000:8000 --env-file .env fastapi-llm-rag-starter
```

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `google` |
| `ANTHROPIC_API_KEY` | — | Required if `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | |
| `GOOGLE_API_KEY` | — | Required if `LLM_PROVIDER=google` or `EMBEDDINGS_PROVIDER=google` |
| `GOOGLE_MODEL` | `gemini-2.5-flash` | |
| `EMBEDDINGS_PROVIDER` | `fake` | `google` (text-embedding-004) or `fake` (no API) |
| `CHROMA_PERSIST_DIR` | `./.chroma` | |
| `CHROMA_COLLECTION` | `documents` | |

## License

MIT
