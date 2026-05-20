from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_embeddings, get_llm, get_vector_store
from app.main import app
from app.services.embeddings import FakeEmbeddings
from app.services.vector_store import VectorStore


class StubLLM:
    """Echoes the last context line. Lets us assert RAG plumbing without API keys."""

    def complete(self, system: str, user: str) -> str:
        return f"STUB_ANSWER::{user[-60:]}"


@pytest.fixture
def tmp_vector_store() -> Iterator[VectorStore]:
    tmpdir = tempfile.mkdtemp(prefix="chroma-test-")
    store = VectorStore(persist_dir=tmpdir, collection="test")
    yield store
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def client(tmp_vector_store: VectorStore) -> Iterator[TestClient]:
    app.dependency_overrides[get_vector_store] = lambda: tmp_vector_store
    app.dependency_overrides[get_embeddings] = lambda: FakeEmbeddings()
    app.dependency_overrides[get_llm] = lambda: StubLLM()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
