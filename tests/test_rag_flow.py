from fastapi.testclient import TestClient


def test_ingest_search_chat_flow(client: TestClient) -> None:
    docs = {
        "documents": [
            {"id": "a", "text": "FastAPI is a modern Python web framework.", "metadata": {"src": "doc1"}},
            {"id": "b", "text": "ChromaDB stores embeddings for semantic search.", "metadata": {"src": "doc2"}},
            {"id": "c", "text": "Claude and Gemini are LLM APIs from Anthropic and Google.", "metadata": {"src": "doc3"}},
        ]
    }
    r = client.post("/ingest", json=docs)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ingested"] == 3
    assert body["total"] >= 3

    r = client.post("/search", json={"query": "what is fastapi?", "top_k": 2})
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert len(hits) == 2
    assert all(set(h.keys()) == {"id", "text", "score", "metadata"} for h in hits)

    r = client.post("/chat", json={"question": "summarize fastapi", "top_k": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"].startswith("STUB_ANSWER::")
    assert len(body["citations"]) == 2


def test_ingest_validates_empty(client: TestClient) -> None:
    r = client.post("/ingest", json={"documents": []})
    assert r.status_code == 422


def test_search_validates_top_k_bounds(client: TestClient) -> None:
    r = client.post("/search", json={"query": "x", "top_k": 0})
    assert r.status_code == 422
    r = client.post("/search", json={"query": "x", "top_k": 99})
    assert r.status_code == 422
