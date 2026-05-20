"""Thin ChromaDB wrapper used by the RAG endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings


@dataclass
class SearchHit:
    id: str
    text: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(self, persist_dir: str, collection: str) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        # ChromaDB rejects empty dicts in metadatas; inject a placeholder
        # so callers can stay metadata-agnostic.
        cleaned = (
            [m if m else {"_": "_"} for m in metadatas]
            if metadatas is not None
            else [{"_": "_"} for _ in ids]
        )
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=cleaned,
        )

    def query(self, embedding: list[float], top_k: int = 4) -> list[SearchHit]:
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        dists = result.get("distances", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        hits: list[SearchHit] = []
        for i, doc, dist, meta in zip(ids, docs, dists, metas, strict=False):
            clean_meta = {k: v for k, v in (meta or {}).items() if k != "_"}
            hits.append(SearchHit(id=i, text=doc, score=1.0 - dist, metadata=clean_meta))
        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
