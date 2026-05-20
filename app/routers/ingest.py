from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import get_embeddings, get_vector_store
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/ingest", tags=["ingest"])


class Document(BaseModel):
    id: str | None = None
    text: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[Document] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    ingested: int
    ids: list[str]
    total: int


@router.post("", response_model=IngestResponse)
def ingest(
    payload: IngestRequest,
    store: VectorStore = Depends(get_vector_store),
    embeddings: EmbeddingProvider = Depends(get_embeddings),
) -> IngestResponse:
    ids = [d.id or str(uuid.uuid4()) for d in payload.documents]
    texts = [d.text for d in payload.documents]
    metas = [d.metadata for d in payload.documents]
    vectors = embeddings.embed(texts)
    store.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    return IngestResponse(ingested=len(ids), ids=ids, total=store.count())
