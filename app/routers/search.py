from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import get_embeddings, get_vector_store
from app.services.embeddings import EmbeddingProvider
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class SearchHitOut(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHitOut]


@router.post("", response_model=SearchResponse)
def search(
    payload: SearchRequest,
    store: VectorStore = Depends(get_vector_store),
    embeddings: EmbeddingProvider = Depends(get_embeddings),
) -> SearchResponse:
    vector = embeddings.embed([payload.query])[0]
    hits = store.query(vector, top_k=payload.top_k)
    return SearchResponse(
        query=payload.query,
        hits=[SearchHitOut(id=h.id, text=h.text, score=h.score, metadata=h.metadata) for h in hits],
    )
