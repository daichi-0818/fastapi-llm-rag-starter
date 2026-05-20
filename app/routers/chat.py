from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import get_embeddings, get_llm, get_vector_store
from app.services.embeddings import EmbeddingProvider
from app.services.llm import SYSTEM_PROMPT, LLMProvider, build_user_prompt
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)


class ChatCitation(BaseModel):
    id: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[ChatCitation]


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    store: VectorStore = Depends(get_vector_store),
    embeddings: EmbeddingProvider = Depends(get_embeddings),
    llm: LLMProvider = Depends(get_llm),
) -> ChatResponse:
    vector = embeddings.embed([payload.question])[0]
    hits = store.query(vector, top_k=payload.top_k)
    contexts = [h.text for h in hits]
    user_prompt = build_user_prompt(payload.question, contexts)
    answer = llm.complete(SYSTEM_PROMPT, user_prompt)
    citations = [
        ChatCitation(id=h.id, score=h.score, snippet=h.text[:200]) for h in hits
    ]
    return ChatResponse(answer=answer, citations=citations)
