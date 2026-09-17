"""Pydantic request/response contracts for the API."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, description="Natural-language question about the knowledge base."
    )
    top_k: int | None = Field(None, ge=1, le=20, description="Override the number of chunks retrieved.")


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[Citation]


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    llm_provider: str
    index_size: int
