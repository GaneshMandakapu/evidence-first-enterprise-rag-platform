"""FastAPI entrypoint — wires providers into the RAG pipeline and serves the demo."""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from app.services.embeddings import build_embedding_provider
from app.services.llm_provider import build_llm_provider
from app.services.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Evidence-First Enterprise RAG Platform",
    description="Grounded document intelligence over an internal knowledge base.",
    version="0.1.0",
)

_embedding_provider = build_embedding_provider(settings.embedding_provider)
_llm_provider = build_llm_provider(
    settings.llm_provider, model=settings.anthropic_model, api_key=settings.anthropic_api_key
)
pipeline = RAGPipeline(_embedding_provider, _llm_provider, settings, settings.vector_store)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_store=settings.vector_store,
        llm_provider=settings.llm_provider,
        index_size=pipeline.index_size(),
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest(directory: str = "data/sample_docs") -> IngestResponse:
    docs, chunks = pipeline.ingest_directory(directory)
    return IngestResponse(documents_ingested=docs, chunks_indexed=chunks)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(request: QueryRequest) -> QueryResponse:
    return pipeline.query(request.question, top_k=request.top_k)


_static_directory = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_directory), name="static")
app.mount("/", StaticFiles(directory=_static_directory, html=True), name="demo")
