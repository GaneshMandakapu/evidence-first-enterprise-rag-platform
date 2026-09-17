"""End-to-end pipeline tests using MockProvider — no API key required,
runnable in CI. Mirrors the grounding discipline from the thesis this
project grew out of: verify the refusal path, not just the happy path."""
from __future__ import annotations

from app.config import Settings
from app.services.embeddings import build_embedding_provider
from app.services.llm_provider import build_llm_provider
from app.services.rag_pipeline import RAGPipeline


def _make_pipeline(tmp_path, min_similarity: float = 0.05) -> RAGPipeline:
    settings = Settings(
        llm_provider="mock",
        vector_store="faiss",
        embedding_provider="tfidf",
        min_similarity=min_similarity,
        top_k=3,
        vector_store_path=str(tmp_path / "index"),
    )
    embedding_provider = build_embedding_provider(settings.embedding_provider)
    llm_provider = build_llm_provider("mock", model="unused")
    pipeline = RAGPipeline(embedding_provider, llm_provider, settings, settings.vector_store)
    return pipeline


def test_chunking_respects_paragraph_boundaries():
    from app.services.ingestion import chunk_text

    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_text(text, source="doc.md", chunk_size=1000, overlap=0)
    assert len(chunks) == 1
    assert "Para one." in chunks[0].text
    assert "Para three." in chunks[0].text


def test_query_returns_grounded_answer_with_citations(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    docs, chunks = pipeline.ingest_directory("data/sample_docs")
    assert docs == 4
    assert chunks > 0

    response = pipeline.query("How many days per week can I work remotely?")
    assert response.grounded is True
    assert len(response.citations) > 0
    assert response.citations[0].source == "hr_policy.md"


def test_query_refuses_when_ungrounded(tmp_path):
    pipeline = _make_pipeline(tmp_path, min_similarity=0.999)
    pipeline.ingest_directory("data/sample_docs")

    response = pipeline.query("What is the capital of France?")
    assert response.grounded is False
    assert response.citations == []
    assert "don't have grounded information" in response.answer


def test_query_before_ingest_reports_empty_index(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    response = pipeline.query("Anything?")
    assert response.grounded is False
    assert "empty" in response.answer.lower()
