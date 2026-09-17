"""Orchestrates retrieval + grounded generation.

The core design goal, carried over from the thesis work this project
grew out of: never let the model answer past what the evidence supports.
If nothing relevant clears MIN_SIMILARITY, the pipeline refuses to answer
rather than letting the LLM improvise — this is what "grounded" means in
practice, not just a marketing word in the project title.
"""
from __future__ import annotations

from app.config import Settings
from app.models.schemas import Citation, QueryResponse
from app.services.embeddings import EmbeddingProvider
from app.services.ingestion import chunk_text, load_documents
from app.services.llm_provider import LLMProvider
from app.services.vector_store import VectorStore, build_vector_store

SYSTEM_PROMPT = (
    "You are an enterprise knowledge-base assistant. Answer ONLY using the "
    "numbered context passages provided. Every claim in your answer must be "
    "traceable to at least one passage. If the passages do not contain enough "
    "information to answer, say so explicitly instead of guessing. Cite "
    "passages inline using their [n] marker."
)


class RAGPipeline:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        settings: Settings,
        vector_store_kind: str,
    ):
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.settings = settings
        self.vector_store_kind = vector_store_kind
        self.vector_store: VectorStore | None = None
        self._try_load_persisted()

    def _try_load_persisted(self) -> None:
        """Best-effort: if a previously ingested index exists on disk, load it
        so the API doesn't start out empty after a restart."""
        loaded = self.embedding_provider.load(self.settings.vector_store_path)
        if loaded and self.embedding_provider.dimension:
            self.vector_store = build_vector_store(self.vector_store_kind, self.embedding_provider.dimension)
            self.vector_store.load(self.settings.vector_store_path)

    def index_size(self) -> int:
        return len(self.vector_store) if self.vector_store is not None else 0

    def ingest_directory(self, directory: str) -> tuple[int, int]:
        documents = load_documents(directory)
        all_chunks = []
        for source, text in documents:
            all_chunks.extend(
                chunk_text(text, source, self.settings.chunk_size, self.settings.chunk_overlap)
            )
        if not all_chunks:
            return (len(documents), 0)

        texts = [c.text for c in all_chunks]
        self.embedding_provider.fit(texts)
        vectors = self.embedding_provider.embed(texts)

        self.vector_store = build_vector_store(self.vector_store_kind, self.embedding_provider.dimension)
        ids = [c.chunk_id for c in all_chunks]
        metadatas = [{"source": c.source, "text": c.text} for c in all_chunks]
        self.vector_store.add(ids, vectors, metadatas)

        self.vector_store.persist(self.settings.vector_store_path)
        self.embedding_provider.persist(self.settings.vector_store_path)
        return (len(documents), len(all_chunks))

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        if self.vector_store is None or len(self.vector_store) == 0:
            return QueryResponse(
                answer="The knowledge base is empty. Call POST /ingest first.",
                grounded=False,
                citations=[],
            )

        top_k = top_k or self.settings.top_k
        query_vector = self.embedding_provider.embed_one(question)
        results = self.vector_store.search(query_vector, top_k=top_k)

        grounded_results = [r for r in results if r["score"] >= self.settings.min_similarity]

        if not grounded_results:
            return QueryResponse(
                answer=(
                    "I don't have grounded information in the knowledge base to answer that. "
                    "Try rephrasing, or this may be outside the ingested documents' scope."
                ),
                grounded=False,
                citations=[],
            )

        context_block = "\n\n".join(
            f"[{i+1}] (source: {r['metadata']['source']})\n{r['metadata']['text']}"
            for i, r in enumerate(grounded_results)
        )
        user_prompt = f"Context passages:\n\n{context_block}\n\nQuestion: {question}"
        answer = self.llm_provider.generate(SYSTEM_PROMPT, user_prompt)

        citations = [
            Citation(
                source=r["metadata"]["source"],
                chunk_id=r["id"],
                score=round(r["score"], 4),
                snippet=r["metadata"]["text"][:200],
            )
            for r in grounded_results
        ]
        return QueryResponse(answer=answer, grounded=True, citations=citations)
