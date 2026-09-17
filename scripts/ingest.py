#!/usr/bin/env python3
"""CLI: build/refresh the vector index from data/sample_docs without
starting the API server.

Usage:
    python scripts/ingest.py [--dir data/sample_docs]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.services.embeddings import build_embedding_provider  # noqa: E402
from app.services.llm_provider import build_llm_provider  # noqa: E402
from app.services.rag_pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/sample_docs")
    args = parser.parse_args()

    embedding_provider = build_embedding_provider(settings.embedding_provider)
    llm_provider = build_llm_provider("mock", model=settings.anthropic_model)  # not used for ingest
    pipeline = RAGPipeline(embedding_provider, llm_provider, settings, settings.vector_store)

    docs, chunks = pipeline.ingest_directory(args.dir)
    print(f"Ingested {docs} document(s) into {chunks} chunk(s).")
    print(f"Index persisted to {settings.vector_store_path}")


if __name__ == "__main__":
    main()
