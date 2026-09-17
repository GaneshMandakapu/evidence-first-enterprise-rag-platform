"""Document loading and chunking.

Deliberately simple (paragraph-aware sliding window over plain text /
markdown) so the pipeline stays dependency-light. Swappable for a
production loader (PDF/Office parsing, Azure Blob Storage, SharePoint
connectors, etc.) without touching embeddings, retrieval or generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Greedily pack paragraphs into ~chunk_size-character windows with
    a character overlap between consecutive chunks, so a fact split
    across a paragraph boundary is still retrievable."""
    paragraphs = _split_paragraphs(text)
    chunks: list[Chunk] = []
    buffer = ""
    idx = 0

    def flush(buf: str):
        nonlocal idx
        if buf.strip():
            chunks.append(Chunk(chunk_id=f"{Path(source).stem}-{idx}", source=source, text=buf.strip()))
            idx += 1

    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= chunk_size:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            flush(buffer)
            tail = buffer[-overlap:] if overlap and buffer else ""
            buffer = f"{tail}\n\n{para}".strip() if tail else para
    flush(buffer)
    return chunks


def load_documents(directory: str) -> list[tuple[str, str]]:
    """Return list of (source_name, raw_text) for every .md/.txt file in directory."""
    docs = []
    for path in sorted(Path(directory).glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs
