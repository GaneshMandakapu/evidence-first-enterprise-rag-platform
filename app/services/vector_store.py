"""Vector store abstraction.

FAISSVectorStore is the concrete, working implementation used by this
demo (local dev / CI, no cloud resources required).

AzureAISearchVectorStore is a production adapter *stub*: it documents the
interface a real enterprise deployment would implement against Azure AI
Search (hybrid keyword+vector search, managed scaling, RBAC-scoped
indexes) so the rest of the pipeline (ingestion, retrieval, generation)
does not need to change when swapping the backing store. It intentionally
raises NotImplementedError — no Azure resources are provisioned for this
portfolio build.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]: ...

    @abstractmethod
    def persist(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> bool:
        """Return True if an index was found and loaded, else False."""

    @abstractmethod
    def __len__(self) -> int: ...


class FAISSVectorStore(VectorStore):
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity via normalised vectors
        self.ids: list[str] = []
        self.metadatas: list[dict] = []

    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        self.index.add(vectors)
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        if len(self.ids) == 0:
            return []
        query_vector = np.asarray([query_vector], dtype="float32")
        scores, indices = self.index.search(query_vector, min(top_k, len(self.ids)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {"id": self.ids[idx], "score": float(score), "metadata": self.metadatas[idx]}
            )
        return results

    def persist(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "meta.json", "w") as f:
            json.dump({"ids": self.ids, "metadatas": self.metadatas, "dimension": self.dimension}, f)

    def load(self, path: str) -> bool:
        p = Path(path)
        index_file, meta_file = p / "index.faiss", p / "meta.json"
        if not (index_file.exists() and meta_file.exists()):
            return False
        self.index = faiss.read_index(str(index_file))
        with open(meta_file) as f:
            data = json.load(f)
        self.ids = data["ids"]
        self.metadatas = data["metadatas"]
        self.dimension = data["dimension"]
        return True

    def __len__(self) -> int:
        return len(self.ids)


class AzureAISearchVectorStore(VectorStore):
    """Production adapter stub — see module docstring."""

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Provision an Azure AI Search index and implement this adapter using "
            "azure-search-documents. See README 'Production deployment' section."
        )

    def add(self, ids, vectors, metadatas):
        raise NotImplementedError

    def search(self, query_vector, top_k=4):
        raise NotImplementedError

    def persist(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def build_vector_store(kind: str, dimension: int) -> VectorStore:
    if kind == "faiss":
        return FAISSVectorStore(dimension)
    if kind == "azure_ai_search":
        return AzureAISearchVectorStore()
    raise ValueError(f"Unknown vector store backend: {kind}")
