"""Embedding provider abstraction.

TfidfEmbeddingProvider is the default: no heavy ML dependencies (no
torch/transformers), installs in seconds, and mirrors the TF-IDF
lexical-similarity mode used by the backlog-coherence detector in the
thesis this project grew out of. Good enough for a knowledge base this
size, and it keeps local setup and CI fast.

SentenceTransformerEmbeddingProvider is an optional, stronger upgrade
path (true semantic embeddings) for a larger or more lexically diverse
corpus. Install it separately (see requirements-embeddings-extra.txt);
swapping it in is a one-line config change (EMBEDDING_PROVIDER=
sentence_transformer in .env) — nothing else in the pipeline changes.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingProvider(ABC):
    dimension: int = 0

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Fit on the corpus being ingested (no-op for pretrained models)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @abstractmethod
    def persist(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> bool:
        """Return True if a fitted/loadable state was found, else False."""


class TfidfEmbeddingProvider(EmbeddingProvider):
    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), stop_words="english"
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True
        self.dimension = len(self._vectorizer.vocabulary_)

    def embed(self, texts: list[str]) -> np.ndarray:
        from sklearn.preprocessing import normalize

        if not self._fitted:
            raise RuntimeError("fit() must be called (via /ingest) before embed().")
        dense = self._vectorizer.transform(texts).toarray().astype("float32")
        return normalize(dense)  # L2-normalised so inner product == cosine similarity

    def persist(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "vectorizer.pkl", "wb") as f:
            pickle.dump(self._vectorizer, f)

    def load(self, path: str) -> bool:
        vec_file = Path(path) / "vectorizer.pkl"
        if not vec_file.exists():
            return False
        with open(vec_file, "rb") as f:
            self._vectorizer = pickle.load(f)
        self._fitted = True
        self.dimension = len(self._vectorizer.vocabulary_)
        return True


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Optional upgrade path. Requires:
    pip install -r requirements-embeddings-extra.txt"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: "
                "pip install -r requirements-embeddings-extra.txt"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def fit(self, texts: list[str]) -> None:
        pass  # pretrained model — nothing to fit

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    def persist(self, path: str) -> None:
        pass  # nothing beyond the model name to persist

    def load(self, path: str) -> bool:
        return True  # pretrained model is always "ready"


def build_embedding_provider(kind: str) -> EmbeddingProvider:
    if kind == "tfidf":
        return TfidfEmbeddingProvider()
    if kind == "sentence_transformer":
        return SentenceTransformerEmbeddingProvider()
    raise ValueError(f"Unknown embedding provider: {kind}")
