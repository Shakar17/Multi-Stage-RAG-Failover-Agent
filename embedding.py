"""
Embedding Service
-----------------
Wraps sentence-transformers to produce L2-normalised embeddings suitable
for cosine-similarity comparisons inside FAISS.

Design decisions:
  * Singleton pattern so the model loads only once per process.
  * Returns numpy float32 arrays for direct FAISS compatibility.
  * Batch encode for efficiency during ingestion.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a single query string → (dim,) float32 array."""
        vec = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine sim via dot-product in FAISS
        )
        return vec[0].astype(np.float32)

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts → (N, dim) float32 array."""
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return vecs.astype(np.float32)


# ── Module-level singleton ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """Return a cached EmbeddingService instance (loaded once per process)."""
    return EmbeddingService(model_name)
