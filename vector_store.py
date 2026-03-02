"""
Vector Store
------------
FAISS-backed store with:
  * Cosine similarity search (IndexFlatIP on L2-normalised vectors)
  * BM25-style keyword overlap fallback for hybrid scoring
  * Thread-safe read/write with a RW-lock

Metadata (chunk text, source, etc.) is stored in a parallel in-memory list
serialised to JSON so no additional database is needed for dev/test.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)

# ── Keyword scoring helpers ───────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lowercase alphabetic tokens (poor-man's BM25 base)."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _keyword_score(query_tokens: set[str], doc_text: str) -> float:
    """Jaccard overlap between query tokens and document tokens."""
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thread-safe FAISS vector store with hybrid semantic + keyword search.

    Parameters
    ----------
    dim : int
        Embedding dimension (must match the embedding model).
    index_path : str
        File path for persisting the FAISS index.
    meta_path : str
        File path for persisting chunk metadata (JSON).
    hybrid_alpha : float
        Weight [0, 1] for blending semantic (alpha) vs keyword (1-alpha) scores.
    """

    def __init__(
        self,
        dim: int,
        index_path: str = "data/faiss.index",
        meta_path: str = "data/faiss_meta.json",
        hybrid_alpha: float = 0.8,
    ) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            ) from exc

        import faiss

        self._dim = dim
        self._index_path = index_path
        self._meta_path = meta_path
        self._hybrid_alpha = hybrid_alpha

        # Metadata parallel to FAISS index rows
        self._meta: List[Dict] = []

        # RW lock: concurrent reads are fine, writes are exclusive
        self._lock = threading.RLock()

        # Build or load FAISS index
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self._load()
        else:
            self._index = faiss.IndexFlatIP(dim)   # inner-product == cosine on L2-normed vecs
            logger.info("Initialised fresh FAISS IndexFlatIP (dim=%d)", dim)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add document chunks and their pre-computed embeddings.

        Parameters
        ----------
        chunks : list of DocumentChunk
        embeddings : (N, dim) float32 array

        Returns
        -------
        int : number of vectors added
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have the same length")

        with self._lock:
            self._index.add(embeddings)
            for chunk in chunks:
                self._meta.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "metadata": chunk.metadata,
                })
            self._save()

        logger.info("Indexed %d chunks. Total: %d", len(chunks), self._index.ntotal)
        return len(chunks)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        query_text: str,
        top_k: int = 5,
        threshold: float = 0.60,
        use_hybrid: bool = True,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks above `threshold`.

        Scoring:
          hybrid_score = alpha * semantic_score + (1-alpha) * keyword_score

        Parameters
        ----------
        query_vec  : (dim,) float32 – normalised query embedding
        query_text : raw query string for keyword scoring
        top_k      : how many candidates to retrieve from FAISS
        threshold  : minimum score to include a chunk
        use_hybrid : whether to blend in keyword score

        Returns
        -------
        List[RetrievedChunk] sorted by score descending
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []

            # Retrieve 2× top_k from FAISS so hybrid re-ranking has room
            k = min(top_k * 2, self._index.ntotal)
            scores, indices = self._index.search(
                query_vec.reshape(1, -1), k
            )

        scores = scores[0]       # shape (k,)
        indices = indices[0]     # shape (k,)

        query_tokens = _tokenize(query_text)
        results: List[RetrievedChunk] = []

        for raw_score, idx in zip(scores, indices):
            if idx < 0:          # FAISS padding
                continue

            meta = self._meta[idx]
            semantic_score = float(raw_score)

            if use_hybrid:
                kw_score = _keyword_score(query_tokens, meta["text"])
                final_score = (
                    self._hybrid_alpha * semantic_score
                    + (1 - self._hybrid_alpha) * kw_score
                )
            else:
                final_score = semantic_score

            if final_score >= threshold:
                results.append(
                    RetrievedChunk(
                        chunk_id=meta["chunk_id"],
                        text=meta["text"],
                        source=meta["source"],
                        score=round(final_score, 4),
                        metadata=meta.get("metadata", {}),
                    )
                )

        # Sort by score descending, truncate to top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        import faiss
        os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)
        faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)
        logger.debug("Persisted FAISS index (%d vectors)", self._index.ntotal)

    def _load(self) -> None:
        """Load FAISS index and metadata from disk."""
        import faiss
        self._index = faiss.read_index(self._index_path)
        with open(self._meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)
        logger.info(
            "Loaded FAISS index: %d vectors, %d meta records",
            self._index.ntotal, len(self._meta)
        )

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal
