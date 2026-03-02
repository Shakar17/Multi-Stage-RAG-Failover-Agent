"""
FastAPI Application
-------------------
Exposes three endpoints:

  POST /ingest   – index document chunks into FAISS
  POST /query    – run the multi-stage RAG failover pipeline
  GET  /health   – liveness probe

Dependency injection via FastAPI's `Depends` keeps the app testable.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Logging setup (must happen before any imports that call getLogger) ─────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from app.config import settings
from app.controller import RAGFailoverController
from app.embedding import get_embedding_service
from app.llm_client import LLMClient
from app.models import IngestRequest, IngestResponse, QueryRequest, RAGResponse
from app.vector_store import VectorStore

# ── Singletons ────────────────────────────────────────────────────────────────

_embedder = get_embedding_service(settings.embedding_model)
_vector_store = VectorStore(
    dim=settings.embedding_dim,
    index_path=settings.faiss_index_path,
    meta_path=settings.faiss_meta_path,
)
_llm_client = LLMClient()
_controller = RAGFailoverController(
    vector_store=_vector_store,
    embedder=_embedder,
    llm_client=_llm_client,
)

# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Stage RAG Failover Agent",
    description=(
        "Production-ready retrieval pipeline with 5-stage failover, "
        "FAISS vector store, hybrid search, LLM query expansion, "
        "and strict no-hallucination synthesis."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ── Dependency injection helpers ──────────────────────────────────────────────

def get_controller() -> RAGFailoverController:
    return _controller


def get_vector_store() -> VectorStore:
    return _vector_store


def get_embedder() -> ...:
    return _embedder


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    """Liveness probe. Returns vector store stats."""
    return {
        "status": "ok",
        "total_chunks": _vector_store.total_chunks,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
    }


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(
    payload: IngestRequest,
    vs: VectorStore = Depends(get_vector_store),
    embedder=Depends(get_embedder),
):
    """
    Embed and index a batch of document chunks.

    Each chunk must have: chunk_id, text, source.
    Duplicate chunk_ids are allowed (the store will contain multiple copies).
    """
    if not payload.chunks:
        raise HTTPException(status_code=422, detail="No chunks provided.")

    texts = [c.text for c in payload.chunks]
    # Assign auto-generated IDs if empty strings were sent
    for c in payload.chunks:
        if not c.chunk_id:
            c.chunk_id = str(uuid.uuid4())

    logger.info("Ingesting %d chunks…", len(texts))
    embeddings: np.ndarray = embedder.embed_batch(texts)
    count = vs.add_chunks(payload.chunks, embeddings)

    return IngestResponse(indexed=count, message=f"Successfully indexed {count} chunk(s).")


@app.post("/query", response_model=RAGResponse, tags=["retrieval"])
async def query_endpoint(
    payload: QueryRequest,
    ctrl: RAGFailoverController = Depends(get_controller),
):
    """
    Run the multi-stage RAG failover pipeline.

    Response JSON shape:
    ```json
    {
      "stage_used":        "stage1_high_threshold",
      "confidence":        0.912,
      "answer":            "...",
      "chunks_used":       3,
      "expanded_queries":  [],
      "latency_ms":        42.1
    }
    ```
    """
    logger.info("Received query: %r", payload.query[:120])
    response = await ctrl.query(
        query=payload.query,
        top_k_override=payload.top_k_override,
    )
    logger.info(
        "Query answered by %s | confidence=%.4f | latency=%.1fms",
        response.stage_used, response.confidence, response.latency_ms or 0,
    )
    return response
