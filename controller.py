"""
RAG Failover Controller
-----------------------
Implements the deterministic 5-stage failover pipeline:

  Stage 1 ── High-threshold strict retrieval  (≥ 0.85)
      │ fail
      ▼
  Stage 2 ── Broad fallback search            (≥ 0.60, larger top-k, hybrid)
      │ fail
      ▼
  Stage 3 ── LLM query expansion + re-retrieval (≥ 0.55)
      │ fail
      ▼
  Stage 4 ── Context-only synthesis with retrieved chunks
             (if no chunks survived any stage → refusal message inside synthesis)
      │ synthesis returns refusal phrase
      ▼
  Stage 5 ── Hard refusal: "Sorry, this question is outside the supported domain."

Every stage logs its outcome so operators can trace exactly which path answered.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from app.config import settings
from app.embedding import EmbeddingService
from app.llm_client import LLMClient
from app.models import RAGResponse, RAGStage, RetrievedChunk
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Sentinel phrase that the LLM returns when context is insufficient
_INSUFFICIENT_CONTEXT_PHRASE = "I don't have enough information in the knowledge base."

# Final refusal message
_FINAL_REFUSAL = "Sorry, this question is outside the supported domain."


class RAGFailoverController:
    """
    Orchestrates all five retrieval + synthesis stages.

    Parameters
    ----------
    vector_store   : VectorStore instance (FAISS-backed)
    embedder       : EmbeddingService instance
    llm_client     : LLMClient instance
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingService,
        llm_client: LLMClient,
    ) -> None:
        self._vs = vector_store
        self._emb = embedder
        self._llm = llm_client

    # ── Public entry point ────────────────────────────────────────────────────

    async def query(self, query: str, top_k_override: Optional[int] = None) -> RAGResponse:
        """
        Run the full failover pipeline and return a structured RAGResponse.

        Deterministic execution order: Stage 1 → 2 → 3 → 4 → 5.
        """
        t0 = time.perf_counter()
        query_vec = self._emb.embed_query(query)

        # ── Stage 1: High-threshold retrieval ─────────────────────────────────
        chunks1 = self._retrieve(
            query_vec, query,
            top_k=top_k_override or settings.stage1_top_k,
            threshold=settings.stage1_threshold,
            use_hybrid=False,           # strict semantic only at stage 1
        )
        if chunks1:
            best_score = chunks1[0].score
            logger.info(
                "[STAGE 1] ✓ Found %d chunk(s). Best score=%.4f",
                len(chunks1), best_score,
            )
            answer = await self._synthesize(query, chunks1)
            if not self._is_refusal(answer):
                return self._build_response(
                    stage=RAGStage.STAGE1_HIGH_THRESHOLD,
                    answer=answer,
                    chunks=chunks1,
                    confidence=best_score,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        logger.info("[STAGE 1] ✗ No chunks above threshold=%.2f", settings.stage1_threshold)

        # ── Stage 2: Broad fallback with hybrid search ─────────────────────────
        chunks2 = self._retrieve(
            query_vec, query,
            top_k=top_k_override or settings.stage2_top_k,
            threshold=settings.stage2_threshold,
            use_hybrid=True,            # semantic + keyword at stage 2
        )
        if chunks2:
            best_score = chunks2[0].score
            logger.info(
                "[STAGE 2] ✓ Found %d chunk(s). Best score=%.4f",
                len(chunks2), best_score,
            )
            answer = await self._synthesize(query, chunks2)
            if not self._is_refusal(answer):
                return self._build_response(
                    stage=RAGStage.STAGE2_BROAD_FALLBACK,
                    answer=answer,
                    chunks=chunks2,
                    confidence=best_score,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        logger.info("[STAGE 2] ✗ No chunks above threshold=%.2f", settings.stage2_threshold)

        # ── Stage 3: LLM query expansion + re-retrieval ────────────────────────
        expanded_queries = await self._llm.expand_query(query)
        logger.info("[STAGE 3] Expanded into %d queries: %s", len(expanded_queries), expanded_queries)

        best_chunks3: List[RetrievedChunk] = []
        best_score3 = 0.0

        for eq in expanded_queries:
            eq_vec = self._emb.embed_query(eq)
            eq_chunks = self._retrieve(
                eq_vec, eq,
                top_k=top_k_override or settings.stage3_top_k,
                threshold=settings.stage3_threshold,
                use_hybrid=True,
            )
            if eq_chunks and eq_chunks[0].score > best_score3:
                best_score3 = eq_chunks[0].score
                best_chunks3 = eq_chunks

        if best_chunks3:
            logger.info(
                "[STAGE 3] ✓ Found %d chunk(s) via expansion. Best score=%.4f",
                len(best_chunks3), best_score3,
            )
            answer = await self._synthesize(query, best_chunks3)
            if not self._is_refusal(answer):
                return self._build_response(
                    stage=RAGStage.STAGE3_QUERY_EXPANSION,
                    answer=answer,
                    chunks=best_chunks3,
                    confidence=best_score3,
                    expanded_queries=expanded_queries,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        logger.info("[STAGE 3] ✗ Query expansion did not yield sufficient chunks.")

        # ── Stage 4: Context-only synthesis with whatever we have ─────────────
        # Collect any chunks we found (even if below threshold at prior stages)
        fallback_chunks = self._retrieve(
            query_vec, query,
            top_k=settings.stage2_top_k,
            threshold=0.30,             # very low bar – gather whatever exists
            use_hybrid=True,
        )

        if fallback_chunks:
            answer = await self._synthesize(query, fallback_chunks)
            if not self._is_refusal(answer):
                logger.info(
                    "[STAGE 4] ✓ Context synthesis answered with %d low-confidence chunk(s).",
                    len(fallback_chunks),
                )
                return self._build_response(
                    stage=RAGStage.STAGE4_CONTEXT_SYNTHESIS,
                    answer=answer,
                    chunks=fallback_chunks,
                    confidence=fallback_chunks[0].score,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            # LLM itself decided context was insufficient
            logger.info("[STAGE 4] LLM indicated insufficient context.")
            return self._build_response(
                stage=RAGStage.STAGE4_CONTEXT_SYNTHESIS,
                answer=_INSUFFICIENT_CONTEXT_PHRASE,
                chunks=[],
                confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # ── Stage 5: Final refusal ────────────────────────────────────────────
        logger.info("[STAGE 5] ✗ All stages exhausted. Returning hard refusal.")
        return self._build_response(
            stage=RAGStage.STAGE5_REFUSAL,
            answer=_FINAL_REFUSAL,
            chunks=[],
            confidence=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _retrieve(
        self,
        query_vec: np.ndarray,
        query_text: str,
        top_k: int,
        threshold: float,
        use_hybrid: bool,
    ) -> List[RetrievedChunk]:
        """Delegate to VectorStore with given parameters."""
        return self._vs.search(
            query_vec=query_vec,
            query_text=query_text,
            top_k=top_k,
            threshold=threshold,
            use_hybrid=use_hybrid,
        )

    async def _synthesize(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Ask the LLM to produce an answer from the provided chunks."""
        return await self._llm.synthesize(query, chunks)

    @staticmethod
    def _is_refusal(answer: str) -> bool:
        """Return True if the LLM's answer is the insufficient-context refusal."""
        return _INSUFFICIENT_CONTEXT_PHRASE in answer

    @staticmethod
    def _build_response(
        stage: RAGStage,
        answer: str,
        chunks: List[RetrievedChunk],
        confidence: float,
        expanded_queries: Optional[List[str]] = None,
        latency_ms: Optional[float] = None,
    ) -> RAGResponse:
        return RAGResponse(
            stage_used=stage,
            confidence=round(confidence, 4),
            answer=answer,
            chunks_used=len(chunks),
            expanded_queries=expanded_queries or [],
            latency_ms=round(latency_ms or 0.0, 2),
        )
