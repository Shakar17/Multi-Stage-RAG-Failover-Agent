"""
Shared Pydantic schemas used across the API and internal modules.
Keeping models in one place makes the contract explicit and easy to change.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


# ── Stage identifiers ─────────────────────────────────────────────────────────

class RAGStage(str, Enum):
    STAGE1_HIGH_THRESHOLD    = "stage1_high_threshold"
    STAGE2_BROAD_FALLBACK    = "stage2_broad_fallback"
    STAGE3_QUERY_EXPANSION   = "stage3_query_expansion"
    STAGE4_CONTEXT_SYNTHESIS = "stage4_context_synthesis"
    STAGE5_REFUSAL           = "stage5_refusal"


# ── Retrieval chunk ───────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    score: float = Field(..., description="Cosine similarity score [0, 1]")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── API request / response ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000,
                       description="Natural-language question to answer")
    top_k_override: Optional[int] = Field(None, ge=1, le=50,
                                          description="Override default top-k")


class RAGResponse(BaseModel):
    stage_used: RAGStage
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Best similarity score achieved")
    answer: str
    chunks_used: int = Field(default=0,
                             description="Number of retrieved chunks used")
    expanded_queries: List[str] = Field(default_factory=list,
                                        description="LLM-generated expansions (stage 3 only)")
    latency_ms: Optional[float] = None


# ── Ingestion ────────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    chunks: List[DocumentChunk]


class IngestResponse(BaseModel):
    indexed: int
    message: str
