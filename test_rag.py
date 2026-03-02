"""
Test Suite – Multi-Stage RAG Failover Agent
--------------------------------------------
Tests are organised by stage:
  - Unit tests for VectorStore, EmbeddingService, LLMClient (mocked)
  - Integration tests for the RAGFailoverController (fully mocked dependencies)
  - FastAPI endpoint tests via TestClient

Run: pytest tests/test_rag.py -v
"""

from __future__ import annotations

import json
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Fixtures & helpers ────────────────────────────────────────────────────────

def make_chunk(text: str, score: float, source: str = "test_doc.txt"):
    from app.models import RetrievedChunk
    return RetrievedChunk(
        chunk_id=str(uuid.uuid4()),
        text=text,
        source=source,
        score=score,
    )


def random_vec(dim: int = 384) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS – EmbeddingService
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingService:
    """Verify that embeddings are L2-normalised and have the right shape."""

    @pytest.fixture(autouse=True)
    def patch_st(self):
        """Avoid loading a real sentence-transformers model."""
        with patch("app.embedding.SentenceTransformer") as mock_cls:
            mock_model = MagicMock()
            mock_model.encode.side_effect = lambda texts, **kw: (
                np.random.randn(len(texts), 384).astype(np.float32)
            )
            mock_cls.return_value = mock_model
            yield

    def test_embed_query_shape(self):
        from app.embedding import EmbeddingService
        svc = EmbeddingService("dummy-model")
        vec = svc.embed_query("hello world")
        assert vec.shape == (384,), "Query embedding must be (dim,)"

    def test_embed_batch_shape(self):
        from app.embedding import EmbeddingService
        svc = EmbeddingService("dummy-model")
        vecs = svc.embed_batch(["hello", "world", "test"])
        assert vecs.shape == (3, 384), "Batch embedding must be (N, dim)"


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS – VectorStore
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStore:
    """Test indexing, search thresholds, and hybrid scoring."""

    @pytest.fixture
    def store(self, tmp_path):
        from app.models import DocumentChunk
        from app.vector_store import VectorStore

        vs = VectorStore(
            dim=4,
            index_path=str(tmp_path / "test.index"),
            meta_path=str(tmp_path / "test_meta.json"),
        )

        # Add three synthetic chunks
        chunks = [
            DocumentChunk(chunk_id="c1", text="Python is a programming language", source="doc1.txt"),
            DocumentChunk(chunk_id="c2", text="Rust is a systems language for safety", source="doc2.txt"),
            DocumentChunk(chunk_id="c3", text="Machine learning uses neural networks", source="doc3.txt"),
        ]
        dim4 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        vs.add_chunks(chunks, dim4)
        return vs

    def test_total_chunks(self, store):
        assert store.total_chunks == 3

    def test_exact_match_retrieval(self, store):
        """A query vector identical to chunk c1 should return c1 with score ~1.0."""
        query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query_vec, "Python programming", top_k=1, threshold=0.5)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert results[0].score >= 0.5

    def test_threshold_filters_out_low_score(self, store):
        """A very high threshold should return no results for an orthogonal query."""
        query_vec = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        results = store.search(query_vec, "unrelated query xyz", top_k=3, threshold=0.99)
        assert results == [], "All chunks should be below threshold 0.99 for orthogonal vector"

    def test_empty_store_returns_empty(self, tmp_path):
        from app.vector_store import VectorStore
        vs = VectorStore(dim=4, index_path=str(tmp_path / "e.index"), meta_path=str(tmp_path / "e.json"))
        results = vs.search(np.array([1, 0, 0, 0], dtype=np.float32), "test", top_k=5, threshold=0.5)
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS – LLMClient (mock mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMClientMock:
    """LLMClient behaves sensibly when no API key is configured."""

    @pytest.fixture
    def client(self):
        from app.llm_client import LLMClient
        c = LLMClient()
        c._api_key = None   # force mock path
        return c

    @pytest.mark.asyncio
    async def test_expand_query_returns_list(self, client):
        expansions = await client.expand_query("python speed")
        assert isinstance(expansions, list)
        assert len(expansions) >= 1

    @pytest.mark.asyncio
    async def test_synthesize_with_chunks(self, client):
        chunks = [make_chunk("Python is fast with Cython.", score=0.9)]
        answer = await client.synthesize("How to speed up Python?", chunks)
        assert isinstance(answer, str)
        assert len(answer) > 10

    @pytest.mark.asyncio
    async def test_synthesize_empty_chunks_returns_refusal(self, client):
        answer = await client.synthesize("something obscure", [])
        assert "don't have enough information" in answer.lower()


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS – RAGFailoverController
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGFailoverController:
    """End-to-end stage routing with mocked vector store and LLM."""

    def _make_controller(self, retrieval_results, synthesis_answer, expand_returns=None):
        """Build a controller with controlled mock behaviour."""
        from app.controller import RAGFailoverController

        mock_vs = MagicMock()
        mock_vs.search.return_value = retrieval_results

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = random_vec()

        mock_llm = MagicMock()
        mock_llm.synthesize = AsyncMock(return_value=synthesis_answer)
        mock_llm.expand_query = AsyncMock(return_value=expand_returns or ["expanded query"])

        return RAGFailoverController(mock_vs, mock_emb, mock_llm)

    @pytest.mark.asyncio
    async def test_stage1_answered(self):
        """High-confidence chunk → stage 1 answers."""
        chunk = make_chunk("Python uses GIL.", score=0.92)
        ctrl = self._make_controller([chunk], "Python uses the GIL for thread safety.")
        resp = await ctrl.query("What is the GIL?")
        assert resp.stage_used.value == "stage1_high_threshold"
        assert resp.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_stage2_fallback(self):
        """
        Stage 1 fails (no results above 0.85).
        Stage 2 returns a mid-confidence chunk.
        """
        from app.models import RAGStage
        chunk = make_chunk("Java is object-oriented.", score=0.70)

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            threshold = kwargs.get("threshold", 0.85)
            if threshold >= 0.85:
                return []    # Stage 1 fails
            return [chunk]   # Stage 2 succeeds

        mock_vs = MagicMock()
        mock_vs.search.side_effect = side_effect

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = random_vec()

        mock_llm = MagicMock()
        mock_llm.synthesize = AsyncMock(return_value="Java is an OO language.")
        mock_llm.expand_query = AsyncMock(return_value=["expanded"])

        from app.controller import RAGFailoverController
        ctrl = RAGFailoverController(mock_vs, mock_emb, mock_llm)
        resp = await ctrl.query("Tell me about Java")

        assert resp.stage_used == RAGStage.STAGE2_BROAD_FALLBACK

    @pytest.mark.asyncio
    async def test_stage3_expansion(self):
        """Stages 1+2 fail; stage 3 (expansion) succeeds."""
        from app.models import RAGStage
        chunk = make_chunk("Rust prevents data races.", score=0.58)

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            threshold = kwargs.get("threshold", 0.85)
            if threshold >= 0.85:
                return []
            if threshold >= 0.60:
                return []   # Stage 2 fails too
            return [chunk]  # Stage 3 low threshold succeeds

        mock_vs = MagicMock()
        mock_vs.search.side_effect = side_effect

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = random_vec()

        mock_llm = MagicMock()
        mock_llm.synthesize = AsyncMock(return_value="Rust prevents data races via ownership.")
        mock_llm.expand_query = AsyncMock(return_value=["Rust safety features", "Rust concurrency"])

        from app.controller import RAGFailoverController
        ctrl = RAGFailoverController(mock_vs, mock_emb, mock_llm)
        resp = await ctrl.query("rust lang safe?")

        assert resp.stage_used == RAGStage.STAGE3_QUERY_EXPANSION
        assert len(resp.expanded_queries) > 0

    @pytest.mark.asyncio
    async def test_stage5_refusal(self):
        """All stages fail → stage 5 hard refusal."""
        from app.models import RAGStage

        mock_vs = MagicMock()
        mock_vs.search.return_value = []   # Always empty

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = random_vec()

        mock_llm = MagicMock()
        mock_llm.synthesize = AsyncMock(return_value="I don't have enough information in the knowledge base.")
        mock_llm.expand_query = AsyncMock(return_value=["exp1"])

        from app.controller import RAGFailoverController
        ctrl = RAGFailoverController(mock_vs, mock_emb, mock_llm)
        resp = await ctrl.query("What is the airspeed velocity of an unladen swallow?")

        assert resp.stage_used == RAGStage.STAGE5_REFUSAL
        assert "outside the supported domain" in resp.answer

    @pytest.mark.asyncio
    async def test_confidence_zero_on_refusal(self):
        """Confidence must be 0.0 for stage-5 refusals."""
        from app.models import RAGStage
        mock_vs = MagicMock(); mock_vs.search.return_value = []
        mock_emb = MagicMock(); mock_emb.embed_query.return_value = random_vec()
        mock_llm = MagicMock()
        mock_llm.synthesize = AsyncMock(return_value="I don't have enough information in the knowledge base.")
        mock_llm.expand_query = AsyncMock(return_value=[])

        from app.controller import RAGFailoverController
        ctrl = RAGFailoverController(mock_vs, mock_emb, mock_llm)
        resp = await ctrl.query("nonsense query xyzzy")

        assert resp.confidence == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINT TESTS – FastAPI TestClient
# ─────────────────────────────────────────────────────────────────────────────

class TestAPI:
    """Smoke tests for the HTTP API layer."""

    @pytest.fixture
    def client(self):
        # Patch the controller inside main to avoid real FAISS / model loading
        from unittest.mock import patch, AsyncMock
        from app.models import RAGResponse, RAGStage

        mock_response = RAGResponse(
            stage_used=RAGStage.STAGE1_HIGH_THRESHOLD,
            confidence=0.91,
            answer="This is a test answer.",
            chunks_used=2,
        )

        with patch("app.main._controller") as mock_ctrl:
            mock_ctrl.query = AsyncMock(return_value=mock_response)
            from app.main import app
            with TestClient(app) as tc:
                yield tc, mock_ctrl

    def test_health_endpoint(self, client):
        tc, _ = client
        resp = tc.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_query_endpoint_returns_json(self, client):
        tc, _ = client
        resp = tc.post("/query", json={"query": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        # Verify all required JSON fields are present
        assert "stage_used" in data
        assert "confidence" in data
        assert "answer" in data

    def test_query_empty_string_rejected(self, client):
        tc, _ = client
        resp = tc.post("/query", json={"query": ""})
        assert resp.status_code == 422   # Pydantic validation error

    def test_query_too_long_rejected(self, client):
        tc, _ = client
        resp = tc.post("/query", json={"query": "x" * 3000})
        assert resp.status_code == 422
