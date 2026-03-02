"""
LLM Client
----------
Centralises all calls to the language model.

Two responsibilities:
  1. Query expansion  – rewrite an ambiguous query into 3-5 structured alternatives
  2. Context synthesis – answer STRICTLY from retrieved chunks (no hallucination)

The synthesis prompt is intentionally restrictive:
  - The model is told it MUST refuse if context is insufficient.
  - Temperature = 0 for determinism.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

import httpx

from app.config import settings
from app.models import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

EXPANSION_SYSTEM = """You are a query expansion assistant.
Given an ambiguous or short user query, produce 3-5 semantically distinct
reformulations that a retrieval system could use to find relevant documents.

Rules:
- Output ONLY a JSON array of strings, no other text.
- Each string is a standalone search query.
- Do not add information the original query doesn't imply.
- Vary vocabulary and phrasing to maximise retrieval coverage.

Example input: "python speed"
Example output: ["How to improve Python performance", "Python optimization techniques",
 "Speeding up Python code execution", "Python profiling and bottlenecks"]
"""

SYNTHESIS_SYSTEM = """You are a precise question-answering assistant.
You MUST answer ONLY using the context passages provided below.
You MUST NOT use any external knowledge, training data, or assumptions.

If the context does not contain enough information to answer the question:
  - Respond EXACTLY: "I don't have enough information in the knowledge base."

Otherwise:
  - Answer concisely and accurately, citing the relevant passage(s).
  - Never fabricate facts, names, numbers, or dates.
"""


# ── LLMClient ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin async wrapper around the Anthropic Messages API.
    Falls back to a mock implementation when no API key is configured
    (useful for unit testing without hitting the network).
    """

    def __init__(self) -> None:
        self._api_key = settings.anthropic_api_key
        self._model = settings.llm_model
        self._base_url = "https://api.anthropic.com/v1/messages"

    # ── Query expansion ───────────────────────────────────────────────────────

    async def expand_query(self, query: str) -> List[str]:
        """
        Use the LLM to rewrite `query` into multiple search variants.

        Returns
        -------
        List[str] : 3-5 expanded queries, or [query] on failure.
        """
        if not self._api_key:
            logger.warning("No API key – returning mock query expansions")
            return self._mock_expand(query)

        payload = {
            "model": self._model,
            "max_tokens": 512,
            "temperature": 0.3,   # slight creativity for diverse expansions
            "system": EXPANSION_SYSTEM,
            "messages": [{"role": "user", "content": query}],
        }

        try:
            raw = await self._post(payload)
            text = self._extract_text(raw)
            expansions = self._parse_json_list(text)
            logger.info("Query expanded into %d variants", len(expansions))
            return expansions if expansions else [query]
        except Exception as exc:
            logger.error("Query expansion failed: %s", exc)
            return [query]

    # ── Context synthesis ─────────────────────────────────────────────────────

    async def synthesize(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> str:
        """
        Generate an answer from `chunks` only.

        Returns
        -------
        str : answer string (may be the "not enough information" refusal)
        """
        if not self._api_key:
            logger.warning("No API key – returning mock synthesis")
            return self._mock_synthesize(query, chunks)

        context_block = self._format_context(chunks)
        user_message = (
            f"Context passages:\n{context_block}\n\n"
            f"Question: {query}"
        )

        payload = {
            "model": self._model,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "system": SYNTHESIS_SYSTEM,
            "messages": [{"role": "user", "content": user_message}],
        }

        try:
            raw = await self._post(payload)
            return self._extract_text(raw)
        except Exception as exc:
            logger.error("Synthesis failed: %s", exc)
            return "I don't have enough information in the knowledge base."

    # ── HTTP helper ───────────────────────────────────────────────────────────

    async def _post(self, payload: dict) -> dict:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self._base_url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _extract_text(response: dict) -> str:
        """Pull the first text block from an Anthropic Messages response."""
        for block in response.get("content", []):
            if block.get("type") == "text":
                return block["text"].strip()
        return ""

    @staticmethod
    def _parse_json_list(text: str) -> List[str]:
        """Safely parse a JSON array from the model output."""
        try:
            # Strip markdown fences if present
            cleaned = re.sub(r"```[a-z]*|```", "", text).strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(s) for s in parsed if s]
        except json.JSONDecodeError:
            pass
        return []

    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> str:
        lines = []
        for i, chunk in enumerate(chunks, 1):
            lines.append(
                f"[{i}] (source: {chunk.source}, score: {chunk.score:.3f})\n"
                f"{chunk.text}"
            )
        return "\n\n".join(lines)

    # ── Mock implementations (no API key required) ────────────────────────────

    @staticmethod
    def _mock_expand(query: str) -> List[str]:
        words = query.split()
        return [
            query,
            f"What is {query}",
            f"Explain {query}",
            f"{' '.join(reversed(words))} definition",
        ]

    @staticmethod
    def _mock_synthesize(query: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "I don't have enough information in the knowledge base."
        top = chunks[0]
        return (
            f"[MOCK ANSWER] Based on the retrieved context (score={top.score:.3f}): "
            f"{top.text[:300]}..."
        )
