"""
Configuration management for the Multi-Stage RAG Failover Agent.
All thresholds, model names, and tunable parameters live here.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── Anthropic / OpenAI ────────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # ── Embedding model ────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"   # sentence-transformers model
    embedding_dim: int = 384

    # ── Stage thresholds ──────────────────────────────────────────────────────
    stage1_threshold: float = 0.85   # R1 – high-confidence strict retrieval
    stage2_threshold: float = 0.60   # R2 – broad fallback
    stage3_threshold: float = 0.55   # R3 – post-expansion fallback

    # ── Retrieval top-k per stage ─────────────────────────────────────────────
    stage1_top_k: int = 3
    stage2_top_k: int = 8
    stage3_top_k: int = 6

    # ── FAISS index path ──────────────────────────────────────────────────────
    faiss_index_path: str = "data/faiss.index"
    faiss_meta_path: str = "data/faiss_meta.json"

    # ── LLM synthesis / expansion ─────────────────────────────────────────────
    llm_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0          # deterministic synthesis

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
