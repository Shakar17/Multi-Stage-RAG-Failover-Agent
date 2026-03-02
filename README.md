# Multi-Stage RAG Failover Agent

Production-ready retrieval pipeline with strict no-hallucination guarantees.

---

## Architecture

```
                         ┌─────────────────────────────────────────────────┐
                         │           RAG FAILOVER PIPELINE                 │
                         └─────────────────────────────────────────────────┘

User Query
    │
    ▼
┌───────────────────────────────────┐
│  FastAPI  POST /query             │
│  QueryRequest { query, top_k? }   │
└──────────────┬────────────────────┘
               │
               ▼
┌───────────────────────────────────┐
│  EmbeddingService                 │
│  sentence-transformers            │
│  L2-normalised float32 vector     │
└──────────────┬────────────────────┘
               │ query_vec
               ▼
┌══════════════════════════════════════════════════════════════════════════════╗
║  RAGFailoverController                                                       ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 1 – High-Threshold Retrieval                                 │    ║
║  │  • threshold ≥ 0.85  •  top_k = 3  •  semantic only                │    ║
║  │  • VectorStore.search(use_hybrid=False)                             │    ║
║  │  • if chunks found → LLM.synthesize() → return                     │    ║
║  └──────────────────────────────────┬──────────────────────────────────┘    ║
║                                     │ FAIL (no chunks ≥ 0.85)               ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 2 – Broad Fallback Search                                    │    ║
║  │  • threshold ≥ 0.60  •  top_k = 8  •  hybrid (semantic + keyword)  │    ║
║  │  • VectorStore.search(use_hybrid=True)                              │    ║
║  │  • if chunks found → LLM.synthesize() → return                     │    ║
║  └──────────────────────────────────┬──────────────────────────────────┘    ║
║                                     │ FAIL (no chunks ≥ 0.60)               ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 3 – LLM Query Expansion                                      │    ║
║  │  • LLM.expand_query() → 3-5 alternative phrasings                  │    ║
║  │  • Re-embed each expansion                                          │    ║
║  │  • threshold ≥ 0.55  •  top_k = 6  •  hybrid                       │    ║
║  │  • Keep best-scoring set of chunks across all expansions            │    ║
║  │  • if chunks found → LLM.synthesize() → return                     │    ║
║  └──────────────────────────────────┬──────────────────────────────────┘    ║
║                                     │ FAIL (expansion didn't help)           ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 4 – Context-Only Synthesis                                   │    ║
║  │  • Very low threshold (0.30) – gather whatever exists               │    ║
║  │  • LLM forced to synthesise from context ONLY                       │    ║
║  │  • If context insufficient → "I don't have enough information…"    │    ║
║  └──────────────────────────────────┬──────────────────────────────────┘    ║
║                                     │ FAIL (LLM itself refused)              ║
║                                     ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  STAGE 5 – Final Refusal                                            │    ║
║  │  "Sorry, this question is outside the supported domain."            │    ║
║  │  confidence = 0.0                                                   │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
               │
               ▼
        RAGResponse JSON
        {
          "stage_used":       "stage1_high_threshold",
          "confidence":       0.912,
          "answer":           "...",
          "chunks_used":      3,
          "expanded_queries": [],
          "latency_ms":       42.1
        }


┌──────────────────────────────────────────────────────────┐
│  COMPONENT MAP                                            │
│                                                           │
│  app/config.py      – All thresholds & model settings    │
│  app/models.py      – Pydantic schemas (RAGResponse etc) │
│  app/embedding.py   – sentence-transformers wrapper       │
│  app/vector_store.py– FAISS + hybrid keyword scoring     │
│  app/llm_client.py  – Anthropic API: expand + synthesize │
│  app/controller.py  – 5-stage failover orchestration     │
│  app/main.py        – FastAPI routes & DI                 │
│  tests/test_rag.py  – Unit + integration + API tests      │
└──────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (optional – mock mode works without it)
```

### 3. Run the API server
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Ingest documents
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {"chunk_id": "1", "text": "Python is a high-level programming language.", "source": "python_intro.txt"},
      {"chunk_id": "2", "text": "FastAPI is a modern web framework for building APIs with Python.", "source": "fastapi_docs.txt"},
      {"chunk_id": "3", "text": "FAISS enables efficient similarity search over dense vectors.", "source": "faiss_readme.txt"}
    ]
  }'
```

### 5. Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FastAPI?"}'
```

Expected response:
```json
{
  "stage_used": "stage1_high_threshold",
  "confidence": 0.9123,
  "answer": "FastAPI is a modern web framework for building APIs with Python.",
  "chunks_used": 1,
  "expanded_queries": [],
  "latency_ms": 38.4
}
```

### 6. Run tests
```bash
pytest tests/ -v
```

---

## Stage Thresholds (configurable in `.env`)

| Stage | Description             | Threshold | top_k | Hybrid |
|-------|-------------------------|-----------|-------|--------|
| 1     | High-confidence strict  | ≥ 0.85    | 3     | No     |
| 2     | Broad fallback          | ≥ 0.60    | 8     | Yes    |
| 3     | Post-expansion fallback | ≥ 0.55    | 6     | Yes    |
| 4     | Context-only synthesis  | ≥ 0.30    | 8     | Yes    |
| 5     | Hard refusal            | —         | —     | —      |

---

## Environment Variables

```
ANTHROPIC_API_KEY=sk-ant-...      # Optional: enables real LLM calls
EMBEDDING_MODEL=all-MiniLM-L6-v2  # sentence-transformers model
EMBEDDING_DIM=384
STAGE1_THRESHOLD=0.85
STAGE2_THRESHOLD=0.60
STAGE3_THRESHOLD=0.55
FAISS_INDEX_PATH=data/faiss.index
FAISS_META_PATH=data/faiss_meta.json
LLM_MODEL=claude-sonnet-4-20250514
LOG_LEVEL=INFO
```

---

## Design Principles

- **No hallucinations**: LLM synthesis prompt explicitly forbids using external knowledge.
  The model is instructed to return a fixed refusal phrase if context is insufficient.
- **Deterministic failover**: Stages always run in fixed order (1→2→3→4→5).
- **Separation of retrieval and synthesis**: `VectorStore` handles retrieval;
  `LLMClient` handles generation. The `Controller` orchestrates but does neither.
- **Extensible**: Swap FAISS for Chroma by replacing `VectorStore`. Swap Anthropic
  for OpenAI by updating `LLMClient`. All thresholds are in `config.py`.
- **Observable**: Every stage logs its outcome (`[STAGE N] ✓/✗ ...`).
  `stage_used` in the response tells operators exactly what path answered.
