# RAG Docs API

A production-ready Retrieval-Augmented Generation (RAG) backend that turns your document corpus into a searchable, answerable knowledge base. Upload documents; get semantically grounded answers with citations.

---

## The Problem

Large language models hallucinate. They answer from parametric memory, not your data. The solution is to ground every answer in retrieved excerpts from your own documents — so the model can only say what your data actually contains, and every claim is traceable to a source.

This service provides the infrastructure layer for that pattern: ingest documents, embed them locally, retrieve the most relevant passages per query, and return an answer plus citations — all without requiring a hosted LLM.

---

## Architecture

```
INGEST FLOW
===========

  POST /documents
       │
       ▼
  ┌──────────────┐     split into        ┌─────────────────────────────────┐
  │  Raw text /  │  overlapping chunks   │  chunk_0  chunk_1  chunk_2 ...  │
  │  file upload │ ──────────────────►   │  (~500 tokens, 50 tok overlap)  │
  └──────────────┘                       └────────────┬────────────────────┘
                                                      │ embed each chunk
                                                      ▼
                                         sentence-transformers
                                         all-MiniLM-L6-v2 (384-dim)
                                         runs locally, no API key
                                                      │
                                                      ▼
                                         ┌─────────────────────┐
                                         │  pgvector (Postgres) │
                                         │  IVFFlat cosine idx  │
                                         └─────────────────────┘


QUERY FLOW
==========

  POST /query  { "question": "...", "top_k": 5 }
       │
       ▼
  embed question  ──►  cosine similarity search  ──►  top-k chunks
  (same model)         (pgvector <=> operator)
                                                        │
                                             ┌──────────┴──────────┐
                                             │  LLM_API_* set?     │
                                             │                     │
                                        YES  ▼                NO   ▼
                                   OpenAI-compatible    extractive answer
                                   chat endpoint        (top chunks +
                                   grounded prompt      citation markers)
                                             │                     │
                                             └──────────┬──────────┘
                                                        ▼
                                         answer + citations (doc id,
                                         title, chunk index, similarity)
```

---

## Chunking & Embedding Choices

**Chunking** — whitespace-tokenised sliding window, 500 tokens wide, 50 token overlap. No external tokeniser required; deterministic and testable. Overlap ensures that sentences at chunk boundaries aren't lost. Adjust `CHUNK_SIZE` / `CHUNK_OVERLAP` in env if your documents have different density.

**Embedding model** — `all-MiniLM-L6-v2` (384 dimensions, ~90 MB). Best-in-class quality/speed tradeoff for English semantic similarity. Runs entirely on CPU in under 50 ms per chunk on a modern laptop. Loaded once at startup and cached in the worker process.

---

## Why pgvector + Cosine + IVFFlat

- **pgvector** keeps the vector store co-located with your relational data — no separate infrastructure, transactions span both, backups are unified.
- **Cosine similarity** is the standard for normalised sentence embeddings (dot product ≡ cosine when vectors are unit-length, which sentence-transformers produces).
- **IVFFlat index** (`lists=100`) gives sub-linear query time at the cost of approximate results. For up to ~1 M chunks the recall loss is negligible. For larger corpora, switch to the `hnsw` index type — one line change in the migration.

---

## Pluggable LLM vs Extractive Fallback

The service ships two answer modes:

| Mode | When active | What happens |
|------|-------------|--------------|
| **extractive** | Default (no config required) | Top-k retrieved chunks are concatenated with `[1] … [2] …` citation markers and returned as-is. Zero external dependencies, deterministic, works on an air-gapped server. |
| **llm** | When `LLM_API_BASE` + `LLM_API_KEY` + `LLM_MODEL` are all set | Retrieved chunks are sent as context to any OpenAI-compatible chat endpoint. The model is instructed to answer only from the provided context and to cite `[Source N]`. Falls back to extractive if the LLM call fails. |

The extractive fallback is the **default** because:
1. It works with zero additional cost or dependencies.
2. It is fully grounded — the answer literally is the retrieved text.
3. Organisations with strict data-residency requirements may not be able to send data to an external LLM.
4. It makes CI trivial — no mocked LLM calls needed.

---

## Running Locally

**Requirements:** Docker + Docker Compose.

```bash
git clone <repo>
cd rag-docs-api
docker compose up --build
```

The API is available at `http://localhost:8000`.

- Demo UI: `http://localhost:8000/`
- Swagger: `http://localhost:8000/api/docs/`
- Health: `http://localhost:8000/healthz`

The first startup downloads the `all-MiniLM-L6-v2` model (~90 MB) into the container image if you have the pre-bake step enabled in the Dockerfile.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | (insecure default) | Django secret key — **change in production** |
| `DATABASE_URL` | derived from PG_* | Full Postgres DSN |
| `PG_HOST` | `localhost` | Postgres host |
| `PG_PORT` | `5432` | Postgres port |
| `PG_USER` | `postgres` | Postgres user |
| `PG_PASSWORD` | `postgres` | Postgres password |
| `PG_NAME` | `ragdocs` | Postgres database name |
| `DEBUG` | `true` | Django debug mode |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `EMBEDDING_DIM` | `384` | Embedding vector dimension |
| `CHUNK_SIZE` | `500` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `LLM_API_BASE` | _(unset)_ | OpenAI-compatible base URL — enables LLM mode |
| `LLM_API_KEY` | _(unset)_ | API key for LLM provider |
| `LLM_MODEL` | _(unset)_ | Model name, e.g. `gpt-4o-mini` |
| `PORT` | `8000` | Gunicorn bind port |

---

## API Usage Examples

### Ingest a document

```bash
curl -s -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Cell Biology Overview",
    "raw_text": "The mitochondria is the powerhouse of the cell. It produces ATP through oxidative phosphorylation. Cells require a continuous supply of energy to maintain homeostasis. ATP is the universal energy currency..."
  }' | jq .
```

Response:
```json
{
  "id": 1,
  "title": "Cell Biology Overview",
  "chunk_count": 3,
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Upload a file

```bash
curl -s -X POST http://localhost:8000/api/v1/documents \
  -F "title=My Report" \
  -F "file=@report.txt" | jq .
```

### Query

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does mitochondria produce?", "top_k": 3}' | jq .
```

Response (extractive mode):
```json
{
  "question": "What does mitochondria produce?",
  "answer": "[1] The mitochondria is the powerhouse of the cell. It produces ATP through oxidative phosphorylation.\n\n[2] ATP is the universal energy currency...",
  "mode": "extractive",
  "citations": [
    {
      "document_id": 1,
      "document_title": "Cell Biology Overview",
      "chunk_index": 0,
      "similarity": 0.8921
    }
  ],
  "retrieved_chunks": [...]
}
```

### List documents

```bash
curl -s http://localhost:8000/api/v1/documents | jq .
```

---

## Running Tests

Tests use SQLite in-memory and a `FakeEmbedder` (deterministic hash-based vectors). No Postgres, no model download, no network.

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

---

## API Documentation

Interactive Swagger UI: `http://localhost:8000/api/docs/`

OpenAPI schema (JSON): `http://localhost:8000/api/schema/`
