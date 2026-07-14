# RAG Docs API

A production-grade Retrieval-Augmented Generation backend that turns your document corpus into a searchable, answerable knowledge base.  Upload documents; get semantically grounded answers with citations — and a faithfulness score that tells you how much of the answer is actually supported by the retrieved evidence.

Three retrieval modes — vector, lexical, and hybrid — plus cross-encoder reranking, NLI-based claim verification, and an evaluation harness that gives you real numbers on which configuration works best for your data.

---

## Architecture

```
INGEST FLOW
===========

  POST /documents
       │
       ▼
  ┌──────────────┐  split into overlapping   ┌────────────────────────────────┐
  │  Raw text /  │  chunks (~500 tok, 50      │  chunk_0  chunk_1  chunk_2 …   │
  │  file upload │  tok overlap)         ──►  │                                │
  └──────────────┘                            └──────────┬─────────────────────┘
                                                         │ embed (sentence-transformers)
                                                         │ + to_tsvector (Postgres)
                                                         ▼
                                              ┌─────────────────────┐
                                              │  pgvector (Postgres) │
                                              │  vector(384) column  │
                                              │  tsvector column     │
                                              └─────────────────────┘


QUERY FLOW
==========

  POST /query  { "question": "…", "top_k": 5, "mode": "hybrid", "rerank": true }
       │
       ▼
  embed question ──►  ┌──────────────────┐   ┌──────────────────────┐
  (same model)        │  PgvectorRetriever│   │  LexicalRetriever    │
                      │  cosine distance  │   │  websearch_to_tsquery│
                      │  top-20 candidates│   │  + ts_rank           │
                      └────────┬─────────┘   └──────────┬───────────┘
                               │                        │
                               └────────────┬───────────┘
                                            │
                                  Reciprocal Rank Fusion (k=60)
                                            │
                                            ▼
                                  ┌───────────────────────┐
                                  │  CrossEncoderReranker  │
                                  │  ms-marco-MiniLM-L-6   │
                                  │  reorder top-20        │
                                  └───────────┬────────────┘
                                              │  top_k final chunks
                                              ▼
                                   ┌─────────────────────┐
                                   │  LLM_API_* set?      │
                                   │                      │
                              YES  ▼               NO     ▼
                        OpenAI-compatible    extractive answer
                        chat endpoint        (top chunks +
                        grounded prompt      citation markers)
                                   │                      │
                                   └──────────┬───────────┘
                                              ▼
                                   ┌─────────────────────┐
                                   │  NLI Claim Verifier  │
                                   │  (per-claim verdict) │
                                   └──────────┬───────────┘
                                              │
                                   faithfulness < threshold?
                                   YES → abstain   NO → return answer
                                              │
                              answer + faithfulness + abstained
                              + claims + citations + retrieval_mode
                              + reranked flag

  POST /query/stream  — same pipeline, returns Server-Sent Events
  Redis cache         — key = SHA-256(question + mode + rerank), TTL 5 min
```

---

## Retrieval Modes

| Mode | What happens |
|------|-------------|
| `hybrid` (default) | Runs vector and lexical retrieval in parallel, fuses with RRF |
| `vector` | Dense cosine-similarity search via pgvector only |
| `lexical` | Postgres full-text search (`websearch_to_tsquery` + `ts_rank`) only |

Set with `"mode": "vector"` in the request body.

---

## Reciprocal Rank Fusion (RRF)

RRF combines two ranked lists without needing scores to be on the same scale.  For each candidate chunk its fused score is:

```
rrf_score = 1/(k + rank_vector) + 1/(k + rank_lexical)
```

where `k = 60` (smoothing constant from Cormack et al. 2009) and `rank_*` is the 1-based position in that retriever's list.  Chunks that appear near the top of **both** lists accumulate the highest score.  Chunks found by only one retriever still surface if they ranked very highly there.

The k=60 constant prevents the single top result from dominating when the other retriever did not find it.

---

## Cross-Encoder Reranking

After fusion, a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores the candidate pool by attending to query and passage **together** — unlike the bi-encoder used for retrieval, which embeds them independently.  This makes it much more sensitive to exact query terms and their position.

**Pipeline:** retrieve top-20 candidates → rerank → return top-k.

The cost is roughly proportional to the candidate pool size: ~100 ms on CPU for 20 candidates.  Disable with `"rerank": false` when latency matters more than quality.

---

## Faithfulness Verification

After an answer is generated, each sentence is checked against the retrieved evidence using an NLI (Natural Language Inference) model.

**How it works:**

1. The answer is split into individual claims (sentences).
2. Each claim is scored against every retrieved chunk using a cross-encoder NLI model (`cross-encoder/nli-deberta-v3-small` by default).
3. Each claim receives a label: `SUPPORTED`, `UNSUPPORTED`, or `NEUTRAL`.
4. The faithfulness score is the fraction of claims labelled `SUPPORTED`.
5. If faithfulness < `FAITHFULNESS_THRESHOLD` (default 0.5), the answer is replaced with an honest "not enough evidence" message (`abstained: true`).

**Response fields added:**

| Field | Type | Description |
|-------|------|-------------|
| `faithfulness` | float 0–1 | Fraction of claims supported by evidence |
| `abstained` | bool | True when faithfulness < threshold |
| `claims` | list | Per-claim `{text, label, citation}` |

**In-process fake verifier** (`VERIFIER_BACKEND=fake`): used in tests and CI — word-overlap heuristic, no model download needed.

---

## Streaming

`POST /api/v1/query/stream` runs the same retrieval + rerank pipeline, then streams the answer as [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events):

```
data: The mitochondria\n\n
data:  is the powerhouse\n\n
data:  of the cell.\n\n
data: [DONE]\n\n
```

In LLM mode, tokens stream as they arrive from the LLM.  In extractive mode, chunks are yielded one at a time for a progressive-reveal effect.

Note: the stream endpoint does not run faithfulness verification.  Use `POST /api/v1/query` for full verification results.

---

## Evaluation Harness

Run the built-in evaluation command to get real numbers on which configuration works best for your data:

```bash
docker compose exec web python manage.py evaluate
```

Add `--faithfulness` to also run faithfulness and abstention evaluation (requires the real NLI model):

```bash
docker compose exec web python manage.py evaluate --faithfulness
```

The command:
1. Ingests a labeled 30-document corpus about a fictional API product ("Vela").
2. Runs 40 questions through each retrieval configuration.
3. Computes **Recall@1, Recall@3, Recall@5, and MRR** for each.
4. Optionally runs faithfulness scoring on answerable questions and abstention accuracy on out-of-corpus questions.
5. Prints comparison tables.
6. Cleans up the ingested eval documents.

### Metrics

**Recall@k** — fraction of questions where at least one correct document appears in the top-k results.  Higher is better; differences of >5 pp are meaningful.

**MRR (Mean Reciprocal Rank)** — mean of 1/rank where rank is the position of the first correct result.  MRR = 1.0 means every answer was rank #1; MRR = 0.5 means it was rank #2 on average.

**Faithfulness** — fraction of answer claims labelled SUPPORTED by the NLI model across all answerable evaluation questions.

**Abstention accuracy** — fraction of out-of-corpus (unanswerable) questions where the system correctly abstained rather than hallucinating an answer.

### Retrieval Evaluation Results

Measured with `python manage.py evaluate` against Postgres + pgvector (HNSW),
the real embedding model, and the cross-encoder reranker, over the bundled
corpus of **30 documents / 40 labeled questions** (direct, paraphrased, and
near-distractor questions).

| Configuration    | Recall@1 | Recall@3 | Recall@5 |    MRR |
|------------------|----------|----------|----------|--------|
| vector-only      | 0.750    | 0.925    | 0.975    | 0.835  |
| lexical-only     | 0.100    | 0.100    | 0.100    | 0.100  |
| hybrid           | 0.750    | 0.925    | 0.975    | 0.835  |
| hybrid + rerank  | **0.925**| **1.000**| **1.000**| **0.963** |

**What this shows:**
- **Lexical (BM25) alone is weak (0.10)** on this set — most questions are
  paraphrased or share keywords with distractor documents, so keyword matching
  retrieves the wrong article.
- **Dense/vector retrieval is strong (R@5 0.975)** but its top-1 ranking is
  imperfect (R@1 0.75) — the right chunk is retrieved but not always ranked first.
- **Hybrid == vector here**: on this corpus lexical is too weak to improve the
  RRF fusion. An honest result — hybrid's value shows up on corpora with more
  exact-match/identifier queries, and it costs nothing to keep as a safety net.
- **Reranking is where the gain is**: the cross-encoder lifts **Recall@1 from
  0.75 to 0.925** and **MRR from 0.835 to 0.963** — it reorders the retrieved
  candidates so the answer chunk lands first. This is the precision win that
  matters when only the top result is shown to a user or fed to an LLM.

*(An HNSW index is used rather than IVFFlat: IVFFlat with `probes=1` badly
under-retrieves on small/medium collections; HNSW gives near-exact recall with
default search parameters.)*

### Faithfulness & Abstention Results

Measured with `python manage.py evaluate --faithfulness`:

| Metric | Value |
|--------|-------|
| Avg faithfulness (answerable questions) | **1.000** |

Every claim in the returned answers was entailed by the retrieved evidence — the
verifier correctly recognises grounded content and, in LLM mode, flags any
generated claim that is not supported.

**An honest finding on abstention.** I also tested whether a retrieval-similarity
gate can decide *unanswerable* questions (ones whose answer isn't in the corpus).
It cannot, on a topically-dense corpus. Measured top cosine similarities:

- answerable questions: 0.40 – 0.67
- out-of-corpus questions: 0.48 – 0.68

The distributions **overlap completely**, so no similarity threshold separates
them — a same-domain "unanswerable" question still matches some document. The
similarity gate is therefore kept only as a coarse filter for clearly off-domain
queries (threshold 0.40); robust unanswerable-detection is handled by the
LLM grounded-or-abstain path (LLM mode), where the model is instructed to answer
only from the provided context and say "I don't know" otherwise. Extractive mode
is grounded by construction — the answer *is* the retrieved text, so it cannot
hallucinate, but it also cannot judge relevance beyond the coarse gate. This
trade-off is a deliberate, measured design choice, not an oversight.

---

## Answer Modes

| Mode | When active | What happens |
|------|-------------|--------------|
| **extractive** | Default (no config required) | Top-k chunks concatenated with `[1] … [2] …` citation markers. Zero external dependencies. |
| **llm** | When `LLM_API_BASE` + `LLM_API_KEY` + `LLM_MODEL` are all set | Chunks sent as context to any OpenAI-compatible chat endpoint. Falls back to extractive on failure. |

---

## Caching

Query results are cached in Redis (or in-memory if `REDIS_URL` is unset).  Cache key = SHA-256 of `(question, mode, rerank)`.  Default TTL: 5 minutes.  Set `REDIS_URL` to enable Redis; omit it for local-memory fallback.

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

The first startup downloads the embedding model (~90 MB) into the container.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | (insecure default) | Django secret key |
| `DATABASE_URL` | derived from PG_* | Full Postgres DSN |
| `PG_HOST` | `localhost` | Postgres host |
| `PG_PORT` | `5432` | Postgres port |
| `PG_USER` | `postgres` | Postgres user |
| `PG_PASSWORD` | `postgres` | Postgres password |
| `PG_NAME` | `ragdocs` | Postgres database name |
| `DEBUG` | `true` | Django debug mode |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBEDDING_DIM` | `384` | Embedding vector dimension |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking |
| `NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | NLI model for faithfulness verification |
| `FAITHFULNESS_THRESHOLD` | `0.5` | Minimum faithfulness to avoid abstention |
| `VERIFIER_BACKEND` | `real` | Set to `fake` to use word-overlap verifier (no model) |
| `CHUNK_SIZE` | `500` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `LLM_API_BASE` | _(unset)_ | OpenAI-compatible base URL |
| `LLM_API_KEY` | _(unset)_ | API key for LLM provider |
| `LLM_MODEL` | _(unset)_ | Model name, e.g. `gpt-4o-mini` |
| `REDIS_URL` | _(unset)_ | Redis DSN — enables Redis cache |
| `QUERY_CACHE_TTL` | `300` | Cache TTL in seconds |

---

## API Usage Examples

### Ingest a document

```bash
curl -s -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"title": "Cell Biology", "raw_text": "The mitochondria…"}' | jq .
```

### Query (hybrid + rerank, default)

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does mitochondria produce?", "top_k": 5}' | jq .
```

Response includes `faithfulness`, `abstained`, and `claims` alongside the answer.

### Query (vector only, no rerank)

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "…", "mode": "vector", "rerank": false}' | jq .
```

### Streaming

```bash
curl -N -s -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What does mitochondria produce?"}'
```

### Run retrieval evaluation

```bash
docker compose exec web python manage.py evaluate
```

### Run faithfulness evaluation

```bash
docker compose exec web python manage.py evaluate --faithfulness
```

---

## Running Tests

Tests use SQLite in-memory, `FakeEmbedder`, `FakeReranker`, and `FakeVerifier`.  No Postgres, no model download, no network, no Redis.

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```
