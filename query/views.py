"""
Query views.

QueryView        — POST /api/v1/query
    Supports retrieval mode (vector | lexical | hybrid, default hybrid),
    optional cross-encoder reranking (rerank=true, default true), and
    Redis-backed result caching.

QueryStreamView  — POST /api/v1/query/stream
    Same pipeline as QueryView but streams the answer as Server-Sent Events
    (text/event-stream).  Each SSE event carries a "data:" line with a text
    fragment.  A final event with data: [DONE] signals completion.

Cache key: SHA-256 of (question, mode, rerank_flag).  Cache TTL is
QUERY_CACHE_TTL seconds (default 300).  The cache is a no-op if only
locmem is configured, which is the case in tests.
"""

from __future__ import annotations

import hashlib
import json

from django.conf import settings
from django.core.cache import cache
from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

from query.serializers import QueryRequestSerializer, QueryResponseSerializer
from query.retrieval import PgvectorRetriever, LexicalRetriever, HybridRetriever
from query.answer import generate_answer, stream_answer
from core.embedder import get_embedder


def _get_reranker():
    """Return the reranker instance appropriate for the current environment."""
    backend = getattr(settings, "RERANKER_BACKEND", "real")
    if backend == "fake":
        from core.reranker import FakeReranker  # noqa: PLC0415
        return FakeReranker()
    from core.reranker import get_reranker  # noqa: PLC0415
    return get_reranker()


def _cache_key(question: str, mode: str, rerank: bool) -> str:
    payload = f"{question}|{mode}|{rerank}"
    return "query:" + hashlib.sha256(payload.encode()).hexdigest()


def _retrieve(question: str, query_embedding: list[float], mode: str, top_k: int):
    """Run the appropriate retriever based on mode."""
    if mode == "vector":
        return PgvectorRetriever().retrieve(query_embedding, top_k)
    if mode == "lexical":
        return LexicalRetriever().retrieve(question, top_k)
    # hybrid (default)
    return HybridRetriever().retrieve(query_embedding, question, top_k)


# Candidate pool size fed to the reranker before trimming to top_k.
_RERANK_CANDIDATE_K = 20


class QueryView(APIView):

    @extend_schema(
        request=QueryRequestSerializer,
        responses=QueryResponseSerializer,
        summary="Ask a question against ingested documents",
        description=(
            "Embeds the question, retrieves the most relevant document chunks "
            "(vector, lexical, or hybrid mode), optionally reranks with a "
            "cross-encoder, and returns a grounded answer with citations.\n\n"
            "**Retrieval modes**\n"
            "- `hybrid` (default): combines dense vector search (pgvector cosine) "
            "and lexical full-text search (Postgres tsvector) using Reciprocal Rank "
            "Fusion.\n"
            "- `vector`: dense cosine similarity only.\n"
            "- `lexical`: Postgres full-text search only.\n\n"
            "**Reranking** (`rerank=true`, default): a cross-encoder model scores "
            "each (query, chunk) pair independently and reorders the candidates. "
            "More accurate than bi-encoder retrieval but adds ~100 ms latency.\n\n"
            "**Answer mode** depends on environment:\n"
            "- `extractive` (default): top chunks concatenated with citation markers.\n"
            "- `llm`: OpenAI-compatible endpoint if LLM_API_* env vars are set."
        ),
    )
    def post(self, request):
        serializer = QueryRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        question: str = serializer.validated_data["question"]
        top_k: int = serializer.validated_data["top_k"]
        mode: str = serializer.validated_data["mode"]
        rerank: bool = serializer.validated_data["rerank"]

        cache_key = _cache_key(question, mode, rerank)
        cached = cache.get(cache_key)
        if cached is not None:
            return Response(cached, status=status.HTTP_200_OK)

        embedder = get_embedder()
        query_embedding = embedder.embed(question)

        candidate_k = max(top_k, _RERANK_CANDIDATE_K) if rerank else top_k
        chunks = _retrieve(question, query_embedding, mode, candidate_k)

        reranked = False
        if rerank and chunks:
            reranker = _get_reranker()
            chunks = reranker.rerank(question, chunks)[:top_k]
            reranked = True
        else:
            chunks = chunks[:top_k]

        result = generate_answer(question, chunks)

        response_data = {
            "question": question,
            "answer": result.answer,
            "mode": result.mode,
            "retrieval_mode": mode,
            "reranked": reranked,
            "citations": [
                {
                    "document_id": c.document_id,
                    "document_title": c.document_title,
                    "chunk_index": c.chunk_index,
                    "similarity": c.similarity,
                }
                for c in result.citations
            ],
            "retrieved_chunks": [
                {
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "document_id": c.document_id,
                    "document_title": c.document_title,
                    "similarity": c.similarity,
                }
                for c in chunks
            ],
        }

        ttl = getattr(settings, "QUERY_CACHE_TTL", 300)
        cache.set(cache_key, response_data, ttl)

        return Response(response_data, status=status.HTTP_200_OK)


class QueryStreamView(APIView):
    """
    POST /api/v1/query/stream

    Same retrieval pipeline as QueryView.  Returns a Server-Sent Events stream
    (Content-Type: text/event-stream).

    Each SSE message is:
        data: <text fragment>\\n\\n

    The final message is:
        data: [DONE]\\n\\n

    Clients should concatenate all data fragments (excluding [DONE]) to
    reconstruct the full answer.
    """

    @extend_schema(exclude=True)
    def post(self, request):
        serializer = QueryRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        question: str = serializer.validated_data["question"]
        top_k: int = serializer.validated_data["top_k"]
        mode: str = serializer.validated_data["mode"]
        rerank: bool = serializer.validated_data["rerank"]

        embedder = get_embedder()
        query_embedding = embedder.embed(question)

        candidate_k = max(top_k, _RERANK_CANDIDATE_K) if rerank else top_k
        chunks = _retrieve(question, query_embedding, mode, candidate_k)

        if rerank and chunks:
            reranker = _get_reranker()
            chunks = reranker.rerank(question, chunks)[:top_k]
        else:
            chunks = chunks[:top_k]

        def event_stream():
            for fragment in stream_answer(question, chunks):
                # Escape newlines inside a fragment so each SSE message is intact
                escaped = fragment.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
        )
