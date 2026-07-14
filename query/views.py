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
from query.verification import AnswerVerification, no_evidence_result as _abstain_no_evidence
from core.embedder import get_embedder


def _get_reranker():
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
    if mode == "vector":
        return PgvectorRetriever().retrieve(query_embedding, top_k)
    if mode == "lexical":
        return LexicalRetriever().retrieve(question, top_k)
    return HybridRetriever().retrieve(query_embedding, question, top_k)


_RERANK_CANDIDATE_K = 20


class QueryView(APIView):

    @extend_schema(
        request=QueryRequestSerializer,
        responses=QueryResponseSerializer,
        summary="Ask a question against ingested documents",
        description=(
            "Embeds the question, retrieves the most relevant document chunks "
            "(vector, lexical, or hybrid mode), optionally reranks with a "
            "cross-encoder, composes an answer, then verifies each claim in "
            "the answer against the retrieved evidence using NLI.\n\n"
            "New fields in response:\n"
            "- faithfulness (float 0-1): fraction of answer claims supported by evidence.\n"
            "- abstained (bool): true when faithfulness < FAITHFULNESS_THRESHOLD.\n"
            "- claims (list): per-claim {text, label, citation}.\n\n"
            "When abstained is true, answer is replaced with an honest "
            "not enough evidence message; retrieved_chunks are still returned.\n\n"
            "Retrieval modes\n"
            "- hybrid (default): combines dense vector search (pgvector cosine) "
            "and lexical full-text search (Postgres tsvector) using Reciprocal Rank Fusion.\n"
            "- vector: dense cosine similarity only.\n"
            "- lexical: Postgres full-text search only.\n\n"
            "Reranking (rerank=true, default): a cross-encoder model scores "
            "each (query, chunk) pair independently and reorders the candidates.\n\n"
            "Answer mode depends on environment:\n"
            "- extractive (default): top chunks concatenated with citation markers.\n"
            "- llm: OpenAI-compatible endpoint if LLM_API_* env vars are set."
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

        # Relevance gate: if the best-matching chunk is not similar enough to the
        # question, there is no grounded evidence in the corpus, so abstain rather
        # than echo an irrelevant passage. (Cosine similarity is only meaningful
        # for vector/hybrid retrieval; skip the gate when no similarity is present,
        # e.g. lexical-only mode.)
        similarities = [c.similarity for c in chunks if getattr(c, "similarity", None) is not None]
        top_similarity = max(similarities) if similarities else None
        relevance_threshold = getattr(settings, "RELEVANCE_THRESHOLD", 0.0)

        if not chunks or (top_similarity is not None and top_similarity < relevance_threshold):
            result = generate_answer(question, chunks)
            verification = _abstain_no_evidence()
        else:
            result = generate_answer(question, chunks)
            verification = AnswerVerification().verify(answer=result.answer, chunks=chunks)

        response_data = {
            "question": question,
            "answer": verification.answer,
            "mode": result.mode,
            "retrieval_mode": mode,
            "reranked": reranked,
            "faithfulness": verification.faithfulness,
            "abstained": verification.abstained,
            "claims": verification.claims,
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

    Same retrieval pipeline as QueryView. Returns Server-Sent Events stream.
    Stream carries answer text only; use POST /api/v1/query for full verification results.
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
                escaped = fragment.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
        )
