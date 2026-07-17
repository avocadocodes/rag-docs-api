"""
Retrieval backends.

Three retrievers are provided:

  PgvectorRetriever - dense cosine-similarity search via pgvector.
  LexicalRetriever  - Postgres full-text search (websearch_to_tsquery + ts_rank).
  HybridRetriever   - runs both and fuses results with Reciprocal Rank Fusion (RRF).

All retrievers are Postgres-only at runtime; the module is safe to import on
SQLite because every pgvector / DB-specific import is deferred inside methods.
Tests inject fakes via dependency injection rather than importing these classes.

--- Reciprocal Rank Fusion ---

RRF combines ranked lists from multiple retrievers without requiring scores
to be on the same scale.  For each chunk, its RRF score is:

    rrf_score = Σ  1 / (k + rank_i)

where k is a smoothing constant (default 60, from the original Cormack et al.
2009 paper) and rank_i is the 1-based position in list i.  Chunks that appear
near the top of multiple lists accumulate a high RRF score and float to the top
of the fused list.  The k=60 constant prevents the #1 result in a single list
from dominating when the other retriever didn't find that chunk at all.
"""

from __future__ import annotations

from collections import defaultdict

from core.interfaces import ChunkResult

# RRF smoothing constant.  k=60 is the value used in the original paper and
# works well in practice for retrieval fusion of 2–4 ranked lists.
_RRF_K = 60


class PgvectorRetriever:
    """Retrieve chunks from PostgreSQL/pgvector using cosine similarity."""

    def retrieve(self, query_embedding: list[float], top_k: int) -> list[ChunkResult]:
        from pgvector.django import CosineDistance  # noqa: PLC0415
        from documents.models import DocumentChunk  # noqa: PLC0415

        rows = (
            DocumentChunk.objects.select_related("document")
            .annotate(distance=CosineDistance("embedding", query_embedding))
            .order_by("distance")[:top_k]
        )

        return [
            ChunkResult(
                text=row.text,
                chunk_index=row.chunk_index,
                document_id=row.document_id,
                document_title=row.document.title,
                similarity=round(1.0 - float(row.distance), 4),
            )
            for row in rows
        ]


class LexicalRetriever:
    """
    Retrieve chunks using Postgres full-text search.

    Uses websearch_to_tsquery (Postgres 11+) which understands quoted phrases
    and minus-exclusion without requiring a custom parser.  Results are ranked
    by ts_rank which weights term frequency and document position.
    """

    def retrieve(self, query_text: str, top_k: int) -> list[ChunkResult]:
        from django.db.models import F  # noqa: PLC0415
        from django.db.models.expressions import RawSQL  # noqa: PLC0415
        from documents.models import DocumentChunk  # noqa: PLC0415

        rank_expr = RawSQL(
            "ts_rank(search_vector, websearch_to_tsquery('english', %s))",
            [query_text],
        )
        match_expr = RawSQL(
            "search_vector @@ websearch_to_tsquery('english', %s)",
            [query_text],
        )

        rows = (
            DocumentChunk.objects.select_related("document")
            .annotate(rank=rank_expr, matched=match_expr)
            .filter(matched=True)
            .order_by("-rank")[:top_k]
        )

        return [
            ChunkResult(
                text=row.text,
                chunk_index=row.chunk_index,
                document_id=row.document_id,
                document_title=row.document.title,
                similarity=round(float(row.rank), 4),
            )
            for row in rows
        ]


def _chunk_key(chunk: ChunkResult) -> tuple[int, int]:
    """Stable identity key for a chunk across retriever lists."""
    return (chunk.document_id, chunk.chunk_index)


def reciprocal_rank_fusion(
    ranked_lists: list[list[ChunkResult]],
    top_k: int,
    k: int = _RRF_K,
) -> list[ChunkResult]:
    """
    Fuse multiple ranked lists of ChunkResult using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : list of ranked ChunkResult lists (one per retriever)
    top_k        : number of results to return
    k            : RRF smoothing constant (default 60)

    Returns
    -------
    A deduplicated list of ChunkResult ordered by descending RRF score.
    The similarity field is set to the RRF score for transparency.
    """
    scores: dict[tuple, float] = defaultdict(float)
    best_chunk: dict[tuple, ChunkResult] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = _chunk_key(chunk)
            scores[key] += 1.0 / (k + rank)
            if key not in best_chunk:
                best_chunk[key] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for key, score in fused:
        chunk = best_chunk[key]
        results.append(
            ChunkResult(
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                similarity=round(score, 6),
            )
        )
    return results


class HybridRetriever:
    """
    Combines vector search and lexical search via Reciprocal Rank Fusion.

    Retrieves `candidate_k` results from each backend (default 20) so the
    reranker has a diverse candidate pool to work with, then fuses and
    returns the top `top_k`.
    """

    def __init__(self, candidate_k: int = 20) -> None:
        self._candidate_k = candidate_k

    def retrieve(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int,
    ) -> list[ChunkResult]:
        vector_results = PgvectorRetriever().retrieve(query_embedding, self._candidate_k)
        lexical_results = LexicalRetriever().retrieve(query_text, self._candidate_k)
        fused = reciprocal_rank_fusion([vector_results, lexical_results], top_k=top_k)

        # RRF orders the results, but its fused score is not on a meaningful scale.
        # Report the true dense cosine similarity (from the vector arm) instead, so
        # callers and the relevance gate see a consistent, interpretable score.
        cosine_by_key = {
            (c.document_id, c.chunk_index): c.similarity for c in vector_results
        }
        for chunk in fused:
            chunk.similarity = cosine_by_key.get((chunk.document_id, chunk.chunk_index), 0.0)
        return fused
