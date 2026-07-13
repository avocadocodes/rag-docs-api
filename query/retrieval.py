"""
pgvector-backed retrieval service.

Uses cosine distance (<=> operator) to find the top-k most similar chunks
to a query embedding.  The IVFFlat index created in the migration makes
this fast at scale.

cosine_distance = 1 - cosine_similarity, so lower is better.
We convert back to similarity score = 1 - distance for the response.

This module is only used at runtime with PostgreSQL.  Tests inject a
fake retriever directly, so this code is never imported during the test
suite.
"""

from __future__ import annotations

from core.interfaces import ChunkResult


class PgvectorRetriever:
    """Retrieve chunks from PostgreSQL/pgvector."""

    def retrieve(self, query_embedding: list[float], top_k: int) -> list[ChunkResult]:
        # Late import so this module is safe to import under SQLite / test env
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
