"""
Abstract interfaces for the embedding and retrieval layers.

These exist so that tests can inject lightweight fakes without touching
the real sentence-transformers model or a live PostgreSQL/pgvector database.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class EmbedderProtocol(Protocol):
    """Converts a text string into a fixed-length float vector."""

    def embed(self, text: str) -> list[float]:
        ...

    @property
    def dim(self) -> int:
        ...


class ChunkResult:
    __slots__ = ("text", "chunk_index", "document_id", "document_title", "similarity")

    def __init__(
        self,
        text: str,
        chunk_index: int,
        document_id: int,
        document_title: str,
        similarity: float,
    ) -> None:
        self.text = text
        self.chunk_index = chunk_index
        self.document_id = document_id
        self.document_title = document_title
        self.similarity = similarity


class RetrieverProtocol(Protocol):
    """Finds the top-k most similar chunks for a query embedding."""

    def retrieve(self, query_embedding: list[float], top_k: int) -> list[ChunkResult]:
        ...
