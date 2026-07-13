"""
Tests for retrieval ordering using an in-memory fake retriever.

These tests verify that the retrieval contract (top-k, ordered by similarity)
is satisfied, without touching PostgreSQL or pgvector.
"""

import math
import pytest
from core.fake_embedder import FakeEmbedder
from core.interfaces import ChunkResult


class InMemoryRetriever:
    """Fake retriever that holds chunks in memory and scores by dot product."""

    def __init__(self, chunks: list[tuple[str, list[float], int, str, int]]):
        # Each entry: (text, embedding, doc_id, doc_title, chunk_index)
        self._chunks = chunks

    def retrieve(self, query_embedding: list[float], top_k: int) -> list[ChunkResult]:
        scored = []
        for text, emb, doc_id, doc_title, chunk_index in self._chunks:
            # cosine similarity = dot product (vectors are unit-length from FakeEmbedder)
            similarity = sum(a * b for a, b in zip(query_embedding, emb))
            scored.append((similarity, text, emb, doc_id, doc_title, chunk_index))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            ChunkResult(
                text=text,
                chunk_index=chunk_index,
                document_id=doc_id,
                document_title=doc_title,
                similarity=round(sim, 4),
            )
            for sim, text, emb, doc_id, doc_title, chunk_index in scored[:top_k]
        ]


def test_retriever_returns_top_k():
    embedder = FakeEmbedder()
    texts = [f"document text number {i}" for i in range(10)]
    chunks = [
        (t, embedder.embed(t), 1, "Doc", i)
        for i, t in enumerate(texts)
    ]
    retriever = InMemoryRetriever(chunks)
    query_emb = embedder.embed("document text number 3")
    results = retriever.retrieve(query_emb, top_k=3)
    assert len(results) == 3


def test_retriever_results_ordered_by_similarity():
    embedder = FakeEmbedder()
    texts = [f"unique phrase {i}" for i in range(5)]
    chunks = [(t, embedder.embed(t), 1, "Doc", i) for i, t in enumerate(texts)]
    retriever = InMemoryRetriever(chunks)

    query_emb = embedder.embed("unique phrase 2")
    results = retriever.retrieve(query_emb, top_k=5)

    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_retriever_top_1_is_most_similar():
    embedder = FakeEmbedder()
    target = "the exact query text"
    other = "completely different content about something else"

    chunks = [
        (target, embedder.embed(target), 1, "Doc", 0),
        (other, embedder.embed(other), 1, "Doc", 1),
    ]
    retriever = InMemoryRetriever(chunks)
    query_emb = embedder.embed(target)
    results = retriever.retrieve(query_emb, top_k=2)

    assert results[0].text == target


def test_top_k_limits_results():
    embedder = FakeEmbedder()
    chunks = [(f"chunk {i}", embedder.embed(f"chunk {i}"), 1, "Doc", i) for i in range(20)]
    retriever = InMemoryRetriever(chunks)
    query_emb = embedder.embed("chunk 5")

    for k in [1, 3, 5, 10]:
        results = retriever.retrieve(query_emb, top_k=k)
        assert len(results) == k
