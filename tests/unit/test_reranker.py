"""
Tests for the FakeReranker.

No torch, no model download.  FakeReranker scores by lexical overlap
(shared words) which makes ordering fully predictable.
"""

import pytest
from core.interfaces import ChunkResult
from core.reranker import FakeReranker


def _chunk(text: str, doc_id: int = 1, chunk_index: int = 0) -> ChunkResult:
    return ChunkResult(
        text=text,
        chunk_index=chunk_index,
        document_id=doc_id,
        document_title="Doc",
        similarity=0.5,
    )


def test_fake_reranker_orders_by_overlap():
    reranker = FakeReranker()
    query = "mitochondria ATP energy"
    chunks = [
        _chunk("The cell wall provides structure.", chunk_index=0),
        _chunk("Mitochondria produce ATP through oxidative phosphorylation.", chunk_index=1),
        _chunk("ATP is the primary energy currency of the cell.", chunk_index=2),
    ]
    ranked = reranker.rerank(query, chunks)
    # chunk 1 shares "mitochondria" and "atp"; chunk 2 shares "atp" and "energy"
    # both score higher than chunk 0 which shares nothing
    top_texts = {c.chunk_index for c in ranked[:2]}
    assert top_texts == {1, 2}, f"Expected chunks 1 and 2 at top, got {top_texts}"
    assert ranked[-1].chunk_index == 0


def test_fake_reranker_empty_input():
    reranker = FakeReranker()
    assert reranker.rerank("query", []) == []


def test_fake_reranker_single_chunk():
    reranker = FakeReranker()
    c = _chunk("some text")
    result = reranker.rerank("some", [c])
    assert result == [c]


def test_fake_reranker_preserves_chunk_fields():
    reranker = FakeReranker()
    c = _chunk("hello world", doc_id=42, chunk_index=7)
    result = reranker.rerank("hello", [c])
    assert result[0].document_id == 42
    assert result[0].chunk_index == 7


def test_fake_reranker_no_overlap_preserves_stable_order():
    """When all chunks have zero overlap with query, original order is preserved."""
    reranker = FakeReranker()
    chunks = [_chunk(f"unique_{i}", chunk_index=i) for i in range(4)]
    result = reranker.rerank("zzz", chunks)
    assert [c.chunk_index for c in result] == [0, 1, 2, 3]
