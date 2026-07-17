"""
Tests for Reciprocal Rank Fusion (RRF) fusion logic.

These tests run on pure Python - no DB, no embedder, no torch.
"""

import pytest
from core.interfaces import ChunkResult
from query.retrieval import reciprocal_rank_fusion, _RRF_K


def _make_chunk(doc_id: int, chunk_index: int, text: str = "x") -> ChunkResult:
    return ChunkResult(
        text=text,
        chunk_index=chunk_index,
        document_id=doc_id,
        document_title=f"Doc {doc_id}",
        similarity=0.5,
    )


def test_rrf_single_list_preserves_order():
    chunks = [_make_chunk(1, i) for i in range(5)]
    fused = reciprocal_rank_fusion([chunks], top_k=5)
    assert [c.chunk_index for c in fused] == [0, 1, 2, 3, 4]


def test_rrf_top_k_limits_results():
    chunks = [_make_chunk(1, i) for i in range(10)]
    fused = reciprocal_rank_fusion([chunks], top_k=3)
    assert len(fused) == 3


def test_rrf_chunk_appearing_in_both_lists_ranks_higher():
    """A chunk ranked #1 in both lists should outscore a chunk ranked #1 in only one."""
    shared = _make_chunk(doc_id=99, chunk_index=0, text="shared")
    list_a = [shared, _make_chunk(1, 1), _make_chunk(1, 2)]
    list_b = [shared, _make_chunk(2, 0), _make_chunk(2, 1)]

    only_a_top = _make_chunk(1, 1)
    only_b_top = _make_chunk(2, 0)

    fused = reciprocal_rank_fusion([list_a, list_b], top_k=5)
    keys = [(c.document_id, c.chunk_index) for c in fused]
    assert keys[0] == (99, 0), "shared chunk should be first"


def test_rrf_scores_decrease_monotonically():
    """RRF scores should be in descending order in the output."""
    list_a = [_make_chunk(1, i) for i in range(6)]
    list_b = [_make_chunk(1, 5 - i) for i in range(6)]  # reversed
    fused = reciprocal_rank_fusion([list_a, list_b], top_k=6)
    scores = [c.similarity for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_deduplicates_chunks():
    """Same chunk appearing in multiple lists should appear only once in output."""
    chunk = _make_chunk(1, 0)
    list_a = [chunk, _make_chunk(1, 1)]
    list_b = [chunk, _make_chunk(1, 2)]
    fused = reciprocal_rank_fusion([list_a, list_b], top_k=10)
    keys = [(c.document_id, c.chunk_index) for c in fused]
    assert len(keys) == len(set(keys)), "duplicate chunks in output"


def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([[], []], top_k=5)
    assert fused == []


def test_rrf_one_empty_list():
    chunks = [_make_chunk(1, i) for i in range(3)]
    fused = reciprocal_rank_fusion([chunks, []], top_k=3)
    assert len(fused) == 3


def test_rrf_known_scores():
    """Verify exact RRF scores for a controlled input."""
    k = _RRF_K  # 60
    c0 = _make_chunk(1, 0)
    c1 = _make_chunk(1, 1)

    # c0 is rank 1 in list A; c1 is rank 1 in list B
    # c0 is rank 2 in list B; c1 is rank 2 in list A
    list_a = [c0, c1]
    list_b = [c1, c0]

    fused = reciprocal_rank_fusion([list_a, list_b], top_k=2)
    # Both get 1/(k+1) + 1/(k+2) - equal - so order may be either
    score_c0 = fused[0].similarity if fused[0].chunk_index == 0 else fused[1].similarity
    score_c1 = fused[0].similarity if fused[0].chunk_index == 1 else fused[1].similarity
    expected = round(1 / (k + 1) + 1 / (k + 2), 6)
    assert abs(score_c0 - expected) < 1e-5
    assert abs(score_c1 - expected) < 1e-5
