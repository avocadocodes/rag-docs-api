"""
Tests for eval/metrics.py — recall@k and MRR.

Pure Python, no DB, no model.
"""

import pytest
from eval.metrics import recall_at_k, mrr


def test_recall_at_k_perfect():
    results = [([1, 2, 3], [1])]
    assert recall_at_k(results, k=1) == 1.0


def test_recall_at_k_miss():
    results = [([2, 3, 4], [1])]
    assert recall_at_k(results, k=3) == 0.0


def test_recall_at_k_hit_beyond_cutoff():
    """Relevant doc at rank 4 — should not count for k=3."""
    results = [([2, 3, 4, 1], [1])]
    assert recall_at_k(results, k=3) == 0.0
    assert recall_at_k(results, k=4) == 1.0


def test_recall_at_k_partial_hits():
    results = [
        ([1, 2], [1]),    # hit
        ([3, 4], [1]),    # miss
        ([5, 1], [1]),    # hit
    ]
    assert recall_at_k(results, k=2) == pytest.approx(2 / 3)


def test_recall_at_k_multiple_relevant():
    """Any match counts as a hit even when multiple docs are relevant."""
    results = [([3, 5, 7], [5, 9])]
    assert recall_at_k(results, k=3) == 1.0


def test_recall_at_k_empty():
    assert recall_at_k([], k=5) == 0.0


def test_mrr_first_result_relevant():
    results = [([1, 2, 3], [1])]
    assert mrr(results) == pytest.approx(1.0)


def test_mrr_second_result_relevant():
    results = [([2, 1, 3], [1])]
    assert mrr(results) == pytest.approx(0.5)


def test_mrr_not_found():
    results = [([2, 3, 4], [1])]
    assert mrr(results) == pytest.approx(0.0)


def test_mrr_average():
    results = [
        ([1, 2, 3], [1]),    # RR = 1.0
        ([2, 1, 3], [1]),    # RR = 0.5
        ([3, 2, 1], [1]),    # RR = 1/3
    ]
    expected = (1.0 + 0.5 + 1/3) / 3
    assert mrr(results) == pytest.approx(expected)


def test_mrr_empty():
    assert mrr([]) == pytest.approx(0.0)


def test_mrr_multiple_relevant_uses_first_hit():
    """MRR counts the first hit; if doc 3 is at rank 1 and both 1 and 3 are relevant."""
    results = [([3, 1, 2], [1, 3])]
    assert mrr(results) == pytest.approx(1.0)
