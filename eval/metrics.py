"""
Retrieval evaluation metrics.

recall_at_k  — fraction of questions where at least one relevant document
               appears in the top-k retrieved results.

mrr          — Mean Reciprocal Rank: mean of 1/rank where rank is the
               position of the first relevant document in the result list
               (0.0 if not found).

Both functions operate on lists of (retrieved_doc_ids, relevant_doc_ids)
pairs where each element is a list of integer document IDs.
"""

from __future__ import annotations


def recall_at_k(
    results: list[tuple[list[int], list[int]]],
    k: int,
) -> float:
    """
    Compute Recall@k.

    A question is considered a hit if any of its relevant document IDs
    appear in the first k retrieved document IDs.

    Parameters
    ----------
    results : list of (retrieved_doc_ids, relevant_doc_ids).
              retrieved_doc_ids is already in ranked order.
    k       : cutoff rank

    Returns
    -------
    Fraction of questions that are hits (float in [0, 1]).
    """
    if not results:
        return 0.0
    hits = 0
    for retrieved, relevant in results:
        top_k_set = set(retrieved[:k])
        if top_k_set & set(relevant):
            hits += 1
    return hits / len(results)


def mrr(results: list[tuple[list[int], list[int]]]) -> float:
    """
    Compute Mean Reciprocal Rank.

    For each question, find the rank of the first relevant document in
    the retrieved list (1-indexed).  The reciprocal rank is 1/rank.
    If no relevant document is found, the reciprocal rank is 0.

    Parameters
    ----------
    results : list of (retrieved_doc_ids, relevant_doc_ids).

    Returns
    -------
    Mean reciprocal rank (float in [0, 1]).
    """
    if not results:
        return 0.0
    total = 0.0
    for retrieved, relevant in results:
        relevant_set = set(relevant)
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                total += 1.0 / rank
                break
    return total / len(results)
