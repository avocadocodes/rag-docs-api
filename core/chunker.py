"""
Text chunking utility.

Splits a document into overlapping windows measured in whitespace-delimited
tokens (words). This is fast, requires no external library, and produces
deterministic results that are easy to test.

    chunk_size    — target window size in tokens (default from settings)
    chunk_overlap — number of tokens shared between adjacent chunks
"""

from __future__ import annotations

from django.conf import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Return a list of overlapping text chunks.

    Each chunk contains at most *chunk_size* whitespace-delimited tokens.
    Consecutive chunks share *chunk_overlap* tokens at their boundary.

    Edge cases:
    - Empty / whitespace-only text returns an empty list.
    - Text shorter than chunk_size returns a single chunk.
    - overlap >= chunk_size raises ValueError to prevent infinite loops.
    """
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.CHUNK_OVERLAP

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    tokens = text.split()
    if not tokens:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(" ".join(tokens[start:end]))
        start += step

    return chunks
