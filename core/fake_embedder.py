"""
Deterministic fake embedder for tests and CI.

Produces a fixed-length vector by hashing the input text.  Vectors are
normalised to unit length so cosine similarity works correctly.  The same
input always produces the same vector; distinct inputs (usually) produce
distinct vectors, which is sufficient for testing retrieval ordering.

No network access, no model download, no GPU required.
"""

from __future__ import annotations

import hashlib
import math


_DIM = 16  # small enough to keep test overhead tiny


class FakeEmbedder:
    """Drop-in replacement for SentenceTransformerEmbedder in tests."""

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        # Build a DIM-length float vector from the digest bytes
        raw: list[float] = []
        for i in range(_DIM):
            # combine two bytes per dimension so we use more of the digest
            byte_val = digest[i % len(digest)]
            raw.append(float(byte_val) - 127.5)  # centre around 0

        # L2-normalise so dot product == cosine similarity
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    @property
    def dim(self) -> int:
        return _DIM
