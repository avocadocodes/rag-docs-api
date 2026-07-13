"""
Reranking implementations.

CrossEncoderReranker — uses a sentence-transformers CrossEncoder model to score
  each (query, chunk) pair independently.  The model is loaded lazily and cached
  for the process lifetime so the first call pays the load cost; subsequent calls
  are cheap.

  Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  This model was fine-tuned on MS MARCO passage ranking.  It takes a
  (query, passage) pair and returns a relevance score; higher is more relevant.
  It is much slower than bi-encoder retrieval but significantly more accurate
  for ranking — the cross-attention mechanism sees both texts together.

FakeReranker — deterministic, no model download.  Scores by lexical overlap
  (shared lowercase words) so tests get predictable ordering without torch.

Both satisfy RerankerProtocol from core.interfaces.
"""

from __future__ import annotations

from django.conf import settings

from core.interfaces import ChunkResult

_cross_encoder_instance = None


class CrossEncoderReranker:
    """Rerank chunks using a cross-encoder relevance model."""

    _DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or getattr(
            settings, "RERANKER_MODEL", self._DEFAULT_MODEL
        )
        self._model = None  # loaded lazily

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, chunks: list[ChunkResult]) -> list[ChunkResult]:
        if not chunks:
            return chunks
        self._load()
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked]


def get_reranker() -> CrossEncoderReranker:
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        _cross_encoder_instance = CrossEncoderReranker()
    return _cross_encoder_instance


class FakeReranker:
    """
    Deterministic reranker for tests — no model, no torch.

    Scores each chunk by the number of lowercase words it shares with the
    query.  Ties are broken by original list order (stable sort).
    """

    def rerank(self, query: str, chunks: list[ChunkResult]) -> list[ChunkResult]:
        query_words = set(query.lower().split())
        scored = []
        for chunk in chunks:
            chunk_words = set(chunk.text.lower().split())
            score = len(query_words & chunk_words)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]
