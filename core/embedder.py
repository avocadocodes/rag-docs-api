"""
Real embedder — wraps sentence-transformers.

The model is loaded lazily on first use and cached for the process lifetime.
At import time nothing heavy is loaded, so the module is safe to import in
test environments where sentence-transformers is present but we don't want
the model to load.

Usage:
    from core.embedder import get_embedder
    embedder = get_embedder()       # returns the singleton
    vector = embedder.embed("hello world")
"""

from __future__ import annotations

from django.conf import settings

_instance: "SentenceTransformerEmbedder | None" = None


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int) -> None:
        self._model_name = model_name
        self._dim = dim
        self._model = None  # loaded lazily

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> list[float]:
        self._load()
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    @property
    def dim(self) -> int:
        return self._dim


def get_embedder() -> SentenceTransformerEmbedder:
    global _instance
    if _instance is None:
        _instance = SentenceTransformerEmbedder(
            model_name=settings.EMBEDDING_MODEL,
            dim=settings.EMBEDDING_DIM,
        )
    return _instance
