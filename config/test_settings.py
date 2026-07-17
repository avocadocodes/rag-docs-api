"""
Test settings.

Differences from production settings:
  - SQLite in-memory database - no PostgreSQL, no pgvector extension needed.
  - Embedding dimensions reduced to 16 to match FakeEmbedder.
  - EMBEDDING_BACKEND = "fake" signals service code to skip model loading.
  - LLM integration disabled.
  - RERANKER_BACKEND = "fake" signals service code to use FakeReranker.
  - Cache uses locmem backend (no Redis required).
"""

from config.settings import *  # noqa: F401, F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

# Match FakeEmbedder.dim
EMBEDDING_DIM = 16

# Tell service code not to load sentence-transformers
EMBEDDING_BACKEND = "fake"

# Tell service code not to load the cross-encoder
RERANKER_BACKEND = "fake"

# LLM disabled
LLM_API_BASE = ""
LLM_API_KEY = ""
LLM_MODEL = ""
VERIFIER_BACKEND = "fake"
FAITHFULNESS_THRESHOLD = 0.5

# Use locmem cache (no Redis needed)
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}

RELEVANCE_THRESHOLD = 0.0
