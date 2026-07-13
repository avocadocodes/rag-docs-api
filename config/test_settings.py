"""
Test settings.

Differences from production settings:
  - SQLite in-memory database — no PostgreSQL, no pgvector extension needed.
  - Embedding dimensions reduced to 16 to match FakeEmbedder.
  - EMBEDDING_BACKEND = "fake" signals service code to skip model loading.
  - LLM integration disabled.
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

# LLM disabled
LLM_API_BASE = ""
LLM_API_KEY = ""
LLM_MODEL = ""
