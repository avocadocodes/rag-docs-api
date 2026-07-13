"""
Django settings for rag-docs-api.

Runtime configuration is read from environment variables (or a .env file).
All variables have sensible localhost defaults so the project starts without
any configuration for local development.

Required for production:
  SECRET_KEY      — Django secret key
  DATABASE_URL    — postgres://user:pass@host:5432/dbname
                    (or individual PG_* vars below)

Optional LLM integration (leave unset to use extractive-answer fallback):
  LLM_API_BASE    — OpenAI-compatible base URL, e.g. https://api.openai.com/v1
  LLM_API_KEY     — API key for the LLM provider
  LLM_MODEL       — Model name, e.g. gpt-4o-mini
"""

import os
import dj_database_url
from decouple import config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

SECRET_KEY = config("SECRET_KEY", default="dev-insecure-secret-key-change-in-production")
DEBUG = config("DEBUG", default=True, cast=bool)
ALLOWED_HOSTS = config("ALLOWED_HOSTS", default="*").split(",")

# ---------------------------------------------------------------------------
# Application definition
# ---------------------------------------------------------------------------

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "rest_framework",
    "drf_spectacular",
    "documents",
    "query",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_default_db_url = (
    "postgres://{user}:{password}@{host}:{port}/{name}".format(
        user=config("PG_USER", default="postgres"),
        password=config("PG_PASSWORD", default="postgres"),
        host=config("PG_HOST", default="localhost"),
        port=config("PG_PORT", default="5432"),
        name=config("PG_NAME", default="ragdocs"),
    )
)

DATABASE_URL = config("DATABASE_URL", default=_default_db_url)

DATABASES = {
    "default": dj_database_url.parse(
        DATABASE_URL,
        conn_max_age=600,
        conn_health_checks=True,
    )
}

# ---------------------------------------------------------------------------
# Internationalization / Static
# ---------------------------------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# REST Framework
# ---------------------------------------------------------------------------

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
}

# ---------------------------------------------------------------------------
# drf-spectacular
# ---------------------------------------------------------------------------

SPECTACULAR_SETTINGS = {
    "TITLE": "RAG Docs API",
    "DESCRIPTION": (
        "Retrieval-Augmented Generation document Q&A API. "
        "Upload documents, search by semantic similarity, get grounded answers with citations."
    ),
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

# ---------------------------------------------------------------------------
# Embedding / Chunking
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = config("EMBEDDING_MODEL", default="all-MiniLM-L6-v2")
EMBEDDING_DIM = config("EMBEDDING_DIM", default=384, cast=int)

CHUNK_SIZE = config("CHUNK_SIZE", default=500, cast=int)   # approximate tokens
CHUNK_OVERLAP = config("CHUNK_OVERLAP", default=50, cast=int)

# ---------------------------------------------------------------------------
# LLM integration (optional — all three must be set to activate)
# ---------------------------------------------------------------------------

LLM_API_BASE = config("LLM_API_BASE", default="")
LLM_API_KEY = config("LLM_API_KEY", default="")
LLM_MODEL = config("LLM_MODEL", default="")

# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

RERANKER_MODEL = config("RERANKER_MODEL", default="cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------------------
# Cache (Redis when available; falls back to in-memory so tests need no Redis)
# ---------------------------------------------------------------------------

REDIS_URL = config("REDIS_URL", default="")

if REDIS_URL:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "TIMEOUT": 300,  # 5 minutes
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        }
    }

# Query result cache TTL in seconds
QUERY_CACHE_TTL = config("QUERY_CACHE_TTL", default=300, cast=int)
