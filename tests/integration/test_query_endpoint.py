"""
Integration tests for POST /api/v1/query.

The query endpoint uses PgvectorRetriever which requires PostgreSQL.
For CI (SQLite), we mock the retriever and embedder so the endpoint's
HTTP contract can be tested without any real DB vector ops.

Verifies:
  - 200 OK with correct shape
  - citations present
  - extractive mode is the default
  - missing question returns 400
"""

import pytest
from unittest.mock import patch, MagicMock
from django.test import Client
import json

from core.interfaces import ChunkResult
from core.fake_embedder import FakeEmbedder


FAKE_CHUNKS = [
    ChunkResult(
        text="The mitochondria is the powerhouse of the cell.",
        chunk_index=0,
        document_id=1,
        document_title="Biology 101",
        similarity=0.91,
    ),
    ChunkResult(
        text="Cells require ATP produced by mitochondria.",
        chunk_index=1,
        document_id=1,
        document_title="Biology 101",
        similarity=0.85,
    ),
]


@pytest.fixture
def client():
    return Client()


@pytest.fixture(autouse=True)
def patch_embedder_and_retriever():
    fake_embedder = FakeEmbedder()

    with (
        patch("query.views.get_embedder", return_value=fake_embedder),
        patch("query.views.PgvectorRetriever") as mock_retriever_cls,
    ):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = FAKE_CHUNKS
        mock_retriever_cls.return_value = mock_retriever
        yield mock_retriever


def test_query_returns_200(client):
    resp = client.post(
        "/api/v1/query",
        data=json.dumps({"question": "What is a mitochondria?"}),
        content_type="application/json",
    )
    assert resp.status_code == 200


def test_query_response_shape(client):
    resp = client.post(
        "/api/v1/query",
        data=json.dumps({"question": "What is a mitochondria?"}),
        content_type="application/json",
    )
    data = resp.json()
    assert "answer" in data
    assert "citations" in data
    assert "mode" in data
    assert "retrieved_chunks" in data
    assert "question" in data


def test_query_mode_is_extractive_by_default(client, settings):
    settings.LLM_API_BASE = ""
    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""

    resp = client.post(
        "/api/v1/query",
        data=json.dumps({"question": "What is ATP?"}),
        content_type="application/json",
    )
    assert resp.json()["mode"] == "extractive"


def test_query_citations_match_chunks(client):
    resp = client.post(
        "/api/v1/query",
        data=json.dumps({"question": "Tell me about cells"}),
        content_type="application/json",
    )
    data = resp.json()
    assert len(data["citations"]) == len(FAKE_CHUNKS)
    assert data["citations"][0]["document_title"] == "Biology 101"
    assert data["citations"][0]["document_id"] == 1


def test_query_missing_question_returns_400(client):
    resp = client.post(
        "/api/v1/query",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_query_top_k_passed_to_retriever(client, patch_embedder_and_retriever):
    mock_retriever = patch_embedder_and_retriever
    client.post(
        "/api/v1/query",
        data=json.dumps({"question": "test", "top_k": 7}),
        content_type="application/json",
    )
    call_kwargs = mock_retriever.retrieve.call_args
    # top_k should be 7
    assert call_kwargs[0][1] == 7 or call_kwargs[1].get("top_k") == 7
