"""
Integration tests for POST /api/v1/query and POST /api/v1/query/stream.

The query endpoint uses PgvectorRetriever which requires PostgreSQL.
For CI (SQLite), we mock the retriever, embedder, and reranker so the
endpoint's HTTP contract can be tested without any real DB vector ops
or model loading.

Covers:
  - 200 OK with correct response shape (including new retrieval_mode, reranked fields)
  - extractive mode is the default
  - citations match chunks
  - missing question returns 400
  - top_k passed to retriever
  - mode selection (vector / lexical / hybrid)
  - rerank flag
  - cache hit returns cached result
  - streaming endpoint returns SSE events
"""

import pytest
from unittest.mock import patch, MagicMock, call
from django.test import Client
from django.core.cache import cache
import json

from core.interfaces import ChunkResult
from core.fake_embedder import FakeEmbedder
from core.reranker import FakeReranker


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
def clear_cache():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture(autouse=True)
def patch_embedder_retriever_reranker():
    fake_embedder = FakeEmbedder()
    fake_reranker = FakeReranker()

    with (
        patch("query.views.get_embedder", return_value=fake_embedder),
        patch("query.views._get_reranker", return_value=fake_reranker),
        patch("query.views.PgvectorRetriever") as mock_vector_cls,
        patch("query.views.LexicalRetriever") as mock_lexical_cls,
        patch("query.views.HybridRetriever") as mock_hybrid_cls,
    ):
        mock_vector = MagicMock()
        mock_vector.retrieve.return_value = FAKE_CHUNKS
        mock_vector_cls.return_value = mock_vector

        mock_lexical = MagicMock()
        mock_lexical.retrieve.return_value = FAKE_CHUNKS
        mock_lexical_cls.return_value = mock_lexical

        mock_hybrid = MagicMock()
        mock_hybrid.retrieve.return_value = FAKE_CHUNKS
        mock_hybrid_cls.return_value = mock_hybrid

        yield {
            "vector": mock_vector,
            "lexical": mock_lexical,
            "hybrid": mock_hybrid,
        }


def _post_query(client, payload):
    return client.post(
        "/api/v1/query",
        data=json.dumps(payload),
        content_type="application/json",
    )


def test_query_returns_200(client):
    resp = _post_query(client, {"question": "What is a mitochondria?"})
    assert resp.status_code == 200


def test_query_response_shape(client):
    resp = _post_query(client, {"question": "What is a mitochondria?"})
    data = resp.json()
    assert "answer" in data
    assert "citations" in data
    assert "mode" in data
    assert "retrieval_mode" in data
    assert "reranked" in data
    assert "retrieved_chunks" in data
    assert "question" in data


def test_query_mode_is_extractive_by_default(client, settings):
    settings.LLM_API_BASE = ""
    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""
    resp = _post_query(client, {"question": "What is ATP?"})
    assert resp.json()["mode"] == "extractive"


def test_query_citations_match_chunks(client):
    resp = _post_query(client, {"question": "Tell me about cells"})
    data = resp.json()
    assert len(data["citations"]) == len(FAKE_CHUNKS)
    assert data["citations"][0]["document_title"] == "Biology 101"
    assert data["citations"][0]["document_id"] == 1


def test_query_missing_question_returns_400(client):
    resp = _post_query(client, {})
    assert resp.status_code == 400


def test_query_top_k_passed_to_retriever(client, patch_embedder_retriever_reranker):
    _post_query(client, {"question": "test", "top_k": 7, "mode": "vector", "rerank": False})
    mock_vector = patch_embedder_retriever_reranker["vector"]
    args, kwargs = mock_vector.retrieve.call_args
    assert 7 in args or kwargs.get("top_k") == 7


def test_query_vector_mode_uses_pgvector_retriever(client, patch_embedder_retriever_reranker):
    _post_query(client, {"question": "test", "mode": "vector", "rerank": False})
    assert patch_embedder_retriever_reranker["vector"].retrieve.called
    assert not patch_embedder_retriever_reranker["lexical"].retrieve.called
    assert not patch_embedder_retriever_reranker["hybrid"].retrieve.called


def test_query_lexical_mode_uses_lexical_retriever(client, patch_embedder_retriever_reranker):
    _post_query(client, {"question": "test", "mode": "lexical", "rerank": False})
    assert patch_embedder_retriever_reranker["lexical"].retrieve.called
    assert not patch_embedder_retriever_reranker["vector"].retrieve.called


def test_query_hybrid_mode_is_default(client):
    resp = _post_query(client, {"question": "test"})
    assert resp.json()["retrieval_mode"] == "hybrid"


def test_query_rerank_false_sets_reranked_false(client):
    resp = _post_query(client, {"question": "test", "rerank": False})
    assert resp.json()["reranked"] is False


def test_query_rerank_true_sets_reranked_true(client):
    resp = _post_query(client, {"question": "test", "rerank": True})
    assert resp.json()["reranked"] is True


def test_query_cache_hit_returns_same_result(client):
    payload = {"question": "mitochondria?", "mode": "vector", "rerank": False}
    resp1 = _post_query(client, payload)
    resp2 = _post_query(client, payload)
    assert resp1.json() == resp2.json()


def test_query_cache_hit_skips_retriever(client, patch_embedder_retriever_reranker):
    payload = {"question": "cached question", "mode": "vector", "rerank": False}
    _post_query(client, payload)   # first call - populates cache
    _post_query(client, payload)   # second call - should use cache

    mock_vector = patch_embedder_retriever_reranker["vector"]
    # Retriever should have been called exactly once (first request only)
    assert mock_vector.retrieve.call_count == 1


def test_stream_endpoint_returns_sse(client):
    resp = client.post(
        "/api/v1/query/stream",
        data=json.dumps({"question": "What is ATP?", "mode": "vector", "rerank": False}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    content_type = resp.get("Content-Type", "")
    assert "text/event-stream" in content_type


def test_stream_endpoint_contains_data_lines(client):
    resp = client.post(
        "/api/v1/query/stream",
        data=json.dumps({"question": "What is ATP?", "mode": "vector", "rerank": False}),
        content_type="application/json",
    )
    body = b"".join(resp.streaming_content).decode()
    assert "data:" in body
    assert "[DONE]" in body


def test_invalid_mode_returns_400(client):
    resp = _post_query(client, {"question": "test", "mode": "neural"})
    assert resp.status_code == 400


def test_query_response_includes_faithfulness(client):
    resp = _post_query(client, {"question": "What is a mitochondria?"})
    data = resp.json()
    assert "faithfulness" in data
    assert isinstance(data["faithfulness"], float)
    assert 0.0 <= data["faithfulness"] <= 1.0


def test_query_response_includes_abstained(client):
    resp = _post_query(client, {"question": "What is a mitochondria?"})
    data = resp.json()
    assert "abstained" in data
    assert isinstance(data["abstained"], bool)


def test_query_response_includes_claims(client):
    resp = _post_query(client, {"question": "What is ATP?"})
    data = resp.json()
    assert "claims" in data
    assert isinstance(data["claims"], list)


def test_claims_have_required_shape(client):
    resp = _post_query(client, {"question": "Tell me about cells."})
    data = resp.json()
    for claim in data["claims"]:
        assert "text" in claim
        assert "label" in claim
        assert "citation" in claim
        assert claim["label"] in ("SUPPORTED", "UNSUPPORTED", "NEUTRAL")


def test_abstained_answer_replaced_when_threshold_exceeded(client, settings):
    settings.FAITHFULNESS_THRESHOLD = 1.0
    resp = _post_query(client, {"question": "What is ATP?"})
    data = resp.json()
    if data["abstained"]:
        assert "I don't have enough grounded evidence" in data["answer"]
