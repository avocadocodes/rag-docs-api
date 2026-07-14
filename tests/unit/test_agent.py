import pytest
from unittest.mock import patch, MagicMock
from core.interfaces import ChunkResult
from core.fake_embedder import FakeEmbedder
from core.verifier import FakeVerifier


def _make_chunk(text, doc_id=1, chunk_index=0):
    return ChunkResult(
        text=text,
        chunk_index=chunk_index,
        document_id=doc_id,
        document_title="Test Doc",
        similarity=0.9,
    )


FAKE_CHUNKS = [
    _make_chunk("The sky is blue on a clear day."),
    _make_chunk("Clouds form in the upper atmosphere.", doc_id=2, chunk_index=1),
]


class FakeHybridRetriever:
    def retrieve(self, query_embedding, query_text, top_k):
        return FAKE_CHUNKS[:top_k]


def test_agentic_pipeline_without_llm_uses_single_shot(settings):
    settings.LLM_API_BASE = ""
    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""
    settings.VERIFIER_BACKEND = "fake"
    settings.FAITHFULNESS_THRESHOLD = 0.5

    from query.agent import run_agentic_pipeline

    embedder = FakeEmbedder()
    verifier = FakeVerifier(threshold=0.3)

    result = run_agentic_pipeline(
        question="What color is the sky?",
        embedder=embedder,
        retriever=FakeHybridRetriever(),
        verifier=verifier,
        top_k=2,
    )

    assert "answer" in result
    assert "faithfulness" in result
    assert "abstained" in result
    assert "claims" in result


def test_agentic_pipeline_with_llm_decomposes_question(settings):
    settings.LLM_API_BASE = "http://fake-llm"
    settings.LLM_API_KEY = "test-key"
    settings.LLM_MODEL = "fake-model"
    settings.VERIFIER_BACKEND = "fake"
    settings.FAITHFULNESS_THRESHOLD = 0.5

    from query.agent import run_agentic_pipeline

    embedder = FakeEmbedder()
    verifier = FakeVerifier(threshold=0.3)

    mock_decompose = MagicMock(return_value=["What color is the sky?", "How do clouds form?"])
    mock_llm_answer_result = MagicMock()
    mock_llm_answer_result.answer = "The sky is blue. Clouds form in the upper atmosphere."
    mock_llm_answer_result.mode = "llm"
    mock_llm_answer_result.citations = []
    mock_generate = MagicMock(return_value=mock_llm_answer_result)

    with (
        patch("query.agent._decompose_question", mock_decompose),
        patch("query.agent.generate_answer", mock_generate),
    ):
        result = run_agentic_pipeline(
            question="What color is the sky and how do clouds form?",
            embedder=embedder,
            retriever=FakeHybridRetriever(),
            verifier=verifier,
            top_k=2,
        )

    mock_decompose.assert_called_once()
    assert "answer" in result
    assert "faithfulness" in result


def test_agentic_pipeline_retries_on_low_faithfulness(settings):
    settings.LLM_API_BASE = "http://fake-llm"
    settings.LLM_API_KEY = "test-key"
    settings.LLM_MODEL = "fake-model"
    settings.VERIFIER_BACKEND = "fake"
    settings.FAITHFULNESS_THRESHOLD = 0.5

    from query.agent import run_agentic_pipeline

    embedder = FakeEmbedder()
    verifier = FakeVerifier(threshold=0.3)
    call_count = [0]

    def fake_llm_answer(question, chunks):
        call_count[0] += 1
        result = MagicMock()
        if call_count[0] == 1:
            result.answer = "Quantum mechanics governs subatomic particles."
        else:
            result.answer = "The sky is blue on a clear day."
        result.mode = "llm"
        result.citations = []
        return result

    mock_decompose = MagicMock(return_value=["What color is the sky?"])

    with (
        patch("query.agent._decompose_question", mock_decompose),
        patch("query.agent.generate_answer", fake_llm_answer),
    ):
        result = run_agentic_pipeline(
            question="What color is the sky?",
            embedder=embedder,
            retriever=FakeHybridRetriever(),
            verifier=verifier,
            top_k=2,
        )

    assert call_count[0] == 2
    assert "answer" in result


def test_agentic_pipeline_caps_iterations_at_two(settings):
    settings.LLM_API_BASE = "http://fake-llm"
    settings.LLM_API_KEY = "test-key"
    settings.LLM_MODEL = "fake-model"
    settings.VERIFIER_BACKEND = "fake"
    settings.FAITHFULNESS_THRESHOLD = 0.9

    from query.agent import run_agentic_pipeline

    embedder = FakeEmbedder()
    verifier = FakeVerifier(threshold=0.3)
    call_count = [0]

    def always_bad_answer(question, chunks):
        call_count[0] += 1
        result = MagicMock()
        result.answer = "Quantum tunneling is a subatomic phenomenon."
        result.mode = "llm"
        result.citations = []
        return result

    mock_decompose = MagicMock(return_value=["What color is the sky?"])

    with (
        patch("query.agent._decompose_question", mock_decompose),
        patch("query.agent.generate_answer", always_bad_answer),
    ):
        result = run_agentic_pipeline(
            question="What color is the sky?",
            embedder=embedder,
            retriever=FakeHybridRetriever(),
            verifier=verifier,
            top_k=2,
        )

    assert call_count[0] <= 2
