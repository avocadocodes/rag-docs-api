"""
Unit tests for query.answer — extractive fallback and citation assembly.

These tests run without a database or network.
"""

import pytest
from core.interfaces import ChunkResult
from query.answer import generate_answer, _extractive_answer, _build_citations


def _make_chunk(text, chunk_index=0, doc_id=1, doc_title="Test Doc", similarity=0.9):
    return ChunkResult(
        text=text,
        chunk_index=chunk_index,
        document_id=doc_id,
        document_title=doc_title,
        similarity=similarity,
    )


def test_extractive_answer_contains_chunk_text():
    chunk = _make_chunk("Penguins live in Antarctica.")
    result = _extractive_answer([chunk])
    assert "Penguins live in Antarctica." in result.answer
    assert result.mode == "extractive"


def test_extractive_answer_no_chunks():
    result = _extractive_answer([])
    assert "No relevant content found" in result.answer
    assert result.citations == []


def test_citations_contain_all_chunks():
    chunks = [
        _make_chunk("first chunk", chunk_index=0, doc_id=1, doc_title="Doc A", similarity=0.95),
        _make_chunk("second chunk", chunk_index=1, doc_id=2, doc_title="Doc B", similarity=0.80),
    ]
    citations = _build_citations(chunks)
    assert len(citations) == 2
    assert citations[0].document_title == "Doc A"
    assert citations[1].document_id == 2


def test_generate_answer_uses_extractive_when_llm_not_configured(settings):
    settings.LLM_API_BASE = ""
    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""

    chunks = [_make_chunk("The capital of France is Paris.")]
    result = generate_answer("What is the capital of France?", chunks)
    assert result.mode == "extractive"
    assert "Paris" in result.answer


def test_extractive_answer_numbers_citations():
    chunks = [
        _make_chunk("Alpha text", chunk_index=0),
        _make_chunk("Beta text", chunk_index=1),
    ]
    result = _extractive_answer(chunks)
    assert "[1]" in result.answer
    assert "[2]" in result.answer
