from __future__ import annotations
import json
import urllib.request
from typing import Protocol

from django.conf import settings

from core.interfaces import ChunkResult
from query.answer import generate_answer, _llm_configured
from query.verification import AnswerVerification, VerificationResult

_MAX_SUB_QUESTIONS = 3
_RETRY_TOP_K_MULTIPLIER = 3


class _RetrieverLike(Protocol):
    def retrieve(self, query_embedding: list[float], query_text: str, top_k: int) -> list[ChunkResult]:
        ...


def _decompose_question(question: str, max_sub: int = _MAX_SUB_QUESTIONS) -> list[str]:
    system = (
        "You are a query planning assistant. Split the user question into up to "
        f"{max_sub} simpler, self-contained sub-questions that can each be answered "
        "independently. Return ONLY a JSON array of strings. No explanation."
    )
    payload = json.dumps({
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"{settings.LLM_API_BASE.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"].strip()
        sub_questions = json.loads(content)
        if isinstance(sub_questions, list) and sub_questions:
            return [str(q) for q in sub_questions[:max_sub]]
    except Exception:
        pass
    return [question]


def _dedup_chunks(chunks: list[ChunkResult]) -> list[ChunkResult]:
    seen: set[tuple[int, int]] = set()
    out: list[ChunkResult] = []
    for c in chunks:
        key = (c.document_id, c.chunk_index)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _retrieve_for_questions(questions: list[str], embedder, retriever, top_k: int) -> list[ChunkResult]:
    all_chunks: list[ChunkResult] = []
    for q in questions:
        emb = embedder.embed(q)
        chunks = retriever.retrieve(emb, q, top_k)
        all_chunks.extend(chunks)
    return _dedup_chunks(all_chunks)


def run_agentic_pipeline(
    question: str,
    embedder,
    retriever,
    verifier,
    top_k: int = 5,
) -> dict:
    svc = AnswerVerification(verifier=verifier)

    if not _llm_configured():
        emb = embedder.embed(question)
        chunks = retriever.retrieve(emb, question, top_k)
        ans = generate_answer(question, chunks)
        verification = svc.verify(answer=ans.answer, chunks=chunks)
        return _build_result(ans, chunks, verification)

    sub_questions = _decompose_question(question)
    chunks = _retrieve_for_questions(sub_questions, embedder, retriever, top_k)

    ans = generate_answer(question, chunks)
    verification = svc.verify(answer=ans.answer, chunks=chunks)

    has_unsupported = any(c["label"] == "UNSUPPORTED" for c in verification.claims)
    if has_unsupported:
        retry_top_k = top_k * _RETRY_TOP_K_MULTIPLIER
        retry_chunks = _retrieve_for_questions(sub_questions, embedder, retriever, retry_top_k)
        ans = generate_answer(question, retry_chunks)
        verification = svc.verify(answer=ans.answer, chunks=retry_chunks)
        chunks = retry_chunks

    return _build_result(ans, chunks, verification)


def _build_result(ans, chunks: list[ChunkResult], verification: VerificationResult) -> dict:
    return {
        "answer": verification.answer,
        "mode": ans.mode,
        "faithfulness": verification.faithfulness,
        "abstained": verification.abstained,
        "claims": verification.claims,
        "citations": [
            {
                "document_id": c.document_id,
                "document_title": c.document_title,
                "chunk_index": c.chunk_index,
                "similarity": c.similarity,
            }
            for c in ans.citations
        ],
        "retrieved_chunks": [
            {
                "text": c.text,
                "chunk_index": c.chunk_index,
                "document_id": c.document_id,
                "document_title": c.document_title,
                "similarity": c.similarity,
            }
            for c in chunks
        ],
    }
