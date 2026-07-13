"""
Answer generation — pluggable LLM vs extractive fallback.

Design rationale
----------------
The service works out-of-the-box with zero external dependencies:

  Default (extractive):
    Concatenate the retrieved chunks and return them directly as the answer,
    with inline citations.  No network, no API key, no model download beyond
    the embedding model.  Suitable for development, CI, and customers who
    don't want to connect an LLM.

  LLM mode (opt-in):
    If LLM_API_BASE + LLM_API_KEY + LLM_MODEL are all set in the environment,
    the retrieved chunks are sent to any OpenAI-compatible chat endpoint as
    grounding context.  The LLM is instructed to answer using only the provided
    context and to cite sources.

The caller always receives the same response shape regardless of which mode
is active; the "mode" field in the response tells the client which was used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from django.conf import settings

from core.interfaces import ChunkResult


@dataclass
class Citation:
    document_id: int
    document_title: str
    chunk_index: int
    similarity: float


@dataclass
class AnswerResult:
    answer: str
    mode: str          # "extractive" | "llm"
    citations: list[Citation]


def _build_citations(chunks: list[ChunkResult]) -> list[Citation]:
    return [
        Citation(
            document_id=c.document_id,
            document_title=c.document_title,
            chunk_index=c.chunk_index,
            similarity=c.similarity,
        )
        for c in chunks
    ]


def _extractive_answer(chunks: list[ChunkResult]) -> AnswerResult:
    """Compose the answer by concatenating retrieved chunks with citation markers."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] {chunk.text}")

    answer = "\n\n".join(parts) if parts else "No relevant content found."
    return AnswerResult(
        answer=answer,
        mode="extractive",
        citations=_build_citations(chunks),
    )


def _llm_answer(question: str, chunks: list[ChunkResult]) -> AnswerResult:
    """Call an OpenAI-compatible chat endpoint and return the generated answer."""
    import urllib.request  # stdlib only; avoids an openai package dep

    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk.document_title}, chunk {chunk.chunk_index}]\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on "
        "the provided context. Always cite the source numbers (e.g. [Source 1]) "
        "when you use information from them. If the context does not contain "
        "enough information to answer the question, say so."
    )
    user_prompt = f"Context:\n\n{context}\n\nQuestion: {question}"

    payload = json.dumps(
        {
            "model": settings.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
    ).encode()

    req = urllib.request.Request(
        f"{settings.LLM_API_BASE.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.LLM_API_KEY}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    answer = data["choices"][0]["message"]["content"]
    return AnswerResult(
        answer=answer,
        mode="llm",
        citations=_build_citations(chunks),
    )


def _llm_configured() -> bool:
    return bool(
        getattr(settings, "LLM_API_BASE", "")
        and getattr(settings, "LLM_API_KEY", "")
        and getattr(settings, "LLM_MODEL", "")
    )


def generate_answer(question: str, chunks: list[ChunkResult]) -> AnswerResult:
    """Entry point: choose LLM or extractive based on settings."""
    if _llm_configured():
        try:
            return _llm_answer(question, chunks)
        except Exception:
            # Fall through to extractive if LLM call fails
            pass
    return _extractive_answer(chunks)
