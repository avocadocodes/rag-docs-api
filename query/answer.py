"""
Answer generation — pluggable LLM vs extractive fallback, with streaming support.

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

Streaming:
    stream_answer() yields text tokens/chunks as they arrive from the LLM,
    or yields the extractive answer in sentence-sized pieces for consistency.

The caller always receives the same response shape regardless of which mode
is active; the "mode" field in the response tells the client which was used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

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
            pass
    return _extractive_answer(chunks)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def stream_answer(question: str, chunks: list[ChunkResult]) -> Iterator[str]:
    """
    Yield answer text incrementally.

    LLM mode: streams the SSE delta tokens from the chat completions endpoint
    (stream=True).  Each yielded string is a token or small token group.

    Extractive mode: yields one chunk's text at a time, separated by a blank
    line, so the client sees progressive reveal.
    """
    if _llm_configured():
        try:
            yield from _stream_llm(question, chunks)
            return
        except Exception:
            pass
    yield from _stream_extractive(chunks)


def _stream_extractive(chunks: list[ChunkResult]) -> Iterator[str]:
    if not chunks:
        yield "No relevant content found."
        return
    for i, chunk in enumerate(chunks, start=1):
        yield f"[{i}] {chunk.text}"
        if i < len(chunks):
            yield "\n\n"


def _stream_llm(question: str, chunks: list[ChunkResult]) -> Iterator[str]:
    import urllib.request  # noqa: PLC0415

    context_parts = [
        f"[Source {i}: {c.document_title}, chunk {c.chunk_index}]\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
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
            "stream": True,
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

    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
