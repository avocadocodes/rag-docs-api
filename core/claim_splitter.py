from __future__ import annotations
import re

_SENTENCE_END = re.compile(r'(?<=[^0-9])([.?!])\s+')


def split_into_claims(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    parts = _SENTENCE_END.split(text)

    claims: list[str] = []
    i = 0
    while i < len(parts):
        fragment = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        if fragment:
            claim = fragment + punct if punct in ".?!" else fragment
            claims.append(claim)

    return [c for c in claims if c.strip()]
