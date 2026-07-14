from __future__ import annotations
import re
from dataclasses import dataclass
from django.conf import settings
from core.interfaces import ChunkResult
from core.claim_splitter import split_into_claims

# Extractive answers prefix each sentence with a "[n]" citation marker; strip it
# so the marker text does not corrupt the entailment check.
_CITATION_MARKER = re.compile(r"^\s*\[\d+\]\s*")

_ABSTAIN_MESSAGE = (
    "I don't have enough grounded evidence to answer that. "
    "The retrieved sources did not sufficiently support the generated claims. "
    "Please check the retrieved_chunks for the available evidence."
)


@dataclass
class VerificationResult:
    answer: str
    faithfulness: float
    abstained: bool
    claims: list[dict]


def _get_verifier():
    backend = getattr(settings, "VERIFIER_BACKEND", "real")
    if backend == "fake":
        from core.verifier import FakeVerifier  # noqa: PLC0415
        threshold = getattr(settings, "FAITHFULNESS_THRESHOLD", 0.5)
        return FakeVerifier(threshold=threshold)
    from core.verifier import get_verifier  # noqa: PLC0415
    return get_verifier()


def no_evidence_result() -> VerificationResult:
    """Abstention result used when retrieval found no sufficiently relevant evidence."""
    return VerificationResult(
        answer=_ABSTAIN_MESSAGE, faithfulness=0.0, abstained=True, claims=[]
    )


class AnswerVerification:
    def __init__(self, verifier=None, faithfulness_threshold: float | None = None) -> None:
        self._verifier = verifier or _get_verifier()
        self._threshold = (
            faithfulness_threshold
            if faithfulness_threshold is not None
            else getattr(settings, "FAITHFULNESS_THRESHOLD", 0.5)
        )

    def verify(self, answer: str, chunks: list[ChunkResult]) -> VerificationResult:
        claims_text = split_into_claims(answer)
        evidence = [c.text for c in chunks]

        if not claims_text:
            return VerificationResult(answer=answer, faithfulness=0.0, abstained=False, claims=[])

        per_claim: list[dict] = []
        supported = 0

        for claim_str in claims_text:
            claim_str = _CITATION_MARKER.sub("", claim_str).strip()
            if not claim_str:
                continue
            verdict = self._verifier.verify(claim_str, evidence)
            if verdict.label == "SUPPORTED":
                supported += 1

            citation = None
            if verdict.best_evidence_index is not None and verdict.best_evidence_index < len(chunks):
                best = chunks[verdict.best_evidence_index]
                citation = {
                    "document_id": best.document_id,
                    "document_title": best.document_title,
                    "chunk_index": best.chunk_index,
                }

            per_claim.append({"text": claim_str, "label": verdict.label, "citation": citation})

        faithfulness = round(supported / len(claims_text), 4)
        abstained = faithfulness < self._threshold

        final_answer = _ABSTAIN_MESSAGE if abstained else answer
        return VerificationResult(
            answer=final_answer,
            faithfulness=faithfulness,
            abstained=abstained,
            claims=per_claim,
        )
