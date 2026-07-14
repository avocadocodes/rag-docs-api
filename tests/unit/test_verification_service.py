import pytest
from core.interfaces import ChunkResult, ClaimVerdict
from core.verifier import FakeVerifier
from query.verification import AnswerVerification, VerificationResult


def _make_chunk(text, doc_title="Doc", chunk_index=0, doc_id=1, similarity=0.9):
    return ChunkResult(
        text=text,
        chunk_index=chunk_index,
        document_id=doc_id,
        document_title=doc_title,
        similarity=similarity,
    )


CHUNKS = [
    _make_chunk("The sky is blue on a clear day.", doc_title="Weather 101"),
    _make_chunk("Grass is green due to chlorophyll.", doc_title="Plant Bio"),
]


def test_verification_result_has_required_fields():
    verifier = FakeVerifier(threshold=0.3)
    svc = AnswerVerification(verifier=verifier)
    answer = "The sky is blue. Grass is green."
    result = svc.verify(answer=answer, chunks=CHUNKS)
    assert isinstance(result, VerificationResult)
    assert 0.0 <= result.faithfulness <= 1.0
    assert isinstance(result.abstained, bool)
    assert isinstance(result.claims, list)


def test_all_supported_claims_high_faithfulness():
    verifier = FakeVerifier(threshold=0.3)
    svc = AnswerVerification(verifier=verifier, faithfulness_threshold=0.5)
    answer = "The sky is blue. Grass is green."
    result = svc.verify(answer=answer, chunks=CHUNKS)
    assert result.faithfulness == 1.0
    assert result.abstained is False


def test_no_matching_claims_abstains():
    verifier = FakeVerifier(threshold=0.3)
    svc = AnswerVerification(verifier=verifier, faithfulness_threshold=0.5)
    answer = "Quantum tunneling is a subatomic phenomenon. Neutrinos rarely interact."
    result = svc.verify(answer=answer, chunks=CHUNKS)
    assert result.faithfulness < 0.5
    assert result.abstained is True
    assert "I don't have enough grounded evidence" in result.answer


def test_abstained_answer_replaces_original_text():
    verifier = FakeVerifier(threshold=0.3)
    svc = AnswerVerification(verifier=verifier, faithfulness_threshold=0.5)
    # Use an answer with zero lexical overlap with CHUNKS (no content words shared)
    answer = "Quantum tunneling governs subatomic particles."
    result = svc.verify(answer=answer, chunks=CHUNKS)
    assert result.abstained is True
    assert answer not in result.answer


def test_claims_list_contains_per_claim_verdicts():
    verifier = FakeVerifier(threshold=0.3)
    svc = AnswerVerification(verifier=verifier)
    answer = "The sky is blue. Grass is green."
    result = svc.verify(answer=answer, chunks=CHUNKS)
    assert len(result.claims) == 2
    for claim_info in result.claims:
        assert "text" in claim_info
        assert "label" in claim_info
        assert "citation" in claim_info
        assert claim_info["label"] in ("SUPPORTED", "UNSUPPORTED", "NEUTRAL")


def test_empty_answer_returns_zero_faithfulness():
    verifier = FakeVerifier(threshold=0.3)
    svc = AnswerVerification(verifier=verifier)
    result = svc.verify(answer="", chunks=CHUNKS)
    assert result.faithfulness == 0.0


def test_citation_included_when_supported():
    verifier = FakeVerifier(threshold=0.1)
    svc = AnswerVerification(verifier=verifier)
    answer = "The sky is blue."
    result = svc.verify(answer=answer, chunks=CHUNKS)
    supported = [c for c in result.claims if c["label"] == "SUPPORTED"]
    if supported:
        assert supported[0]["citation"] is not None
