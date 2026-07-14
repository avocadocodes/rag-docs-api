import pytest
from core.claim_splitter import split_into_claims


def test_single_sentence_returns_one_claim():
    result = split_into_claims("The sky is blue.")
    assert result == ["The sky is blue."]


def test_two_sentences_returns_two_claims():
    result = split_into_claims("The sky is blue. The grass is green.")
    assert len(result) == 2
    assert result[0] == "The sky is blue."
    assert result[1] == "The grass is green."


def test_empty_string_returns_empty_list():
    result = split_into_claims("")
    assert result == []


def test_whitespace_only_returns_empty_list():
    result = split_into_claims("   \n  ")
    assert result == []


def test_strips_whitespace_from_claims():
    result = split_into_claims("  Hello world.  Next sentence. ")
    assert all(c == c.strip() for c in result)
    assert all(len(c) > 0 for c in result)


def test_question_mark_splits():
    result = split_into_claims("Is this true? Yes it is.")
    assert len(result) == 2


def test_exclamation_splits():
    result = split_into_claims("Watch out! The floor is wet.")
    assert len(result) == 2


def test_no_trailing_empty_claims():
    result = split_into_claims("A sentence.")
    assert all(len(c) > 0 for c in result)


def test_abbreviations_do_not_split():
    result = split_into_claims("Dr. Smith works at the lab.")
    assert all(len(c) > 0 for c in result)


from core.verifier import FakeVerifier
from core.interfaces import ClaimVerdict


def test_fake_verifier_supported_when_claim_words_in_evidence():
    v = FakeVerifier(threshold=0.3)
    verdict = v.verify(
        claim="The sky is blue",
        evidence=["The sky is blue on a clear day.", "Grass is green."],
    )
    assert verdict.label == "SUPPORTED"
    assert verdict.score >= 0.3
    assert verdict.best_evidence_index == 0


def test_fake_verifier_unsupported_when_no_overlap():
    v = FakeVerifier(threshold=0.3)
    verdict = v.verify(
        claim="Penguins live on Mars",
        evidence=["Clouds form in the atmosphere.", "Rain falls from clouds."],
    )
    assert verdict.label == "UNSUPPORTED"
    assert verdict.score < 0.3


def test_fake_verifier_empty_evidence_is_unsupported():
    v = FakeVerifier(threshold=0.3)
    verdict = v.verify(claim="Something happened", evidence=[])
    assert verdict.label == "UNSUPPORTED"
    assert verdict.best_evidence_index is None


def test_fake_verifier_best_evidence_index_is_highest_overlap():
    v = FakeVerifier(threshold=0.1)
    verdict = v.verify(
        claim="token expires after ninety days",
        evidence=[
            "Tokens last for ninety days.",
            "Login with your password.",
        ],
    )
    assert verdict.best_evidence_index == 0


def test_fake_verifier_claim_verdict_is_dataclass():
    v = FakeVerifier()
    verdict = v.verify("hello world", ["hello world is great"])
    assert isinstance(verdict, ClaimVerdict)
    assert verdict.label in ("SUPPORTED", "UNSUPPORTED", "NEUTRAL")
