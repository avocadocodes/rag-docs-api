from __future__ import annotations
from django.conf import settings
from core.interfaces import ClaimVerdict

_nli_instance = None


class NliVerifier:
    _DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
    _ENTAILMENT_IDX = 1

    def __init__(self, model_name: str | None = None, threshold: float = 0.5) -> None:
        self._model_name = model_name or getattr(settings, "NLI_MODEL", self._DEFAULT_MODEL)
        self._threshold = threshold
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            self._model = CrossEncoder(self._model_name, default_activation_function=None)

    def verify(self, claim: str, evidence: list[str]) -> ClaimVerdict:
        if not evidence:
            return ClaimVerdict(label="UNSUPPORTED", score=0.0, best_evidence_index=None)
        self._load()
        from core.claim_splitter import split_into_claims  # noqa: PLC0415

        # NLI models are trained on single-sentence premise/hypothesis pairs and
        # score poorly when the premise is a long multi-sentence passage. Split
        # each evidence chunk into sentences and score the claim against each,
        # taking the max entailment. Track the source chunk index for citations.
        sentence_pairs: list[tuple[int, str]] = []
        for chunk_idx, chunk in enumerate(evidence):
            for sentence in split_into_claims(chunk) or [chunk]:
                sentence_pairs.append((chunk_idx, sentence))

        pairs = [(sentence, claim) for _, sentence in sentence_pairs]
        raw_scores = self._model.predict(pairs, apply_softmax=True)
        best_idx = None
        best_ent = -1.0
        for (chunk_idx, _), scores in zip(sentence_pairs, raw_scores):
            ent = float(scores[self._ENTAILMENT_IDX])
            if ent > best_ent:
                best_ent = ent
                best_idx = chunk_idx
        if best_ent >= self._threshold:
            label = "SUPPORTED"
        elif best_ent < 0.1:
            label = "UNSUPPORTED"
        else:
            label = "NEUTRAL"
        return ClaimVerdict(label=label, score=round(best_ent, 4), best_evidence_index=best_idx)


def get_verifier() -> NliVerifier:
    global _nli_instance
    if _nli_instance is None:
        threshold = getattr(settings, "FAITHFULNESS_THRESHOLD", 0.5)
        _nli_instance = NliVerifier(threshold=threshold)
    return _nli_instance


class FakeVerifier:
    _STOPWORDS = frozenset(
        "a an the is are was were be been being have has had do does did "
        "will would could should may might shall to of in on at by for "
        "with about and or but not this that it its".split()
    )

    def __init__(self, threshold: float = 0.3) -> None:
        self._threshold = threshold

    def _content_words(self, text: str) -> set[str]:
        words = set(text.lower().split())
        return words - self._STOPWORDS or words

    def verify(self, claim: str, evidence: list[str]) -> ClaimVerdict:
        if not evidence:
            return ClaimVerdict(label="UNSUPPORTED", score=0.0, best_evidence_index=None)
        claim_words = self._content_words(claim)
        if not claim_words:
            return ClaimVerdict(label="NEUTRAL", score=0.0, best_evidence_index=None)
        best_idx = -1
        best_ratio = 0.0
        for i, chunk in enumerate(evidence):
            chunk_words = self._content_words(chunk)
            overlap = len(claim_words & chunk_words)
            ratio = overlap / len(claim_words)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
        if best_ratio >= self._threshold:
            label = "SUPPORTED"
        else:
            label = "UNSUPPORTED"
        return ClaimVerdict(
            label=label,
            score=round(best_ratio, 4),
            best_evidence_index=best_idx if best_ratio > 0 else None,
        )
