"""
Tests for the FakeEmbedder.

Verifies determinism, dimensionality, and unit-length normalisation.
"""

import math
import pytest
from core.fake_embedder import FakeEmbedder


@pytest.fixture
def embedder():
    return FakeEmbedder()


def test_dim_matches_property(embedder):
    vec = embedder.embed("hello")
    assert len(vec) == embedder.dim


def test_deterministic(embedder):
    a = embedder.embed("the quick brown fox")
    b = embedder.embed("the quick brown fox")
    assert a == b


def test_different_inputs_produce_different_vectors(embedder):
    a = embedder.embed("hello world")
    b = embedder.embed("goodbye world")
    assert a != b


def test_unit_length(embedder):
    vec = embedder.embed("some text")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6
