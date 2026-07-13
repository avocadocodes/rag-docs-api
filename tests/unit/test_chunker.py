"""
Unit tests for core.chunker.

All tests run without a database or network.
Settings are applied via pytest-django (DJANGO_SETTINGS_MODULE=config.test_settings).
"""

import pytest
from core.chunker import chunk_text


def test_empty_string_returns_empty_list():
    assert chunk_text("", chunk_size=10, chunk_overlap=2) == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\t  ", chunk_size=10, chunk_overlap=2) == []


def test_single_chunk_when_text_shorter_than_window():
    text = "hello world foo bar"
    result = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert result == [text]


def test_correct_chunk_count():
    # 20 tokens, chunk_size=6, overlap=2 → step=4
    # starts: 0, 4, 8, 12, 16  → 5 chunks
    tokens = list(range(20))
    text = " ".join(str(t) for t in tokens)
    result = chunk_text(text, chunk_size=6, chunk_overlap=2)
    assert len(result) == 5


def test_overlap_tokens_appear_in_consecutive_chunks():
    words = [f"w{i}" for i in range(10)]
    text = " ".join(words)
    overlap = 2
    chunks = chunk_text(text, chunk_size=5, chunk_overlap=overlap)

    # The last `overlap` words of chunk N should match the first `overlap`
    # words of chunk N+1, provided both chunks are at least `overlap` tokens long.
    for i in range(len(chunks) - 1):
        head_of_next = chunks[i + 1].split()[:overlap]
        if len(head_of_next) < overlap:
            # The next chunk is shorter than the overlap window (last tail chunk)
            continue
        tail_of_current = chunks[i].split()[-overlap:]
        assert tail_of_current == head_of_next, (
            f"No overlap between chunk {i} and {i+1}: "
            f"tail={tail_of_current}, head={head_of_next}"
        )


def test_overlap_gte_chunk_size_raises():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("a b c d", chunk_size=3, chunk_overlap=3)


def test_single_word_text():
    result = chunk_text("hello", chunk_size=10, chunk_overlap=2)
    assert result == ["hello"]


def test_exact_chunk_size_text():
    text = " ".join(["word"] * 6)
    result = chunk_text(text, chunk_size=6, chunk_overlap=2)
    # Exactly one full chunk, then the remaining 2-token tail
    assert len(result) >= 1
    assert result[0] == text
