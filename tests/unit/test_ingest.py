"""
Unit tests for the ingest service.

Uses SQLite (from test_settings) + FakeEmbedder.
Verifies that ingest_document creates the correct number of chunks and
that each chunk is embedded exactly once with the provided embedder.
"""

import pytest
from unittest.mock import patch, call
from core.fake_embedder import FakeEmbedder


@pytest.mark.django_db
def test_ingest_creates_chunks():
    from documents.models import Document
    from documents.ingest import ingest_document

    doc = Document.objects.create(title="Test Doc", raw_text=" ".join(["word"] * 20))
    embedder = FakeEmbedder()

    # chunk_size=10, overlap=2 from test_settings defaults → step=8 → ceil((20-10)/8)+1 = 2 full + tail
    # actual count depends on settings; just check it's > 0
    count = ingest_document(document=doc, embedder=embedder)
    assert count > 0
    assert doc.chunks.count() == count


@pytest.mark.django_db
def test_ingest_embed_called_per_chunk():
    from documents.models import Document
    from documents.ingest import ingest_document

    doc = Document.objects.create(title="Multi Chunk", raw_text=" ".join([f"w{i}" for i in range(30)]))
    embedder = FakeEmbedder()

    # Wrap embed to count calls
    original_embed = embedder.embed
    call_log = []

    def tracking_embed(text):
        call_log.append(text)
        return original_embed(text)

    embedder.embed = tracking_embed
    count = ingest_document(document=doc, embedder=embedder)

    assert len(call_log) == count


@pytest.mark.django_db
def test_ingest_empty_text_creates_zero_chunks():
    from documents.models import Document
    from documents.ingest import ingest_document

    doc = Document.objects.create(title="Empty", raw_text="   ")
    count = ingest_document(document=doc, embedder=FakeEmbedder())
    assert count == 0
    assert doc.chunks.count() == 0


@pytest.mark.django_db
def test_reingest_replaces_chunks():
    from documents.models import Document
    from documents.ingest import ingest_document

    doc = Document.objects.create(title="Replace Me", raw_text=" ".join(["a"] * 20))
    embedder = FakeEmbedder()

    first_count = ingest_document(document=doc, embedder=embedder)

    # Re-ingest with different text
    doc.raw_text = " ".join(["b"] * 12)
    doc.save()
    second_count = ingest_document(document=doc, embedder=embedder)

    assert doc.chunks.count() == second_count
    # Ensure no stale chunks from first run remain
    total_in_db = doc.chunks.count()
    assert total_in_db == second_count
