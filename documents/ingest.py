"""
Ingestion service.

Responsibilities:
  1. Accept a Document instance whose raw_text is already saved.
  2. Split the text into overlapping chunks (via core.chunker).
  3. Embed each chunk (via an EmbedderProtocol implementation).
  4. Bulk-create DocumentChunk rows.
  5. Return the number of chunks created.

Dependency injection: the embedder is passed in rather than imported
directly so that tests can supply a FakeEmbedder without touching settings
or loading sentence-transformers.
"""

from __future__ import annotations

from core.chunker import chunk_text
from core.interfaces import EmbedderProtocol
from documents.models import Document, DocumentChunk


def ingest_document(document: Document, embedder: EmbedderProtocol) -> int:
    """Chunk, embed, and store all chunks for *document*.

    Existing chunks for the document are deleted first so that
    re-ingesting a document replaces its chunks cleanly.

    Returns the number of chunks created.
    """
    # Remove any stale chunks from a previous ingest
    document.chunks.all().delete()

    texts = chunk_text(document.raw_text)
    if not texts:
        return 0

    chunks = [
        DocumentChunk(
            document=document,
            chunk_index=i,
            text=text,
            embedding=embedder.embed(text),
        )
        for i, text in enumerate(texts)
    ]

    DocumentChunk.objects.bulk_create(chunks)
    return len(chunks)
