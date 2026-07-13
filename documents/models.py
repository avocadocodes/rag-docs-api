from django.db import models


class Document(models.Model):
    title = models.CharField(max_length=500)
    raw_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title

    @property
    def chunk_count(self):
        return self.chunks.count()


class DocumentChunk(models.Model):
    """
    Stores one text chunk and its embedding vector.

    The `embedding` column is defined as TextField in Django's ORM (so
    SQLite tests work without pgvector).  In PostgreSQL, the migration
    alters the column to `vector(384)` via raw SQL so pgvector operators
    work correctly at the DB level.
    """
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    chunk_index = models.PositiveIntegerField()
    text = models.TextField()
    # Stored as vector(384) in Postgres via migration; TextField in SQLite for tests.
    embedding = models.TextField()

    class Meta:
        unique_together = ("document", "chunk_index")
        ordering = ["document", "chunk_index"]

    def __str__(self):
        return f"Chunk {self.chunk_index} of '{self.document.title}'"
