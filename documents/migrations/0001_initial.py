"""
Initial migration.

Step 1: enable the pgvector extension (PostgreSQL only, skipped on SQLite).
Step 2: create Document and DocumentChunk tables.
Step 3: create an IVFFlat index on embedding for cosine similarity (Postgres only).

The IVFFlat index (lists=100) is appropriate up to ~1M chunks.
For larger datasets, switch to HNSW:
    CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);
"""

from django.db import migrations, models
import django.db.models.deletion


# ---------------------------------------------------------------------------
# Conditional SQL operations - skip on SQLite (used by the test suite)
# ---------------------------------------------------------------------------

class _ConditionalSQL(migrations.RunSQL):
    """Wraps RunSQL so it only executes when the database is PostgreSQL."""

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        if schema_editor.connection.vendor == "postgresql":
            super().database_forwards(app_label, schema_editor, from_state, to_state)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        if schema_editor.connection.vendor == "postgresql":
            super().database_backwards(app_label, schema_editor, from_state, to_state)


class _ConditionalVectorField(models.Field):
    """
    A stub field used when running under SQLite so migrations don't import
    pgvector.  At runtime on PostgreSQL the real VectorField is used instead;
    see _EmbeddingField below.
    """
    def db_type(self, connection):
        # SQLite will store the embedding as text; we never query it in tests
        return "text"

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        # Ensure re-usable across migration reconstructions
        path = "django.db.models.TextField"
        return name, path, args, kwargs


def _embedding_field():
    """Return the appropriate field for the current DB backend."""
    try:
        from pgvector.django import VectorField
        from django.conf import settings
        return VectorField(dimensions=settings.EMBEDDING_DIM)
    except Exception:
        return _ConditionalVectorField()


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        _ConditionalSQL(
            sql="CREATE EXTENSION IF NOT EXISTS vector;",
            reverse_sql="DROP EXTENSION IF EXISTS vector;",
        ),

        migrations.CreateModel(
            name="Document",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=500)),
                ("raw_text", models.TextField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),

        migrations.CreateModel(
            name="DocumentChunk",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "document",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="chunks",
                        to="documents.document",
                    ),
                ),
                ("chunk_index", models.PositiveIntegerField()),
                ("text", models.TextField()),
                ("embedding", models.TextField()),   # overridden to VectorField at DB level below
            ],
            options={
                "ordering": ["document", "chunk_index"],
            },
        ),

        migrations.AddConstraint(
            model_name="documentchunk",
            constraint=models.UniqueConstraint(
                fields=["document", "chunk_index"],
                name="unique_document_chunk_index",
            ),
        ),

        # Alter the embedding column to vector(384) on PostgreSQL
        _ConditionalSQL(
            sql="ALTER TABLE documents_documentchunk ALTER COLUMN embedding TYPE vector(384) USING embedding::vector;",
            reverse_sql="ALTER TABLE documents_documentchunk ALTER COLUMN embedding TYPE text;",
        ),

        # HNSW index for cosine similarity. HNSW gives near-exact recall with
        # default search parameters and, unlike IVFFlat, does not degrade on
        # small collections (IVFFlat with lists >> rows and probes=1 misses most
        # neighbours). Suitable from a handful of rows up to millions.
        _ConditionalSQL(
            sql="""
                CREATE INDEX IF NOT EXISTS documentchunk_embedding_cosine_idx
                ON documents_documentchunk
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """,
            reverse_sql="DROP INDEX IF EXISTS documentchunk_embedding_cosine_idx;",
        ),
    ]
