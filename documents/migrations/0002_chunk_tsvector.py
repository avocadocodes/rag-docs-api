"""
Migration 0002 - full-text search column.

Adds a `search_vector` tsvector column to DocumentChunk and a GIN index
so that Postgres full-text search (`websearch_to_tsquery`) works efficiently.

The column is populated at INSERT/UPDATE time by the application; we don't
use a generated column or a trigger so that the Django ORM stays in control.

SQLite note: the _ConditionalSQL helper (same pattern as migration 0001) makes
every Postgres-specific statement a no-op when running under SQLite, so the
test suite migrates cleanly without modification.
"""

from django.db import migrations, models


class _ConditionalSQL(migrations.RunSQL):
    """Wraps RunSQL so it only executes when the database is PostgreSQL."""

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        if schema_editor.connection.vendor == "postgresql":
            super().database_forwards(app_label, schema_editor, from_state, to_state)

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        if schema_editor.connection.vendor == "postgresql":
            super().database_backwards(app_label, schema_editor, from_state, to_state)


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0001_initial"),
    ]

    operations = [
        # Add the search_vector column in Django ORM as TextField so SQLite is happy.
        # On Postgres the _ConditionalSQL below ALTERs the type to tsvector.
        migrations.AddField(
            model_name="documentchunk",
            name="search_vector",
            field=models.TextField(default="", blank=True),
        ),

        # On Postgres: change type to tsvector
        _ConditionalSQL(
            sql=(
                "ALTER TABLE documents_documentchunk "
                "ALTER COLUMN search_vector TYPE tsvector "
                "USING to_tsvector('english', '');"
            ),
            reverse_sql=(
                "ALTER TABLE documents_documentchunk "
                "ALTER COLUMN search_vector TYPE text USING '';"
            ),
        ),

        # GIN index for fast full-text search on Postgres
        _ConditionalSQL(
            sql=(
                "CREATE INDEX IF NOT EXISTS documentchunk_search_vector_gin_idx "
                "ON documents_documentchunk USING gin(search_vector);"
            ),
            reverse_sql="DROP INDEX IF EXISTS documentchunk_search_vector_gin_idx;",
        ),
    ]
