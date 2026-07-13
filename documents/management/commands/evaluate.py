"""
Management command: python manage.py evaluate

Runs the retrieval evaluation harness.

For each retrieval configuration — vector-only, lexical-only, hybrid,
hybrid+rerank — it:
  1. Ingests the evaluation corpus (10 documents about a fictional product).
  2. Runs each of the 20 evaluation questions through the retriever.
  3. Computes Recall@1, Recall@3, Recall@5, and MRR.
  4. Prints a comparison table to stdout.

How to run (Docker Compose):
    docker compose exec web python manage.py evaluate

The command cleans up the documents it created after the run (using a
dedicated title prefix) so it does not pollute production data.

This command requires a live PostgreSQL + pgvector database and the real
sentence-transformers embedder.  It will not produce meaningful results
against the SQLite/FakeEmbedder test environment.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction

from documents.models import Document
from documents.ingest import ingest_document
from eval.dataset import DOCUMENTS, QUESTIONS, EvalQuestion
from eval.metrics import recall_at_k, mrr
from core.embedder import get_embedder
from query.retrieval import (
    PgvectorRetriever,
    LexicalRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)

_EVAL_TITLE_PREFIX = "[eval] "
_CANDIDATE_K = 20   # pool size; metrics are computed at k=1,3,5 from this list


class Command(BaseCommand):
    help = "Evaluate retrieval quality across configurations (vector/lexical/hybrid/hybrid+rerank)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--keep",
            action="store_true",
            help="Do not delete ingested eval documents after the run.",
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("\nVela Eval Corpus — Retrieval Evaluation\n"))

        embedder = get_embedder()

        # ------------------------------------------------------------------
        # 1. Ingest eval corpus
        # ------------------------------------------------------------------
        self.stdout.write("Ingesting evaluation corpus…")
        doc_id_map: dict[int, int] = {}   # eval_doc.id → DB primary key

        try:
            for eval_doc in DOCUMENTS:
                title = f"{_EVAL_TITLE_PREFIX}{eval_doc.title}"
                # Reuse existing if already present (idempotent)
                obj, _ = Document.objects.get_or_create(
                    title=title,
                    defaults={"raw_text": eval_doc.text},
                )
                if _.created if hasattr(_, 'created') else False:
                    pass
                # Always re-ingest to ensure fresh embeddings
                obj.raw_text = eval_doc.text
                obj.save(update_fields=["raw_text"])
                ingest_document(obj, embedder)
                doc_id_map[eval_doc.id] = obj.pk

            self.stdout.write(f"  {len(DOCUMENTS)} documents ingested.\n")

            # ------------------------------------------------------------------
            # 2. Run evaluation per configuration
            # ------------------------------------------------------------------
            configs = [
                ("vector-only",     self._run_vector),
                ("lexical-only",    self._run_lexical),
                ("hybrid",          self._run_hybrid),
                ("hybrid + rerank", self._run_hybrid_rerank),
            ]

            rows = []
            for name, runner in configs:
                self.stdout.write(f"  Evaluating [{name}]…")
                result_pairs = runner(embedder, doc_id_map)
                r1  = recall_at_k(result_pairs, k=1)
                r3  = recall_at_k(result_pairs, k=3)
                r5  = recall_at_k(result_pairs, k=5)
                mrr_score = mrr(result_pairs)
                rows.append((name, r1, r3, r5, mrr_score))

            # ------------------------------------------------------------------
            # 3. Print results table
            # ------------------------------------------------------------------
            self._print_table(rows)

        finally:
            if not options["keep"]:
                deleted, _ = Document.objects.filter(
                    title__startswith=_EVAL_TITLE_PREFIX
                ).delete()
                self.stdout.write(f"\nCleaned up {deleted} eval document(s).")

    # ------------------------------------------------------------------
    # Per-configuration runners
    # ------------------------------------------------------------------

    def _run_vector(self, embedder, doc_id_map):
        retriever = PgvectorRetriever()
        pairs = []
        for q in QUESTIONS:
            emb = embedder.embed(q.question)
            chunks = retriever.retrieve(emb, _CANDIDATE_K)
            retrieved_doc_ids = [c.document_id for c in chunks]
            relevant_db_ids = [doc_id_map[d] for d in q.relevant_doc_ids]
            pairs.append((retrieved_doc_ids, relevant_db_ids))
        return pairs

    def _run_lexical(self, embedder, doc_id_map):
        retriever = LexicalRetriever()
        pairs = []
        for q in QUESTIONS:
            chunks = retriever.retrieve(q.question, _CANDIDATE_K)
            retrieved_doc_ids = [c.document_id for c in chunks]
            relevant_db_ids = [doc_id_map[d] for d in q.relevant_doc_ids]
            pairs.append((retrieved_doc_ids, relevant_db_ids))
        return pairs

    def _run_hybrid(self, embedder, doc_id_map):
        retriever = HybridRetriever(candidate_k=_CANDIDATE_K)
        pairs = []
        for q in QUESTIONS:
            emb = embedder.embed(q.question)
            chunks = retriever.retrieve(emb, q.question, _CANDIDATE_K)
            retrieved_doc_ids = [c.document_id for c in chunks]
            relevant_db_ids = [doc_id_map[d] for d in q.relevant_doc_ids]
            pairs.append((retrieved_doc_ids, relevant_db_ids))
        return pairs

    def _run_hybrid_rerank(self, embedder, doc_id_map):
        from core.reranker import get_reranker  # noqa: PLC0415
        retriever = HybridRetriever(candidate_k=_CANDIDATE_K)
        reranker = get_reranker()
        pairs = []
        for q in QUESTIONS:
            emb = embedder.embed(q.question)
            chunks = retriever.retrieve(emb, q.question, _CANDIDATE_K)
            chunks = reranker.rerank(q.question, chunks)
            retrieved_doc_ids = [c.document_id for c in chunks]
            relevant_db_ids = [doc_id_map[d] for d in q.relevant_doc_ids]
            pairs.append((retrieved_doc_ids, relevant_db_ids))
        return pairs

    # ------------------------------------------------------------------
    # Table formatting
    # ------------------------------------------------------------------

    def _print_table(self, rows):
        self.stdout.write("\n")
        self.stdout.write(self.style.SUCCESS("=" * 65))
        header = f"{'Configuration':<22} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>7}"
        self.stdout.write(self.style.SUCCESS(header))
        self.stdout.write(self.style.SUCCESS("-" * 65))
        for name, r1, r3, r5, mrr_score in rows:
            line = f"{name:<22} {r1:>6.3f} {r3:>6.3f} {r5:>6.3f} {mrr_score:>7.3f}"
            self.stdout.write(line)
        self.stdout.write(self.style.SUCCESS("=" * 65))
        self.stdout.write(
            "\nR@k = Recall@k  |  MRR = Mean Reciprocal Rank  "
            f"|  n={len(QUESTIONS)} questions\n"
        )
