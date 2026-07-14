from __future__ import annotations
from django.core.management.base import BaseCommand
from documents.models import Document
from documents.ingest import ingest_document
from eval.dataset import DOCUMENTS, QUESTIONS, OUT_OF_CORPUS_QUESTIONS, EvalQuestion
from eval.metrics import recall_at_k, mrr, faithfulness_score, abstention_accuracy
from core.embedder import get_embedder
from query.retrieval import (
    PgvectorRetriever,
    LexicalRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)

_EVAL_TITLE_PREFIX = "[eval] "
_CANDIDATE_K = 20


class Command(BaseCommand):
    help = (
        "Evaluate retrieval quality across configurations (vector/lexical/hybrid/hybrid+rerank). "
        "Add --faithfulness to also run the answer faithfulness and abstention evaluation."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--keep",
            action="store_true",
            help="Do not delete ingested eval documents after the run.",
        )
        parser.add_argument(
            "--faithfulness",
            action="store_true",
            help="Also run faithfulness and abstention evaluation (requires LLM + NLI model).",
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("\nVela Eval Corpus — Retrieval Evaluation\n"))
        embedder = get_embedder()

        self.stdout.write("Ingesting evaluation corpus…")
        doc_id_map: dict[int, int] = {}

        try:
            for eval_doc in DOCUMENTS:
                title = f"{_EVAL_TITLE_PREFIX}{eval_doc.title}"
                obj, _ = Document.objects.get_or_create(
                    title=title,
                    defaults={"raw_text": eval_doc.text},
                )
                obj.raw_text = eval_doc.text
                obj.save(update_fields=["raw_text"])
                ingest_document(obj, embedder)
                doc_id_map[eval_doc.id] = obj.pk

            self.stdout.write(f"  {len(DOCUMENTS)} documents ingested.\n")

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

            self._print_retrieval_table(rows)

            if options["faithfulness"]:
                self._run_faithfulness_eval(embedder)

        finally:
            if not options["keep"]:
                deleted, _ = Document.objects.filter(
                    title__startswith=_EVAL_TITLE_PREFIX
                ).delete()
                self.stdout.write(f"\nCleaned up {deleted} eval document(s).")

    def _run_faithfulness_eval(self, embedder):
        from core.reranker import get_reranker
        from core.verifier import get_verifier
        from query.answer import generate_answer
        from query.verification import AnswerVerification

        self.stdout.write(self.style.MIGRATE_HEADING("\nFaithfulness & Abstention Evaluation\n"))

        retriever = HybridRetriever(candidate_k=_CANDIDATE_K)
        reranker = get_reranker()
        verifier = get_verifier()
        svc = AnswerVerification(verifier=verifier)

        self.stdout.write(f"  Running {len(QUESTIONS)} answerable questions…")
        all_verdicts: list[str] = []
        for q in QUESTIONS:
            emb = embedder.embed(q.question)
            chunks = retriever.retrieve(emb, q.question, _CANDIDATE_K)
            chunks = reranker.rerank(q.question, chunks)[:5]
            ans = generate_answer(q.question, chunks)
            verification = svc.verify(answer=ans.answer, chunks=chunks)
            all_verdicts.extend(v["label"] for v in verification.claims)

        avg_faithfulness = faithfulness_score(all_verdicts)

        self.stdout.write(f"  Running {len(OUT_OF_CORPUS_QUESTIONS)} out-of-corpus questions…")
        abstention_results: list[tuple[bool, bool]] = []
        for q in OUT_OF_CORPUS_QUESTIONS:
            emb = embedder.embed(q.question)
            chunks = retriever.retrieve(emb, q.question, _CANDIDATE_K)
            chunks = reranker.rerank(q.question, chunks)[:5]
            ans = generate_answer(q.question, chunks)
            verification = svc.verify(answer=ans.answer, chunks=chunks)
            abstention_results.append((verification.abstained, True))

        abstention_acc = abstention_accuracy(abstention_results)
        abstained_count = sum(1 for ab, _ in abstention_results if ab)

        self._print_faithfulness_table(avg_faithfulness, abstention_acc, abstained_count)

    def _print_faithfulness_table(self, avg_faithfulness, abstention_acc, abstained_count):
        self.stdout.write("\n")
        self.stdout.write(self.style.SUCCESS("=" * 55))
        self.stdout.write(self.style.SUCCESS("Faithfulness & Abstention Results"))
        self.stdout.write(self.style.SUCCESS("-" * 55))
        self.stdout.write(f"{'Metric':<40} {'Value':>10}")
        self.stdout.write(self.style.SUCCESS("-" * 55))
        self.stdout.write(f"{'Avg faithfulness (answerable Qs)':<40} {avg_faithfulness:>10.3f}")
        self.stdout.write(f"{'Abstention accuracy (unanswerable Qs)':<40} {abstention_acc:>10.3f}")
        self.stdout.write(
            f"{'Abstained on':<40} "
            f"{abstained_count}/{len(OUT_OF_CORPUS_QUESTIONS)} out-of-corpus Qs"
        )
        self.stdout.write(self.style.SUCCESS("=" * 55))
        self.stdout.write(
            "\nFaithfulness: fraction of answer claims supported by NLI.\n"
            "Abstention accuracy: fraction of unanswerable Qs where system correctly abstained.\n"
        )

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

    def _print_retrieval_table(self, rows):
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
