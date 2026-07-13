from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

from query.serializers import QueryRequestSerializer, QueryResponseSerializer
from query.retrieval import PgvectorRetriever
from query.answer import generate_answer
from core.embedder import get_embedder


class QueryView(APIView):

    @extend_schema(
        request=QueryRequestSerializer,
        responses=QueryResponseSerializer,
        summary="Ask a question against ingested documents",
        description=(
            "Embeds the question, runs a cosine similarity search over all stored "
            "document chunks, and returns a grounded answer with citations.\n\n"
            "Answer mode depends on environment configuration:\n"
            "- **extractive** (default): top chunks are returned directly as the answer. "
            "No external API needed.\n"
            "- **llm**: if LLM_API_BASE + LLM_API_KEY + LLM_MODEL are set, an "
            "OpenAI-compatible chat endpoint is called with the retrieved context."
        ),
    )
    def post(self, request):
        serializer = QueryRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        question: str = serializer.validated_data["question"]
        top_k: int = serializer.validated_data["top_k"]

        embedder = get_embedder()
        query_embedding = embedder.embed(question)

        retriever = PgvectorRetriever()
        chunks = retriever.retrieve(query_embedding, top_k)

        result = generate_answer(question, chunks)

        return Response(
            {
                "question": question,
                "answer": result.answer,
                "mode": result.mode,
                "citations": [
                    {
                        "document_id": c.document_id,
                        "document_title": c.document_title,
                        "chunk_index": c.chunk_index,
                        "similarity": c.similarity,
                    }
                    for c in result.citations
                ],
                "retrieved_chunks": [
                    {
                        "text": c.text,
                        "chunk_index": c.chunk_index,
                        "document_id": c.document_id,
                        "document_title": c.document_title,
                        "similarity": c.similarity,
                    }
                    for c in chunks
                ],
            },
            status=status.HTTP_200_OK,
        )
