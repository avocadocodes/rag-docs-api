from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiResponse

from documents.models import Document
from documents.serializers import DocumentCreateSerializer, DocumentListSerializer
from documents.ingest import ingest_document
from core.embedder import get_embedder


class DocumentListCreateView(APIView):

    @extend_schema(
        responses=DocumentListSerializer(many=True),
        summary="List all documents",
    )
    def get(self, request):
        docs = Document.objects.prefetch_related("chunks").all()
        serializer = DocumentListSerializer(docs, many=True)
        return Response(serializer.data)

    @extend_schema(
        request=DocumentCreateSerializer,
        responses={
            201: OpenApiResponse(description="Document created, returns id + chunk_count"),
        },
        summary="Upload and ingest a document",
        description=(
            "Accepts either raw_text (string) or a file upload (.txt / .md). "
            "The document is split into overlapping chunks, each chunk is embedded "
            "using sentence-transformers (all-MiniLM-L6-v2, 384 dim) and stored "
            "in pgvector for similarity search."
        ),
    )
    def post(self, request):
        serializer = DocumentCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        document = serializer.save()
        chunk_count = ingest_document(document, get_embedder())

        return Response(
            {
                "id": document.id,
                "title": document.title,
                "chunk_count": chunk_count,
                "created_at": document.created_at,
            },
            status=status.HTTP_201_CREATED,
        )


class DocumentDetailView(APIView):

    @extend_schema(
        responses=DocumentListSerializer,
        summary="Retrieve a single document",
    )
    def get(self, request, pk):
        try:
            doc = Document.objects.prefetch_related("chunks").get(pk=pk)
        except Document.DoesNotExist:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        serializer = DocumentListSerializer(doc)
        return Response(serializer.data)
