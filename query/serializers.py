from rest_framework import serializers


class QueryRequestSerializer(serializers.Serializer):
    question = serializers.CharField(min_length=1, max_length=2000)
    top_k = serializers.IntegerField(min_value=1, max_value=20, default=5)
    mode = serializers.ChoiceField(
        choices=["vector", "lexical", "hybrid"],
        default="hybrid",
    )
    rerank = serializers.BooleanField(default=True)


class CitationSerializer(serializers.Serializer):
    document_id = serializers.IntegerField()
    document_title = serializers.CharField()
    chunk_index = serializers.IntegerField()
    similarity = serializers.FloatField()


class RetrievedChunkSerializer(serializers.Serializer):
    text = serializers.CharField()
    chunk_index = serializers.IntegerField()
    document_id = serializers.IntegerField()
    document_title = serializers.CharField()
    similarity = serializers.FloatField()


class QueryResponseSerializer(serializers.Serializer):
    question = serializers.CharField()
    answer = serializers.CharField()
    mode = serializers.ChoiceField(choices=["extractive", "llm"])
    retrieval_mode = serializers.ChoiceField(choices=["vector", "lexical", "hybrid"])
    reranked = serializers.BooleanField()
    citations = CitationSerializer(many=True)
    retrieved_chunks = RetrievedChunkSerializer(many=True)
