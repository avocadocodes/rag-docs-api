from rest_framework import serializers
from documents.models import Document


class DocumentCreateSerializer(serializers.ModelSerializer):
    file = serializers.FileField(write_only=True, required=False)

    class Meta:
        model = Document
        fields = ["id", "title", "raw_text", "file", "created_at"]
        read_only_fields = ["id", "created_at"]
        extra_kwargs = {
            "raw_text": {"required": False, "allow_blank": True},
        }

    def validate(self, attrs):
        raw_text = attrs.get("raw_text", "")
        file = attrs.pop("file", None)

        if file:
            try:
                raw_text = file.read().decode("utf-8")
            except UnicodeDecodeError:
                raise serializers.ValidationError(
                    {"file": "File must be UTF-8 encoded text (.txt or .md)."}
                )

        if not raw_text or not raw_text.strip():
            raise serializers.ValidationError(
                {"raw_text": "Provide either raw_text or a .txt / .md file."}
            )

        attrs["raw_text"] = raw_text
        return attrs


class DocumentListSerializer(serializers.ModelSerializer):
    chunk_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Document
        fields = ["id", "title", "chunk_count", "created_at"]
