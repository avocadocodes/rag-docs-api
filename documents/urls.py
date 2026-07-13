from django.urls import path
from documents.views import DocumentListCreateView, DocumentDetailView

urlpatterns = [
    path("documents", DocumentListCreateView.as_view(), name="document-list-create"),
    path("documents/<int:pk>", DocumentDetailView.as_view(), name="document-detail"),
]
