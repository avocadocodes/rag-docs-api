from django.urls import path
from query.views import QueryView, QueryStreamView

urlpatterns = [
    path("query", QueryView.as_view(), name="query"),
    path("query/stream", QueryStreamView.as_view(), name="query-stream"),
]
