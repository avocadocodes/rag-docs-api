from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from core.views import healthz, demo


urlpatterns = [
    # Health check
    path("healthz", healthz, name="healthz"),

    # Demo page
    path("", demo, name="demo"),

    # API
    path("api/v1/", include("documents.urls")),
    path("api/v1/", include("query.urls")),

    # OpenAPI / Swagger
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
]
