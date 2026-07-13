from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.shortcuts import render


@require_GET
def healthz(request):
    return JsonResponse({"status": "ok"})


def demo(request):
    return render(request, "demo.html")
