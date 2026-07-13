# syntax=docker/dockerfile:1

# Base image: python:3.11-slim
#
# Image size note: sentence-transformers pulls PyTorch as a dependency.
# The resulting image is typically 2.5–3 GB (CPU-only torch wheel).
# To reduce size in production, pin torch to the CPU-only wheel before build:
#   pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
# That brings the image down to ~1.5 GB.
#
# The all-MiniLM-L6-v2 model (~90 MB) is downloaded at first startup and
# cached under ~/.cache/huggingface inside the container.  To pre-bake it,
# add a RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# layer after pip install.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies (pg_isready is in postgresql-client)
RUN apt-get update && apt-get install -y --no-install-recommends \
      postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source
COPY . .

# Pre-download the embedding model so the first request isn't slow
# Comment this out to keep the image smaller and let it download at startup.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
