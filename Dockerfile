# AI_Biomed Multi-Stage Container Definition
# Optimized for biomedical NLP inference and clinical ML pipelines

FROM python:3.11-slim as base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_NO_TF=1 \
    USE_TF=0

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install pytest

# Copy codebase and assets
COPY . .
RUN pip install -e . --no-deps

# Expose Streamlit dashboard port
EXPOSE 8501

# Healthcheck probe for Streamlit CDSS
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: launch interactive CDSS dashboard
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

