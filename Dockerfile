# ===========================================================================
# Conversational AI Agent - CPU variant
# ===========================================================================
# For GPU support, use Dockerfile.gpu or pass --build-arg BASE_IMAGE=...

FROM python:3.13-slim AS base

WORKDIR /app

# Use an in-container virtualenv so installs mirror local uv workflows.
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY requirements.txt requirements.lock ./

# Create venv and install pinned Python dependencies into it
RUN uv venv ${VIRTUAL_ENV} && \
    uv pip install --python ${VIRTUAL_ENV}/bin/python -r requirements.lock

# Copy application code
COPY . .

# Create directories for runtime data
RUN mkdir -p .chat_sessions chroma_db .prompt_media

# Expose ports for Streamlit and FastAPI
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
