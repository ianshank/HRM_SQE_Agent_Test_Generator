# HRM SQE Agent Test Generator - Production Dockerfile
# Multi-stage build for optimized image size and security

# =============================================================================
# Stage 1: Builder - Install dependencies and build wheels
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

# Build arguments
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy dependency files
WORKDIR /build
COPY pyproject.toml setup.py requirements.txt* ./
COPY hrm_eval/__init__.py hrm_eval/

# Install dependencies (without dev dependencies for production)
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir gunicorn

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Labels for container metadata
LABEL maintainer="Ian Cruickshank <ianshank@gmail.com>"
LABEL org.opencontainers.image.title="HRM SQE Agent Test Generator"
LABEL org.opencontainers.image.description="AI-powered test case generation from requirements"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/ianshank/HRM_SQE_Agent_Test_Generator"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Application settings
    HRM_ENV=production \
    HRM_API_HOST=0.0.0.0 \
    HRM_API_PORT=8000 \
    HRM_LOG_LEVEL=INFO

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup hrm_eval/ ./hrm_eval/
COPY --chown=appuser:appgroup pyproject.toml setup.py ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/vector_store_db /app/drop_folder/input /app/drop_folder/output && \
    chown -R appuser:appgroup /app

# Copy checkpoint if exists (or mount as volume)
# COPY --chown=appuser:appgroup checkpoints_hrm_v9_optimized_step_7566 ./checkpoints_hrm_v9_optimized_step_7566/

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${HRM_API_PORT}/health || exit 1

# Default command - run with gunicorn for production
CMD ["gunicorn", "hrm_eval.api_service.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "300", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output"]

# =============================================================================
# Stage 3: Development image with additional tools
# =============================================================================
FROM runtime AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install dev dependencies
RUN pip install --no-cache-dir pytest pytest-cov pytest-asyncio black ruff mypy

USER appuser

# Override command for development
CMD ["uvicorn", "hrm_eval.api_service.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload"]

# =============================================================================
# Stage 4: GPU-enabled image
# =============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS gpu

# Labels
LABEL maintainer="Ian Cruickshank <ianshank@gmail.com>"
LABEL org.opencontainers.image.title="HRM SQE Agent Test Generator (GPU)"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    DEBIAN_FRONTEND=noninteractive \
    HRM_ENV=production \
    HRM_API_HOST=0.0.0.0 \
    HRM_API_PORT=8000 \
    HRM_DEVICE=cuda

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from builder (we need to rebuild for GPU)
COPY --from=builder /opt/venv /opt/venv

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup hrm_eval/ ./hrm_eval/
COPY --chown=appuser:appgroup pyproject.toml setup.py ./

# Create directories
RUN mkdir -p /app/logs /app/data /app/vector_store_db && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${HRM_API_PORT}/health || exit 1

CMD ["gunicorn", "hrm_eval.api_service.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "300"]
