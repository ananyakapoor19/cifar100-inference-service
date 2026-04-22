# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime (CPU – for Cloud Run) ────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="cifar100-inference-service"
LABEL org.opencontainers.image.description="EfficientNet-B0 CIFAR-100 CPU inference (FP32 + INT8)"

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY api/       api/
COPY model/     model/

# Non-root user for Cloud Run best practice
RUN useradd -m appuser
USER appuser

# Cloud Run injects PORT env var; default to 8080
ENV PORT=8080 \
    HOST=0.0.0.0 \
    LOG_LEVEL=info \
    MODEL_FP32_PATH="" \
    MODEL_INT8_PATH=""

EXPOSE 8080

# Use a single uvicorn worker (Cloud Run scales via replicas, not threads)
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info"]
