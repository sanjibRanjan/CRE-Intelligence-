# ──────────────────────────────────────────────────────────────────────────
# AI Data Engineer — Production Dockerfile
# ──────────────────────────────────────────────────────────────────────────
# Multi-stage build for a lean production image.
#   Stage 1: Install Python deps into a virtual-env
#   Stage 2: Copy only the venv + app code (no build tools)
#
# Exposes port 8501 (Streamlit default) and is structured for
# Google Cloud Run deployment.
# ──────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system-level build deps (gcc needed for some wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the virtual-env from the builder
COPY --from=builder /opt/venv /opt/venv

# Make the venv the active Python
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Copy application source
COPY src/ ./src/
COPY app.py .
COPY data/ ./data/

# Pre-download the sentence-transformers model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
             "--server.port=8501", \
             "--server.address=0.0.0.0"]
