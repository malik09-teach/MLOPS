# Stage 1: Build & dependency installation
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Lean runtime container
FROM python:3.11-slim AS runner

WORKDIR /app

# Install lightweight runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root application user for production security
RUN groupadd -r mlops && useradd -r -g mlops -d /app -s /sbin/nologin mlops

# Create runtime directory structure and assign ownership
RUN mkdir -p /app/data/raw /app/data/processed /app/logs /app/reports /app/mlruns && \
    chown -R mlops:mlops /app

# Copy application code into container
COPY --chown=mlops:mlops src/ /app/src/

USER mlops

EXPOSE 8000

# Container health monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch Uvicorn FastAPI production server
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
