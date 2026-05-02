FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libpq-dev nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake ML models into image at build time (offline after this)
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
ENV HF_HOME=/app/model_cache

# Copy application code
COPY . .

# Build React components
WORKDIR /app/components/live_dag
RUN npm install && npm run build 2>/dev/null || true
WORKDIR /app/components/metric_counter
RUN npm install && npm run build 2>/dev/null || true

WORKDIR /app

# Create dirs
RUN mkdir -p /app/chroma /app/artifacts

EXPOSE 8501

HEALTHCHECK CMD curl -f http://localhost:8501/healthz || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
