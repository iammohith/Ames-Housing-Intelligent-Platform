FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake ML model weights into image at build time so the container is 100% offline
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_OFFLINE=0

# Pre-download flan-t5-base and all-MiniLM-L6-v2 using the versions from requirements.txt
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
AutoTokenizer.from_pretrained('google/flan-t5-base', cache_dir='/app/model_cache'); \
AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base', cache_dir='/app/model_cache')" && \
    python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/model_cache')"

# Copy application code
COPY . .

# Create dirs
RUN mkdir -p /app/chroma /app/artifacts

EXPOSE 8501

HEALTHCHECK --interval=10s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/').read()" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
