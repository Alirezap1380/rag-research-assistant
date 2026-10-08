FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the container never downloads at runtime.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY src/ src/
COPY api/ api/
# Corpus artifacts are generated locally (python src/ingest.py ... && python -m src.index ...)
# and baked in; they are not committed to git. The served config is BM25 over 250-token chunks;
# chroma_db and the embedding model are included so RAG_RETRIEVAL_MODE=dense/hybrid also work.
COPY data/papers/metadata.json data/papers/metadata.json
COPY data/chunks_250.jsonl data/chunks_250.jsonl
COPY chroma_db/ chroma_db/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

# ANTHROPIC_API_KEY must be passed at runtime (docker run -e ANTHROPIC_API_KEY=...) for /ask;
# /search works without it.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
