"""Embed chunks and (re)build the Chroma collection for a chunk file.

    python -m src.index                              # data/chunks.jsonl
    python -m src.index --chunks data/chunks_250.jsonl
"""

import argparse
import logging
import time
from pathlib import Path

from src.retrieval import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_CHUNKS_PATH,
    Embedder,
    collection_name_for,
    get_collection,
    load_chunks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Chroma rejects very large single add() calls; stay well under its max batch size.
ADD_BATCH = 1000


def build_index(chunks_path: Path = DEFAULT_CHUNKS_PATH, chroma_dir: Path = DEFAULT_CHROMA_DIR) -> int:
    start = time.time()
    chunks = load_chunks(chunks_path)
    name = collection_name_for(chunks_path)

    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)  # full rebuild: chunk ids are positional, stale ones must go
    collection = get_collection(chroma_dir, name)

    embedder = Embedder()
    logger.info("Embedding %d chunks from %s with %s", len(chunks), chunks_path, embedder.model_name)
    vectors = embedder.embed_documents([c.text for c in chunks])

    for i in range(0, len(chunks), ADD_BATCH):
        batch = chunks[i : i + ADD_BATCH]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors[i : i + ADD_BATCH].tolist(),
            documents=[c.text for c in batch],
            metadatas=[{"arxiv_id": c.arxiv_id, "title": c.title, "chunk_index": c.chunk_index} for c in batch],
        )
    logger.info("Indexed %d chunks into collection %r in %.1fs", collection.count(), name, time.time() - start)
    return collection.count()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    args = parser.parse_args()
    build_index(args.chunks, args.chroma_dir)
