"""Retrieval layer: dense (Chroma + BGE embeddings), sparse (BM25), and hybrid (RRF).

Every retriever exposes the same `search(query, k) -> list[Hit]` interface so the
evaluation harness and the API can swap them freely.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
DEFAULT_CHROMA_DIR = ROOT / "chroma_db"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE v1.5 is trained with this instruction on the query side only (passages are
# embedded raw). fastembed's query_embed() does not add it, so we do.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Reciprocal Rank Fusion constant from Cormack et al. (2009); 60 is the standard value.
RRF_K = 60


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    arxiv_id: str
    title: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1-based


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


def load_chunks(path: Path = DEFAULT_CHUNKS_PATH) -> list[Chunk]:
    chunks = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=d["chunk_id"],
                    arxiv_id=d["arxiv_id"],
                    title=d["title"],
                    chunk_index=d["chunk_index"],
                    text=d["text"],
                )
            )
    if not chunks:
        raise ValueError(f"No chunks found in {path}; run `python src/ingest.py` first")
    return chunks


def collection_name_for(chunks_path: Path) -> str:
    """One Chroma collection per chunk file, so chunk-size variants never collide."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", Path(chunks_path).stem)


# --------------------------------------------------------------------------- dense


class Embedder:
    """Thin wrapper over fastembed (ONNX runtime, no torch) with BGE query handling."""

    def __init__(self, model_name: str = EMBED_MODEL, query_instruction: str = BGE_QUERY_INSTRUCTION):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.query_instruction = query_instruction
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.array(list(self._model.embed(texts, batch_size=batch_size)), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return np.array(next(iter(self._model.embed([self.query_instruction + query]))), dtype=np.float32)


def get_collection(chroma_dir: Path, name: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    # embedding_function=None: we always pass precomputed vectors, and this stops
    # Chroma from silently downloading its own default MiniLM model.
    return client.get_or_create_collection(
        name=name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None
    )


class DenseRetriever:
    def __init__(self, collection, embedder: Embedder, chunks_by_id: dict[str, Chunk]):
        self.collection = collection
        self.embedder = embedder
        self.chunks_by_id = chunks_by_id

    def search(self, query: str, k: int = 5) -> list[Hit]:
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query).tolist()],
            n_results=k,
            include=["distances"],
        )
        hits = []
        for rank, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), start=1):
            # Chroma's cosine "distance" is 1 - cosine similarity.
            hits.append(Hit(chunk=self.chunks_by_id[cid], score=1.0 - float(dist), rank=rank))
        return hits


# -------------------------------------------------------------------------- sparse

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
# Small hand-picked list; BM25's IDF already discounts common words, this just keeps
# question boilerplate ("what", "does", "how") from contributing noise.
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in is it its of on or "
    "that the their this to was what when which who why with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(chunk=self.chunks[i], score=float(scores[i]), rank=r) for r, i in enumerate(top, start=1)]


# -------------------------------------------------------------------------- hybrid


def reciprocal_rank_fusion(result_lists: list[list[Hit]], k: int, rrf_k: int = RRF_K) -> list[Hit]:
    """Fuse ranked lists by sum of 1/(rrf_k + rank). Rank-based, so BM25 scores and
    cosine similarities never need to be put on a common scale."""
    fused: dict[str, float] = {}
    chunk_of: dict[str, Chunk] = {}
    for hits in result_lists:
        for h in hits:
            fused[h.chunk.chunk_id] = fused.get(h.chunk.chunk_id, 0.0) + 1.0 / (rrf_k + h.rank)
            chunk_of[h.chunk.chunk_id] = h.chunk
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [Hit(chunk=chunk_of[cid], score=s, rank=r) for r, (cid, s) in enumerate(ordered, start=1)]


class HybridRetriever:
    def __init__(self, dense: Retriever, sparse: Retriever, candidates: int = 50):
        self.dense = dense
        self.sparse = sparse
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[Hit]:
        n = max(k, self.candidates)
        return reciprocal_rank_fusion([self.dense.search(query, n), self.sparse.search(query, n)], k=k)


# ------------------------------------------------------------------------- factory

MODES = ("dense", "bm25", "hybrid")


def build_retriever(
    mode: str = "hybrid",
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    embedder: Embedder | None = None,
) -> Retriever:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    chunks = load_chunks(chunks_path)
    if mode == "bm25":
        return BM25Retriever(chunks)

    collection = get_collection(chroma_dir, collection_name_for(chunks_path))
    if collection.count() != len(chunks):
        raise RuntimeError(
            f"Chroma collection {collection.name!r} has {collection.count()} vectors but "
            f"{chunks_path} has {len(chunks)} chunks; run `python -m src.index --chunks {chunks_path}`"
        )
    dense = DenseRetriever(collection, embedder or Embedder(), {c.chunk_id: c for c in chunks})
    if mode == "dense":
        return dense
    return HybridRetriever(dense, BM25Retriever(chunks))
