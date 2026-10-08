import numpy as np
import pytest

from src.retrieval import Chunk, Hit

CORPUS = [
    Chunk("p1::0", "p1", "Paper One", 0, "We quantize the KV cache to 2 bits and prune flat keys."),
    Chunk("p1::1", "p1", "Paper One", 1, "Ablations show sharp queries matter for long-context accuracy."),
    Chunk("p2::0", "p2", "Paper Two", 0, "Speculative decoding drafts tokens that a target model verifies."),
    Chunk("p2::1", "p2", "Paper Two", 1, "The drafter is trained to minimize expected decoding rounds."),
    Chunk("p3::0", "p3", "Paper Three", 0, "Activation steering controls emotion in generated speech."),
]


@pytest.fixture
def corpus() -> list[Chunk]:
    return list(CORPUS)


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic, no model download."""

    model_name = "fake"
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            v[hash(tok.strip(".,")) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query):
        return self._vec(query)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


class StaticRetriever:
    """Returns the corpus in a fixed order; records the queries it saw."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def search(self, query, k=5):
        self.queries.append((query, k))
        return [Hit(chunk=c, score=1.0 / r, rank=r) for r, c in enumerate(self.chunks[:k], start=1)]


@pytest.fixture
def static_retriever(corpus) -> StaticRetriever:
    return StaticRetriever(corpus)
