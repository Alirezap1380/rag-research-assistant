import json

import pytest

from src.retrieval import (
    BM25Retriever,
    DenseRetriever,
    Hit,
    HybridRetriever,
    collection_name_for,
    get_collection,
    load_chunks,
    reciprocal_rank_fusion,
    tokenize,
)
from tests.conftest import StaticRetriever


def test_tokenize_drops_stopwords_keeps_technical_tokens():
    assert tokenize("What is the KV-cache size of GPT-4.5?") == ["kv-cache", "size", "gpt-4.5"]


def test_bm25_ranks_lexical_match_first(corpus):
    hits = BM25Retriever(corpus).search("expected decoding rounds", k=2)
    assert hits[0].chunk.chunk_id == "p2::1"
    assert [h.rank for h in hits] == [1, 2]


def test_rrf_rewards_agreement(corpus):
    a = [Hit(corpus[0], 9, 1), Hit(corpus[1], 8, 2), Hit(corpus[2], 7, 3)]
    b = [Hit(corpus[1], 0.9, 1), Hit(corpus[3], 0.8, 2), Hit(corpus[0], 0.7, 3)]
    fused = reciprocal_rank_fusion([a, b], k=3)
    # p1::1 is rank 2 + rank 1, p1::0 is rank 1 + rank 3: p1::1 wins. Single-list
    # hits follow by rank: p2::1 (rank 2 in b) beats p2::0 (rank 3 in a).
    assert [h.chunk.chunk_id for h in fused] == ["p1::1", "p1::0", "p2::1"]
    assert [h.rank for h in fused] == [1, 2, 3]


def test_rrf_is_scale_free(corpus):
    a = [Hit(corpus[0], 1e6, 1), Hit(corpus[1], 1e5, 2)]
    b = [Hit(corpus[1], 0.02, 1), Hit(corpus[0], 0.01, 2)]
    fused = reciprocal_rank_fusion([a, b], k=2)
    assert fused[0].score == pytest.approx(fused[1].score)  # raw score magnitude is ignored


def test_hybrid_asks_both_retrievers_for_candidates(corpus):
    dense, sparse = StaticRetriever(corpus), StaticRetriever(list(reversed(corpus)))
    hits = HybridRetriever(dense, sparse, candidates=4).search("q", k=2)
    assert dense.queries == [("q", 4)] and sparse.queries == [("q", 4)]
    assert len(hits) == 2


def test_dense_retriever_with_real_chroma(tmp_path, corpus, fake_embedder):
    col = get_collection(tmp_path, "test")
    col.add(
        ids=[c.chunk_id for c in corpus],
        embeddings=fake_embedder.embed_documents([c.text for c in corpus]).tolist(),
    )
    dense = DenseRetriever(col, fake_embedder, {c.chunk_id: c for c in corpus})
    hits = dense.search("activation steering controls emotion in generated speech", k=3)
    assert hits[0].chunk.chunk_id == "p3::0"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)  # cosine similarity, not distance
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_load_chunks_roundtrip_and_empty(tmp_path, corpus):
    p = tmp_path / "chunks.jsonl"
    p.write_text("".join(json.dumps(c.__dict__) + "\n" for c in corpus))
    assert load_chunks(p) == corpus
    (tmp_path / "empty.jsonl").write_text("")
    with pytest.raises(ValueError, match="No chunks"):
        load_chunks(tmp_path / "empty.jsonl")


def test_collection_name_per_chunk_file():
    assert collection_name_for("data/chunks.jsonl") == "chunks"
    assert collection_name_for("data/chunks_250.jsonl") == "chunks_250"
