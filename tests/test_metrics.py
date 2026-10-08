import math

import pytest

from src.evaluate import bootstrap_ci, ndcg_at_k, recall_at_k, reciprocal_rank, score_question

E = set()  # irrelevant chunk


def test_recall_single_unit_hit_and_miss():
    ranked = [E, E, {0}, E]
    assert recall_at_k(ranked, 1, 2) == 0.0
    assert recall_at_k(ranked, 1, 3) == 1.0


def test_recall_multi_unit_partial_and_duplicates():
    ranked = [{0}, {0}, E, {1}]  # unit 0 twice must not count twice
    assert recall_at_k(ranked, 2, 3) == 0.5
    assert recall_at_k(ranked, 2, 4) == 1.0


def test_recall_empty_ranking():
    assert recall_at_k([], 1, 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([{0}]) == 1.0
    assert reciprocal_rank([E, E, {0}]) == pytest.approx(1 / 3)
    assert reciprocal_rank([E] * 10 + [{0}]) == 0.0  # beyond the @10 cutoff


def test_ndcg_perfect_and_first_rank_discount():
    assert ndcg_at_k([{0}, E, E], 1, 3) == 1.0
    assert ndcg_at_k([E, {0}, E], 1, 3) == pytest.approx(1 / math.log2(3))


def test_ndcg_redundant_chunk_earns_nothing():
    # Two chunks carrying the same fact score the same as one.
    assert ndcg_at_k([{0}, {0}], 2, 5) == ndcg_at_k([{0}, E], 2, 5)


def test_ndcg_two_units_ideal_and_capped():
    assert ndcg_at_k([{0}, {1}], 2, 5) == pytest.approx(1.0)
    assert ndcg_at_k([{0, 1}], 2, 5) == 1.0  # one chunk covering both is capped, not > 1


def test_ndcg_no_relevant():
    assert ndcg_at_k([E, E], 1, 2) == 0.0


def test_score_question_keys():
    s = score_question([{0}], 1)
    assert set(s) == {"recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@3", "ndcg@5", "ndcg@10"}


def test_bootstrap_ci_brackets_mean_and_is_degenerate_for_constants():
    vals = [0.0, 1.0] * 20
    lo, hi = bootstrap_ci(vals, resamples=2000)
    assert lo < 0.5 < hi
    assert bootstrap_ci([0.7] * 10, resamples=200) == (pytest.approx(0.7), pytest.approx(0.7))
