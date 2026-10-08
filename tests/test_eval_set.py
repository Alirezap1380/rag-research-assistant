import json

import pytest

from src.evaluate import CHUNKINGS, EVAL_SET_PATH, EvalItem, EvidenceUnit, Judge, load_eval_set, normalize
from src.retrieval import Chunk, load_chunks


def test_normalize_handles_pdf_noise():
    assert normalize("ﬁne-\ntuning  the\nmodel’s") == "finetuning the model's"
    assert normalize("scale-\nspecific") == normalize("scale-specific")
    assert normalize("A  B\n\nC") == "a b c"


def test_judge_requires_matching_paper(corpus):
    item = EvalItem("q", "?", (EvidenceUnit("p2", ("expected decoding rounds",)),))
    judge = Judge([item])
    assert judge.covered("q", corpus[3]) == {0}
    impostor = Chunk("p9::0", "p9", "Other", 0, "minimize expected decoding rounds")
    assert judge.covered("q", impostor) == set()  # right text, wrong paper


def test_judge_is_not_fooled_by_chunk_ids_shared_across_chunkings():
    """Regression: both chunkings name chunks '<arxiv_id>::<i>'. A judge cache keyed
    on chunk_id scored 250-token chunks against the 500-token chunk's text."""
    item = EvalItem("q", "?", (EvidenceUnit("p1", ("the answer is here",)),))
    judge = Judge([item])
    big = Chunk("p1::4", "p1", "T", 4, "preamble. the answer is here. more text")
    small = Chunk("p1::4", "p1", "T", 4, "preamble only")
    assert judge.covered("q", big) == {0}
    assert judge.covered("q", small) == set()
    assert judge.covered("q", big) == {0}


def test_judge_alternative_quotes_and_unresolved(corpus):
    items = [
        EvalItem("a", "?", (EvidenceUnit("p1", ("not in corpus", "2 bits")),)),
        EvalItem("b", "?", (EvidenceUnit("p3", ("nowhere to be found",)),)),
    ]
    judge = Judge(items)
    assert judge.covered("a", corpus[0]) == {0}
    assert judge.unresolved(items, corpus) == [("b", 0)]


def test_eval_set_schema():
    items = load_eval_set()
    assert len(items) >= 40
    assert len({it.qid for it in items}) == len(items)
    raw = [json.loads(line) for line in EVAL_SET_PATH.read_text().splitlines() if line.strip()]
    meta_ids = {p["arxiv_id"] for p in json.loads((EVAL_SET_PATH.parents[1] / "data/papers/metadata.json").read_text())}
    for d in raw:
        assert d["question"].strip().endswith("?")
        for e in d["evidence"]:
            assert e["arxiv_id"] in meta_ids
            assert all(len(q.split()) >= 4 for q in e["quotes"]), "quotes must be specific enough to locate"


@pytest.mark.parametrize("size", sorted(CHUNKINGS))
def test_every_label_resolves_to_a_chunk(size):
    """Every evidence quote must exist in each chunking, or that config is mis-scored.
    Needs the generated chunk files, so it is skipped in CI."""
    path = CHUNKINGS[size]
    if not path.exists():
        pytest.skip(f"{path} not generated")
    items = load_eval_set()
    assert Judge(items).unresolved(items, load_chunks(path)) == []
