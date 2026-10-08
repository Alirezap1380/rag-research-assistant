from types import SimpleNamespace

from src import generate
from src.retrieval import Hit


def _text(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def _cite(idx, cited_text="..."):
    return SimpleNamespace(
        type="search_result_location", search_result_index=idx, cited_text=cited_text,
        source="https://arxiv.org/abs/x", title="t", start_block_index=0, end_block_index=1,
    )


def _response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason, model=generate.MODEL)


def _hits(corpus, n=3):
    return [Hit(c, 1.0 / r, r) for r, c in enumerate(corpus[:n], start=1)]


def test_user_content_is_one_search_result_per_hit_then_question(corpus):
    blocks = generate.build_user_content("Q?", _hits(corpus))
    assert [b["type"] for b in blocks] == ["search_result"] * 3 + ["text"]
    first = blocks[0]
    assert first["source"] == "https://arxiv.org/abs/p1"
    assert "arXiv:p1" in first["title"] and first["citations"] == {"enabled": True}
    assert first["content"] == [{"type": "text", "text": corpus[0].text}]
    assert blocks[-1] == {"type": "text", "text": "Q?"}


def test_citations_map_back_to_the_retrieved_chunk(corpus):
    hits = _hits(corpus)
    resp = _response([
        SimpleNamespace(type="thinking", thinking=""),
        _text("KV cache is 2-bit", [_cite(0, corpus[0].text)]),
        _text(" and drafts are verified", [_cite(2), _cite(2)]),
        _text("."),
    ])
    ans = generate.parse_response("Q?", hits, resp)
    assert ans.answer == "KV cache is 2-bit[1] and drafts are verified[3]."
    assert ans.grounded
    assert [(c.source_n, c.chunk_id, c.arxiv_id) for c in ans.citations] == [
        (1, "p1::0", "p1"), (3, "p2::0", "p2"), (3, "p2::0", "p2")]
    assert ans.citations[0].title == "Paper One"
    assert [s.chunk_id for s in ans.sources] == ["p1::0", "p1::1", "p2::0"]  # all retrieved, cited or not


def test_out_of_range_citation_is_dropped(corpus):
    ans = generate.parse_response("Q?", _hits(corpus, 2), _response([_text("x", [_cite(7)])]))
    assert ans.citations == [] and not ans.grounded


def test_uncited_answer_is_flagged_ungrounded(corpus):
    ans = generate.parse_response("Q?", _hits(corpus), _response([_text("From memory, probably 4 bits.")]))
    assert not ans.grounded
    assert generate.render(ans).startswith("WARNING")


def test_refusal_is_reported_not_parsed(corpus):
    ans = generate.parse_response("Q?", _hits(corpus), _response([], stop_reason="refusal"))
    assert ans.stop_reason == "refusal" and not ans.grounded
    assert "declined" in ans.answer


class FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(response):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(response)))


def test_answer_question_request_shape(static_retriever, corpus):
    client = fake_client(_response([_text("Steering vectors.", [_cite(0)])]))
    ans = generate.answer_question("How is emotion controlled?", static_retriever, k=2, client=client)
    call = client.beta.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "medium"}
    assert "thinking" not in call and "temperature" not in call  # rejected on this model
    assert len(call["messages"][0]["content"]) == 3  # 2 search results + question
    assert static_retriever.queries == [("How is emotion controlled?", 2)]
    assert ans.citations[0].chunk_id == "p1::0"


def test_no_hits_skips_the_llm(corpus):
    class Empty:
        def search(self, q, k=5):
            return []

    client = fake_client(None)
    ans = generate.answer_question("Q?", Empty(), client=client)
    assert client.beta.messages.calls == [] and not ans.grounded
