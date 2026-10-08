from types import SimpleNamespace

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from api import main
from tests.test_generate import _cite, _response, _text, fake_client


@pytest.fixture
def client(static_retriever):
    main.app.dependency_overrides[main.get_retriever] = lambda: static_retriever
    main.app.dependency_overrides[main.get_llm_client] = lambda: fake_client(
        _response([_text("Drafts are verified by the target model.", [_cite(2)])])
    )
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_search_returns_ranked_hits_with_provenance(client):
    r = client.post("/search", json={"query": "speculative decoding", "k": 2})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert [h["rank"] for h in hits] == [1, 2]
    assert hits[0]["arxiv_id"] == "p1" and hits[0]["url"] == "https://arxiv.org/abs/p1"
    assert hits[0]["chunk_id"] == "p1::0" and hits[0]["text"]


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "x", "k": 0}, {"query": "x", "k": 51}, {}])
def test_search_validation(client, body):
    assert client.post("/search", json=body).status_code == 422


def test_ask_returns_cited_answer(client):
    r = client.post("/ask", json={"question": "How are drafts checked?", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Drafts are verified by the target model.[3]"
    assert body["grounded"] is True
    assert body["citations"][0] | {"cited_text": None} == {
        "source_n": 3, "chunk_id": "p2::0", "arxiv_id": "p2", "title": "Paper Two", "cited_text": None}
    assert len(body["sources"]) == 3


def _raise(exc):
    def create(**kwargs):
        raise exc
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


@pytest.mark.parametrize("status,expected", [(401, 503), (429, 429), (500, 502)])
def test_ask_maps_llm_errors(client, status, expected):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req)
    exc_cls = {401: anthropic.AuthenticationError, 429: anthropic.RateLimitError,
               500: anthropic.InternalServerError}[status]
    main.app.dependency_overrides[main.get_llm_client] = lambda: _raise(exc_cls("boom", response=resp, body=None))
    assert client.post("/ask", json={"question": "q?"}).status_code == expected


@pytest.mark.parametrize("config_dir_set", [False, True])
def test_ask_without_any_credentials_is_503_not_500(client, monkeypatch, tmp_path, config_dir_set):
    """No key, token or profile: the real SDK fails client-side (TypeError, or
    CredentialsError when a config dir is named) before any request is sent."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.config/anthropic profile
    if config_dir_set:
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    main.app.dependency_overrides.pop(main.get_llm_client)  # exercise the real client construction
    r = client.post("/ask", json={"question": "q?"})
    assert r.status_code == 503 and "credentials" in r.json()["detail"]


def test_health_reports_retriever_state(monkeypatch):
    monkeypatch.setattr(main, "get_retriever", lambda: (_ for _ in ()).throw(RuntimeError("no index")))
    r = TestClient(main.app).get("/health")
    assert r.status_code == 200 and r.json()["retriever_ready"] is False
