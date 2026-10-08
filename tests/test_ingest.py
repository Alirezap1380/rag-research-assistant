from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src import ingest


def test_default_query_groups_terms_before_category_filter():
    # Unparenthesized, arXiv parsed this as A OR B OR (C AND cats).
    q = ingest.DEFAULT_QUERY
    assert q.startswith("(") and ") AND (cat:cs.CL OR cat:cs.LG)" in q


def test_safe_filename():
    assert ingest._safe_filename("2610.10533v1") == "2610.10533v1.pdf"
    assert ingest._safe_filename("hep-th/9901001v1") == "hep-th_9901001v1.pdf"


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status_code = content, status
        self.headers = {"content-type": "application/pdf" if content.startswith(b"%PDF") else "text/html"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_download_pdf_writes_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResponse(b"%PDF-1.5 body"))
    dest = tmp_path / "x.pdf"
    ingest.download_pdf("https://arxiv.org/pdf/x", dest)
    assert dest.read_bytes() == b"%PDF-1.5 body"
    assert not (tmp_path / "x.part").exists()


def test_download_pdf_rejects_html(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResponse(b"<html>captcha</html>"))
    with pytest.raises(ValueError, match="not a PDF"):
        ingest.download_pdf("https://arxiv.org/pdf/x", tmp_path / "x.pdf")
    assert not (tmp_path / "x.pdf").exists()


def _result(aid):
    return SimpleNamespace(
        get_short_id=lambda: aid,
        pdf_url=f"https://arxiv.org/pdf/{aid}",
        title=f" Title {aid} ",
        authors=[SimpleNamespace(name="A. Author")],
        published=datetime(2026, 10, 1, tzinfo=timezone.utc),
        categories=["cs.CL"],
        summary="Abstract.",
    )


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    papers = tmp_path / "data" / "papers"
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ingest, "PAPERS_DIR", papers)
    monkeypatch.setattr(ingest, "METADATA_PATH", papers / "metadata.json")
    monkeypatch.setattr(ingest, "DOWNLOAD_DELAY_SECONDS", 0)
    return papers


def _fake_client(results):
    return lambda **kw: SimpleNamespace(results=lambda search: iter(results))


def test_fetch_papers_raises_instead_of_writing_empty_corpus(sandbox, monkeypatch):
    """The original bug: every download failed and the run 'succeeded' with []."""
    monkeypatch.setattr(ingest.arxiv, "Client", _fake_client([_result("1"), _result("2")]))

    def boom(url, dest):
        raise AttributeError("'Result' object has no attribute 'download_pdf'")

    monkeypatch.setattr(ingest, "download_pdf", boom)
    with pytest.raises(RuntimeError, match="No papers obtained"):
        ingest.fetch_papers(max_results=2)
    assert not (sandbox / "metadata.json").exists()


def test_fetch_papers_skips_failures_and_stops_at_max(sandbox, monkeypatch):
    monkeypatch.setattr(ingest.arxiv, "Client", _fake_client([_result(str(i)) for i in range(5)]))

    def flaky(url, dest):
        if url.endswith("/1"):
            raise RuntimeError("404")
        dest.write_bytes(b"%PDF")

    monkeypatch.setattr(ingest, "download_pdf", flaky)
    papers = ingest.fetch_papers(max_results=3)
    assert [p.arxiv_id for p in papers] == ["0", "2", "3"]
    assert papers[0].title == "Title 0"


def test_fetch_pinned_ids_restores_order(sandbox, monkeypatch):
    # The id_list endpoint returns papers in arbitrary order.
    monkeypatch.setattr(ingest.arxiv, "Client", _fake_client([_result("b"), _result("c"), _result("a")]))
    monkeypatch.setattr(ingest, "download_pdf", lambda url, dest: dest.write_bytes(b"%PDF"))
    papers = ingest.fetch_papers(id_list=["a", "b", "c"])
    assert [p.arxiv_id for p in papers] == ["a", "b", "c"]
    assert ingest.pinned_ids() == ["a", "b", "c"]


def test_chunk_paper_ids_and_token_bound():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    paper = ingest.PaperMeta("2610.1v1", "T", [], "", [], "", "data/papers/x.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5, length_function=ingest._token_len)
    chunks = ingest.chunk_paper(paper, "word " * 400, splitter)
    assert len(chunks) > 1
    assert [c["chunk_id"] for c in chunks[:2]] == ["2610.1v1::0", "2610.1v1::1"]
    assert all(ingest._token_len(c["text"]) <= 50 for c in chunks)
    assert {c["n_chunks"] for c in chunks} == {len(chunks)}
