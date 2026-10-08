"""Fetch arXiv ML papers, parse to text, and chunk for embedding.

    python src/ingest.py                      # pinned corpus from metadata.json (or a fresh search if empty)
    python src/ingest.py --refresh            # re-run the arXiv search for the newest papers
    python src/ingest.py --chunk-size 250 --chunk-overlap 25 --out data/chunks_250.jsonl
"""

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import arxiv
import pdfplumber
import requests
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
METADATA_PATH = PAPERS_DIR / "metadata.json"
TEXT_DIR = DATA_DIR / "text"  # parsed-text cache so re-chunking skips pdfplumber

# pdfplumber's defaults (x_tolerance=3pt) merged adjacent words in tightly-kerned
# LaTeX PDFs: 40/50 papers had glued tokens like "Thissectionprovidesthebackground"
# (7% of all tokens), invisible to BM25 and garbling embeddings. A tolerance relative
# to font size fixes that, and following the PDF's text flow reads two-column papers
# column by column instead of interleaving lines from both columns.
PARSE_SETTINGS = {"x_tolerance_ratio": 0.15, "use_text_flow": True}
PARSER_TAG = "xr015-flow"  # bump when PARSE_SETTINGS change, so cached text is not reused

# arXiv's query parser does not give AND higher precedence than OR: without the
# parentheses this parsed as `A OR B OR (C AND cats)` and 22 of the top 40 hits
# were outside cs.CL/cs.LG (cs.CV, cs.RO, ...). Verified against the live API.
DEFAULT_QUERY = (
    '(abs:"large language model" OR abs:"transformer" OR abs:"retrieval augmented") '
    "AND (cat:cs.CL OR cat:cs.LG)"
)
MAX_PAPERS = 50
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
# arXiv asks API clients to wait 3s between requests; PDF downloads count too.
DOWNLOAD_DELAY_SECONDS = 3.0
USER_AGENT = "rag-research-assistant/0.1 (portfolio project; arXiv API client)"

_encoding = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_encoding.encode(text))


def _safe_filename(arxiv_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", arxiv_id) + ".pdf"


def download_pdf(url: str, dest: Path, timeout: float = 60.0) -> None:
    """Download a PDF to `dest` atomically, rejecting anything that isn't a PDF.

    `Result.download_pdf` no longer exists in arxiv 4.x, so we fetch `Result.pdf_url` ourselves.
    """
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError(f"response from {url} is not a PDF ({resp.headers.get('content-type')})")
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list
    published: str
    categories: list
    abstract: str
    pdf_path: str


def pinned_ids() -> list[str]:
    """arXiv ids recorded by the last successful fetch, in their original order."""
    if not METADATA_PATH.exists():
        return []
    return [p["arxiv_id"] for p in json.loads(METADATA_PATH.read_text())]


def fetch_papers(
    query: str = DEFAULT_QUERY, max_results: int = MAX_PAPERS, id_list: list[str] | None = None
) -> list[PaperMeta]:
    """Search arXiv and download PDFs into data/papers/ until `max_results` succeed.

    With `id_list`, fetches exactly those papers instead of searching: the default
    query sorts by submission date, so re-running it a week later returns a different
    corpus and silently invalidates the labelled eval set.

    Skips papers already on disk. Over-fetches search results so that a few failed
    downloads (withdrawn papers, transient errors) don't leave the corpus short.
    """
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    if id_list:
        search = arxiv.Search(id_list=id_list, max_results=len(id_list))
        max_results = len(id_list)
    else:
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

    papers: list[PaperMeta] = []
    failures = 0
    for result in client.results(search):
        if len(papers) >= max_results:
            break
        arxiv_id = result.get_short_id()
        filename = _safe_filename(arxiv_id)
        pdf_path = PAPERS_DIR / filename

        if not pdf_path.exists():
            try:
                download_pdf(result.pdf_url, pdf_path)
                logger.info("Downloaded %s: %s", arxiv_id, result.title)
            except Exception as e:
                failures += 1
                logger.warning("Failed to download %s (%s): %s", arxiv_id, result.title, e)
                continue
            finally:
                time.sleep(DOWNLOAD_DELAY_SECONDS)
        else:
            logger.info("Already have %s, skipping download", arxiv_id)

        papers.append(
            PaperMeta(
                arxiv_id=arxiv_id,
                title=result.title.strip(),
                authors=[a.name for a in result.authors],
                published=result.published.isoformat(),
                categories=result.categories,
                abstract=result.summary.strip(),
                pdf_path=str(pdf_path.relative_to(DATA_DIR.parent)),
            )
        )

    if id_list:
        # The id_list endpoint returns papers in arbitrary order; restore the pinned
        # order so re-runs produce byte-identical metadata.json and chunks.jsonl.
        order = {aid: i for i, aid in enumerate(id_list)}
        papers.sort(key=lambda p: order.get(p.arxiv_id, len(order)))

    if not papers:
        # Previously this wrote "[]" and an empty chunks file and exited 0, which hid
        # the fact that every single download was failing.
        raise RuntimeError(f"No papers obtained for query={query!r} ({failures} download failures)")
    if len(papers) < max_results:
        logger.warning(
            "Requested %d papers but only obtained %d (%d download failures, query=%r).",
            max_results,
            len(papers),
            failures,
            query,
        )

    METADATA_PATH.write_text(json.dumps([asdict(p) for p in papers], indent=2))
    logger.info("Wrote metadata for %d papers to %s", len(papers), METADATA_PATH)
    return papers


def parse_pdf_cached(paper: PaperMeta) -> str:
    cache = TEXT_DIR / PARSER_TAG / (_safe_filename(paper.arxiv_id)[: -len(".pdf")] + ".txt")
    if cache.exists():
        return cache.read_text()
    text = parse_pdf(DATA_DIR.parent / paper.pdf_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text


def parse_pdf(pdf_path: Path) -> str:
    """Extract full text from a PDF using pdfplumber, page by page."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(**PARSE_SETTINGS)
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


def chunk_paper(paper: PaperMeta, text: str, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """Split a paper's text into overlapping chunks with source metadata attached."""
    raw_chunks = splitter.split_text(text)
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append(
            {
                "chunk_id": f"{paper.arxiv_id}::{i}",
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "chunk_index": i,
                "n_chunks": len(raw_chunks),
                "text": chunk_text,
            }
        )
    return chunks


def run(
    query: str = DEFAULT_QUERY,
    max_results: int = MAX_PAPERS,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
    out_path: Path = CHUNKS_PATH,
    refresh: bool = False,
) -> None:
    start = time.time()
    ids = [] if refresh else pinned_ids()
    if ids:
        logger.info("Using pinned corpus of %d papers from %s (pass --refresh to re-search)", len(ids), METADATA_PATH)
    papers = fetch_papers(query=query, max_results=max_results, id_list=ids or None)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_len,
    )

    all_chunks = []
    parse_failures = []
    for paper in papers:
        try:
            text = parse_pdf_cached(paper)
        except Exception as e:
            logger.warning("Failed to parse %s (%s): %s", paper.arxiv_id, paper.title, e)
            parse_failures.append(paper.arxiv_id)
            continue

        if not text.strip():
            logger.warning("No extractable text for %s (%s), skipping", paper.arxiv_id, paper.title)
            parse_failures.append(paper.arxiv_id)
            continue

        chunks = chunk_paper(paper, text, splitter)
        all_chunks.extend(chunks)
        logger.info("Chunked %s -> %d chunks", paper.arxiv_id, len(chunks))

    if not all_chunks:
        raise RuntimeError(f"Produced 0 chunks from {len(papers)} papers ({len(parse_failures)} parse failures)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    elapsed = time.time() - start
    logger.info(
        "Done in %.1fs: %d papers, %d parse failures, %d total chunks -> %s",
        elapsed,
        len(papers),
        len(parse_failures),
        len(all_chunks),
        out_path,
    )
    if parse_failures:
        logger.warning("Parse failures: %s", parse_failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="ignore metadata.json and re-run the search")
    parser.add_argument("--max-papers", type=int, default=MAX_PAPERS)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_TOKENS)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--out", type=Path, default=CHUNKS_PATH)
    args = parser.parse_args()
    run(
        max_results=args.max_papers,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        out_path=args.out,
        refresh=args.refresh,
    )
