"""Answer synthesis with Claude, grounded in retrieved chunks and traceable to them.

Retrieved chunks are passed as `search_result` content blocks with citations
enabled, so every cited span in the answer comes back with a `search_result_index`
that maps to exactly one retrieved chunk (and through it to arxiv_id and title).
We don't ask the model to write "[1]"-style markers itself; the citation offsets are
produced by the API, so a citation can't point at a source that wasn't retrieved.

    python -m src.generate "How does Dual-QK prune the KV cache?"
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

from src.retrieval import MODES, SERVED_CHUNKS_PATH, SERVED_MODE, Hit, Retriever, build_retriever

MODEL = "claude-opus-5-5"
# Grounded Q&A over a handful of passages: medium effort (the model's default, set
# explicitly so behaviour doesn't drift if the default changes).
EFFORT = "medium"
MAX_TOKENS = 16000
# Server-side refusal fallback: if the model declines for policy reasons the API
# re-runs the request on Anthropic's recommended fallback model in the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You answer questions about machine-learning research papers for a researcher.

The user message contains search results: passages retrieved from arXiv papers, each \
titled with the paper title and arXiv id. Answer using only information in those \
passages, and cite the passages you rely on. Passages come from PDF text extraction, \
so expect stray line breaks, broken tables and equation fragments.

If the passages do not contain enough information to answer, say so directly and \
state what they do cover, rather than answering from general knowledge. When \
passages from different papers disagree or describe different settings, keep the \
papers distinct instead of blending their claims. Be concise: a few short paragraphs \
at most, no headings."""


@dataclass
class Source:
    n: int  # 1-based number used in the rendered answer
    chunk_id: str
    arxiv_id: str
    title: str
    url: str
    retrieval_rank: int
    retrieval_score: float


@dataclass
class Citation:
    source_n: int
    chunk_id: str
    arxiv_id: str
    title: str
    cited_text: str


@dataclass
class Answer:
    question: str
    answer: str  # rendered text with [n] markers after each cited span
    citations: list[Citation] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)  # every retrieved chunk, cited or not
    model: str | None = None
    stop_reason: str | None = None
    grounded: bool = False  # True iff at least one citation came back

    def to_dict(self) -> dict:
        return asdict(self)


def arxiv_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def build_user_content(question: str, hits: list[Hit]) -> list[dict]:
    blocks: list[dict] = [
        {
            "type": "search_result",
            "source": arxiv_url(h.chunk.arxiv_id),
            "title": f"{h.chunk.title} (arXiv:{h.chunk.arxiv_id}, chunk {h.chunk.chunk_index})",
            "content": [{"type": "text", "text": h.chunk.text}],
            "citations": {"enabled": True},
        }
        for h in hits
    ]
    blocks.append({"type": "text", "text": question})
    return blocks


def parse_response(question: str, hits: list[Hit], response) -> Answer:
    """Turn API content blocks into answer text with [n] markers plus a citation list.

    `search_result_index` counts search_result blocks in request order, which is the
    order of `hits`, so hits[index] is the cited chunk."""
    sources = [
        Source(
            n=i + 1,
            chunk_id=h.chunk.chunk_id,
            arxiv_id=h.chunk.arxiv_id,
            title=h.chunk.title,
            url=arxiv_url(h.chunk.arxiv_id),
            retrieval_rank=h.rank,
            retrieval_score=h.score,
        )
        for i, h in enumerate(hits)
    ]
    answer = Answer(question=question, answer="", sources=sources,
                    model=getattr(response, "model", None), stop_reason=response.stop_reason)

    if response.stop_reason == "refusal":
        answer.answer = "The model declined to answer this question."
        return answer

    parts: list[str] = []
    for block in response.content:
        if block.type != "text":
            continue  # thinking / fallback blocks carry no answer text
        parts.append(block.text)
        markers = []
        for c in getattr(block, "citations", None) or []:
            if c.type != "search_result_location" or not 0 <= c.search_result_index < len(hits):
                continue
            src = sources[c.search_result_index]
            answer.citations.append(Citation(src.n, src.chunk_id, src.arxiv_id, src.title, c.cited_text))
            if f"[{src.n}]" not in markers:
                markers.append(f"[{src.n}]")
        if markers:
            parts.append("".join(markers))
    answer.answer = "".join(parts).strip()
    answer.grounded = bool(answer.citations)
    return answer


def get_client():
    import anthropic

    # Resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile.
    return anthropic.Anthropic()


def answer_question(question: str, retriever: Retriever, k: int = 5, client=None) -> Answer:
    hits = retriever.search(question, k=k)
    if not hits:
        return Answer(question=question, answer="No passages were retrieved for this question.")
    client = client or get_client()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_content(question, hits)}],
        output_config={"effort": EFFORT},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    return parse_response(question, hits, response)


def render(ans: Answer) -> str:
    lines = [ans.answer, "", "Sources:"]
    cited = {c.source_n for c in ans.citations}
    for s in ans.sources:
        mark = "cited" if s.n in cited else "retrieved, not cited"
        lines.append(f"  [{s.n}] {s.title} — arXiv:{s.arxiv_id} ({s.chunk_id}, {mark})")
    if not ans.grounded:
        lines.insert(0, "WARNING: answer carries no citations; treat it as ungrounded.\n")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--mode", default=SERVED_MODE, choices=MODES)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    print(render(answer_question(args.question, build_retriever(args.mode, SERVED_CHUNKS_PATH), k=args.k)))
