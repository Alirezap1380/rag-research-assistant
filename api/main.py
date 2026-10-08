"""FastAPI service: retrieval-only search and cited answer synthesis.

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import AskRequest, AskResponse, HealthResponse, SearchHit, SearchRequest, SearchResponse
from src import generate
from src.retrieval import DEFAULT_CHROMA_DIR, SERVED_CHUNKS_PATH, SERVED_MODE, Retriever, build_retriever

# Defaults are the configuration that won the retrieval eval (see README / eval/results.md).
RETRIEVAL_MODE = os.getenv("RAG_RETRIEVAL_MODE", SERVED_MODE)
CHUNKS_PATH = Path(os.getenv("RAG_CHUNKS_PATH", str(SERVED_CHUNKS_PATH)))
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", str(DEFAULT_CHROMA_DIR)))

app = FastAPI(
    title="arXiv RAG Research Assistant",
    description="Retrieval over arXiv ML papers with citation-grounded answers.",
    version="0.1.0",
)


@app.exception_handler(generate.LLMNotConfigured)
def llm_not_configured(request: Request, exc: generate.LLMNotConfigured) -> JSONResponse:
    # Raised either when the client is constructed (a dependency, outside the endpoint)
    # or on the first request; /search keeps working without credentials.
    return JSONResponse(status_code=503, content={"detail": "LLM credentials are missing or invalid"})


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return build_retriever(RETRIEVAL_MODE, CHUNKS_PATH, CHROMA_DIR)


def get_llm_client():
    return generate.get_client()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_retriever()
        ready = True
    except Exception:
        ready = False
    return HealthResponse(status="ok", retriever_ready=ready, retrieval_mode=RETRIEVAL_MODE, chunks=CHUNKS_PATH.name)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, retriever: Retriever = Depends(get_retriever)) -> SearchResponse:
    hits = retriever.search(req.query, k=req.k)
    return SearchResponse(
        query=req.query,
        hits=[
            SearchHit(
                rank=h.rank,
                score=h.score,
                chunk_id=h.chunk.chunk_id,
                arxiv_id=h.chunk.arxiv_id,
                title=h.chunk.title,
                url=generate.arxiv_url(h.chunk.arxiv_id),
                text=h.chunk.text,
            )
            for h in hits
        ],
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    retriever: Retriever = Depends(get_retriever),
    client=Depends(get_llm_client),
) -> AskResponse:
    import anthropic

    try:
        ans = generate.answer_question(req.question, retriever, k=req.k, client=client)
    except anthropic.AuthenticationError as e:
        raise HTTPException(status_code=503, detail="LLM credentials are missing or invalid") from e
    except anthropic.RateLimitError as e:
        raise HTTPException(status_code=429, detail="LLM rate limit reached; retry later") from e
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error ({e.status_code})") from e
    except anthropic.APIConnectionError as e:
        raise HTTPException(status_code=502, detail="Could not reach the LLM API") from e
    return AskResponse(**ans.to_dict())
