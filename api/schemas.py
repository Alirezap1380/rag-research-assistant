from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    retriever_ready: bool
    retrieval_mode: str
    chunks: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    rank: int
    score: float
    chunk_id: str
    arxiv_id: str
    title: str
    url: str
    text: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    n: int
    chunk_id: str
    arxiv_id: str
    title: str
    url: str
    retrieval_rank: int
    retrieval_score: float


class Citation(BaseModel):
    source_n: int
    chunk_id: str
    arxiv_id: str
    title: str
    cited_text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    sources: list[Source]
    model: str | None
    stop_reason: str | None
    grounded: bool
