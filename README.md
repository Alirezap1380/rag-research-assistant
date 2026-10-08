# arXiv RAG Research Assistant

## Problem

Ask a question about recent machine-learning papers and get an answer whose every
claim points at the exact passage, paper and arXiv id it came from. The corpus is
the 50 newest cs.CL/cs.LG papers matching LLM / transformer / retrieval-augmented
topics (pinned arXiv ids in `data/papers/metadata.json`, submitted October 2026),
parsed from PDF and split into passages.

Most RAG demos stop at "it returns something plausible". This repo instead treats
retrieval as a measurable component: a labelled set of 89 questions, recall@k / MRR
/ nDCG with confidence intervals, and six retrieval configurations compared on it.
The configuration that is served is the one that won that comparison, which turned
out not to be the one originally planned.

## Results

Six configurations: three retrievers × two chunk sizes. `dense` is
`BAAI/bge-small-en-v1.5` embeddings in Chroma (cosine). `bm25` is Okapi BM25
(`rank-bm25`). `hybrid` fuses the top 50 of each with reciprocal rank fusion
(k=60). The chunk sizes are 500 and 250 tiktoken tokens (50 / 25 overlap). The
scores are means over 89 questions, with 95% bootstrap confidence intervals.

| Config | recall@1 | recall@5 | recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| dense-500 | 0.331 | 0.719 [0.62, 0.80] | 0.803 | 0.490 [0.41, 0.57] | 0.565 |
| bm25-500 | 0.478 | 0.815 [0.73, 0.89] | 0.893 | 0.632 [0.55, 0.71] | 0.693 |
| hybrid-500 | 0.404 | 0.764 [0.67, 0.85] | 0.854 | 0.572 [0.49, 0.65] | 0.636 |
| dense-250 | 0.309 | 0.674 [0.58, 0.77] | 0.781 | 0.485 [0.40, 0.57] | 0.555 |
| **bm25-250** | **0.562** | **0.826 [0.75, 0.90]** | 0.871 | **0.681 [0.60, 0.76]** | **0.722** |
| hybrid-250 | 0.461 | 0.747 [0.65, 0.83] | 0.848 | 0.605 [0.52, 0.69] | 0.655 |

Paired differences (same questions, bootstrap over questions):

| Comparison (B − A) | recall@5 | MRR@10 |
|---|---|---|
| hybrid-250 − dense-250 | +0.073 [−0.01, +0.15] | **+0.119 [+0.06, +0.18]** |
| hybrid-250 − bm25-250 | **−0.079 [−0.15, −0.01]** | **−0.077 [−0.15, −0.00]** |
| dense-250 − dense-500 | −0.045 [−0.14, +0.05] | −0.005 [−0.07, +0.07] |
| hybrid-250 − hybrid-500 | −0.017 [−0.07, +0.03] | +0.033 [−0.03, +0.09] |

The full tables, with every metric, CI and breakdown, are in
[`eval/results.md`](eval/results.md), and per-question scores are in
`eval/results.json`. Reproduce them with `python -m src.evaluate`.

**BM25 over 250-token chunks is served.** That is not the original plan, which was
Chroma plus hybrid retrieval. The argument for the switch:

- **BM25 is better, and the difference is significant.** Hybrid RRF clearly beats
  dense alone (MRR +0.12, CI excludes zero), so fusion helps. But adding the dense
  ranking to BM25 makes BM25 worse on recall@5, MRR and nDCG, and every one of those
  CIs excludes zero.
- **The eval set favours BM25, and this was checked rather than assumed.** I wrote
  the questions while reading the passages, so they share vocabulary with the
  answers: the median question has 62% of its content words in its gold chunk.
  Splitting the questions into overlap tertiles:

  | Overlap tertile | dense-250 | bm25-250 | hybrid-250 |
  |---|---|---|---|
  | low (0.00–0.55, n=29) | 0.534 / 0.362 | 0.569 / 0.415 | 0.569 / 0.414 |
  | mid (0.56–0.71, n=30) | 0.717 / 0.528 | 0.900 / 0.781 | 0.800 / 0.679 |
  | high (0.71–1.00, n=30) | 0.767 / 0.561 | 1.000 / 0.839 | 0.867 / 0.714 |

  (recall@5 / MRR@10.) BM25's lead comes entirely from questions that reuse the
  passage's words. On the low-overlap third, which is closest to a user paraphrasing
  a paper they haven't read, hybrid **ties** BM25 but still does not beat it.
  Dense alone is last in every tertile.
- So on this evidence there is no question regime where hybrid wins. BM25 is also
  simpler to run: no embedding model, no vector store, and no query-time inference.
  The case for hybrid (robustness to paraphrase) is plausible but **unmeasured
  here**. The Chroma index stays built and `RAG_RETRIEVAL_MODE=hybrid` switches to
  it, so the decision is easy to reverse once there is evidence such as real user
  queries.
- **Why the corpus suits BM25:** these papers name their own methods (EngramEdit,
  HySPE, AUDITOPUS), datasets and model ids (`Qwen3-4B-Instruct-2507`), and
  questions about them hinge on those rare exact tokens. BM25's IDF rewards them,
  while a 33M-parameter embedding model blurs them into topic similarity. On the
  83 single-paper questions, every retriever puts the right *paper* in the top 5
  every time (1.000 for all three). The difference is entirely in finding the right
  *passage* within it: dense 0.687, hybrid 0.759, BM25 0.831.

**Chunk size makes no measurable difference.** All 250-vs-500 differences have CIs
spanning zero. 250 is served because BM25-250 has the best point estimates on every
headline metric and gives the generator tighter passages. The original motivation for
smaller chunks was that 25.9% of 500-token chunks exceeded BGE's 512-WordPiece input
and were silently truncated. That turned out to be mostly a side effect of the PDF
parsing bug below: glued words split into many WordPiece fragments. After the fix,
only 1.7% of 500-token chunks exceed the limit.

### Why recall@5 and MRR are the headline metrics

The generator is given the top 5 passages, so **recall@5** measures whether the
evidence reaches the model at all. If it doesn't, no prompt can produce a grounded
answer. **MRR@10** measures how high the first relevant passage sits, which matters
because models weight early context more and because users read `/search` results
top-down. **nDCG** is reported with *novelty-aware* gain: a second chunk restating
the same fact earns nothing, because it gives the generator no new information. With
mostly single-fact questions, nDCG tracks MRR closely, so it is a check on MRR rather
than a separate signal.

## How the evaluation set was built

`eval/eval_set.jsonl` has 89 questions covering all 50 papers. 51 are *descriptive*
(paraphrased, without naming the method), 32 are *named* (mention the method or
paper), and 6 are *multi* (need evidence from two papers).

- **Labels are verbatim quotes, not chunk ids.** Each question lists evidence units
  as `{arxiv_id, quotes: [...]}`. A retrieved chunk is relevant if it comes from that
  paper and contains one of the quotes, after normalising ligatures, curly quotes,
  hyphenation and whitespace. Because of this, the same 89 labels score both
  chunkings. With chunk-id labels, a chunk-size comparison would be impossible.
- **Pooling for missed evidence.** Papers restate results in the abstract,
  introduction and conclusion, often in different words. To catch these, every
  configuration's top-5 chunks from the right paper were pooled, screened by
  key-term overlap and judged by hand, TREC-style. 45 of the 89 questions gained
  alternative quotes this way. Without this pass, a retriever that surfaced the
  abstract's version of a fact would have been scored as wrong.
- **Labels are tested.** `tests/test_eval_set.py` asserts that every evidence unit
  resolves to at least one chunk in both chunkings, so re-chunking can't silently
  orphan a label.
- **Pooling also found a harness bug.** The judge cached normalised text by
  `chunk_id`, but both chunkings name chunks `<arxiv_id>::<i>`, so 250-token chunks
  were being judged against 500-token text. It is fixed, with a regression test.

**Limitations.** All labels were written by one person who had read the passages,
which is the overlap bias quantified above. 89 questions give wide CIs (±0.08 on
recall@5), so differences under about 0.07 are not resolvable. The configs were
compared, not tuned, on this set: there is no held-out split, so treat any future
tuning against it (RRF weights, a bigger embedder) as overfitting unless you add
one.

## Ingestion: what was broken

`src/ingest.py` (arXiv fetch → pdfplumber → token-aware chunking) had never produced
output. In order of discovery:

1. **Every download failed silently.** `arxiv` 4.x removed `Result.download_pdf`.
   Each call raised `AttributeError`, which was caught as a per-paper warning, so the
   run wrote `metadata.json = []` and an empty `chunks.jsonl` and still exited 0.
   PDFs are now fetched from `Result.pdf_url`: written atomically, checked for
   `%PDF`, rate-limited to 3s per arXiv's API terms. An empty corpus now raises
   instead of being written. Dependencies are pinned.
2. **The query's boolean precedence was wrong**, though this wasn't the cause of the
   empty output. arXiv parsed `A OR B OR C AND (cats)` as `A OR B OR (C AND cats)`:
   249,078 hits, with 22 of the newest 40 outside cs.CL/cs.LG (cs.CV, cs.RO, …).
   Parenthesised, it gives 85,247 hits and 0 of 40 off-topic.
3. **pdfplumber glued words together.** With default settings, 40 of 50 papers had
   tokens like `Thissectionprovidesthebackground` (7% of all tokens), and two-column
   pages were read line-by-line across both columns. `x_tolerance_ratio=0.15` plus
   `use_text_flow=True` brings glued tokens to 0.01% and reads columns in order.
   This fix came before labelling. BM25 cannot match a glued word, and the BGE
   tokenizer splits it into fragments.
4. **The corpus wasn't reproducible.** "50 newest papers" changes daily, which would
   invalidate the labels. Re-runs now fetch the pinned ids in `metadata.json`
   (`--refresh` re-searches), and a re-run is byte-identical. Parsed text is cached,
   so re-chunking takes about 1s instead of 5 minutes.

## Answers that cite their sources

`src/generate.py` passes the top-k chunks to Claude (`claude-opus-5-5`, effort
`medium`) as [`search_result` content blocks](https://platform.claude.com/docs/en/build-with-claude/search-results)
with citations enabled. Each cited span in the response carries a
`search_result_index`, which maps to exactly one retrieved chunk, so every citation
resolves to `chunk_id`, `arxiv_id`, `title` and the cited text.

- The model is not asked to write `[1]`-style markers. The API attaches citation
  locations to the text it generates, so a citation can only point at a passage that
  was actually retrieved. Out-of-range indices are dropped defensively.
- An answer with no citations comes back `grounded: false`, and the CLI prints a
  warning. The system prompt tells the model to say when the passages don't contain
  the answer, rather than fall back on general knowledge.
- Refusals (`stop_reason: "refusal"`) are reported rather than parsed. Server-side
  refusal fallback (`fallbacks: "default"`) is enabled.
- `/ask` returns 503 when no Anthropic credentials are configured, while `/search`
  keeps working without them.

Response shape (`POST /ask`):

```json
{
  "question": "...",
  "answer": "Answer text with a marker after each cited span.[2]",
  "citations": [{"source_n": 2, "chunk_id": "2610.09835v1::48", "arxiv_id": "2610.09835v1",
                 "title": "A Deafening Silence: ...", "cited_text": "..."}],
  "sources": [{"n": 1, "chunk_id": "...", "arxiv_id": "...", "url": "https://arxiv.org/abs/...",
               "retrieval_rank": 1, "retrieval_score": 20.16, "title": "..."}],
  "grounded": true, "stop_reason": "end_turn", "model": "claude-opus-5-5"
}
```

## Project structure

```
rag-research-assistant/
├── src/
│   ├── ingest.py        # arXiv fetch (pinned ids) → pdfplumber → token-aware chunking
│   ├── index.py         # embed a chunk file with BGE-small (fastembed/ONNX) → Chroma
│   ├── retrieval.py     # Dense / BM25 / Hybrid (RRF) retrievers behind search(query, k)
│   ├── evaluate.py      # quote-based judge, recall/MRR/nDCG, bootstrap CIs, breakdowns
│   └── generate.py      # Claude answer synthesis with search-result citations
├── api/                 # FastAPI: GET /health, POST /search, POST /ask
├── eval/                # eval_set.jsonl (89 questions), results.md / results.json
├── data/papers/metadata.json   # pinned corpus (PDFs, text, chunks are generated, not committed)
├── tests/               # 52 offline tests (fake embedder, temp Chroma, mocked Claude)
├── Dockerfile
└── .github/workflows/ci.yml    # ruff + pytest on every push
```

## Running locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python src/ingest.py                                   # 50 PDFs → data/chunks.jsonl (~5 min first run)
python src/ingest.py --chunk-size 250 --chunk-overlap 25 --out data/chunks_250.jsonl
python -m src.index                                     # Chroma index per chunk file (~3 min each, CPU)
python -m src.index --chunks data/chunks_250.jsonl

python -m src.evaluate                                  # reproduces eval/results.md
python -m src.evaluate --misses bm25-250                # error analysis: failures at k=5

export ANTHROPIC_API_KEY=...                            # only needed for answers
python -m src.generate "Which parameters does catastrophic forgetting concentrate in?"
uvicorn api.main:app --reload                           # Swagger UI at /docs
```

Only `src/index.py` and the `dense`/`hybrid` modes need the embedding model and
Chroma. The served BM25 mode needs only the chunk file. Full-text chunks and PDFs
are not committed: they are derived from arXiv papers, about 120 MB, and are
regenerated from the pinned ids.

```bash
curl -s localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "How do Unrolled Flow Models keep latent states stable?", "k": 3}'
```

## Docker

The image bakes in the 250-token chunk file, the Chroma index and the embedding
model, so it serves with no network access. Build it after generating the
artifacts above:

```bash
docker build -t rag-research-assistant .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... rag-research-assistant
```

`RAG_RETRIEVAL_MODE` (`bm25` | `dense` | `hybrid`) and `RAG_CHUNKS_PATH` override the
served configuration. The container has a `HEALTHCHECK` on `/health`.

## Testing & CI

```bash
pytest -v
ruff check src api tests
```

The 52 tests cover the original ingestion failure (every download failing must raise,
not write `[]`), pinned-order restoration, the metrics' edge cases (duplicate
evidence, novelty-capped nDCG, MRR cutoff), the judge (including the shared-chunk-id
regression), RRF fusion, a real Chroma round-trip with a fake embedder, citation
mapping, and the API (validation, error mapping, missing credentials). Everything runs
offline. The two tests that check labels against generated chunk files skip
themselves in CI, where the corpus isn't present.

## Next steps

- **Answer-quality evaluation.** The retrieval side is measured. The generation side
  (faithfulness of answers to cited passages, answer correctness on the 89
  questions) is not yet. That is the natural next layer, e.g. RAGAS faithfulness or
  an LLM judge checking each citation against its claim.
- **Questions written without seeing the passages**, or real user queries, to
  measure the paraphrase regime where hybrid might pay off. Add a held-out split
  before tuning RRF weights or trying a larger embedder.
