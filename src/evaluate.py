"""Retrieval evaluation: recall@k, MRR@10 and nDCG@k over a labelled question set.

    python -m src.evaluate                  # all configurations -> eval/results.{json,md}
    python -m src.evaluate --misses hybrid-250   # print the questions a config fails at k=5

Labels are chunking-independent. Each question lists *evidence units*; a unit is one
fact, given as a short verbatim quote (plus optional alternative quotes where the
paper states the same fact elsewhere) and the arXiv id it comes from. A retrieved
chunk "covers" a unit if it is from that paper and contains one of its quotes. This
lets the same labels score 500-token and 250-token chunkings, whose chunk ids differ.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from src.retrieval import DATA_DIR, MODES, ROOT, Chunk, Embedder, Retriever, build_retriever, load_chunks, tokenize

EVAL_DIR = ROOT / "eval"
EVAL_SET_PATH = EVAL_DIR / "eval_set.jsonl"
KS = (1, 3, 5, 10)
MRR_CUTOFF = 10
CHUNKINGS = {"500": DATA_DIR / "chunks.jsonl", "250": DATA_DIR / "chunks_250.jsonl"}
BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class EvidenceUnit:
    arxiv_id: str
    quotes: tuple[str, ...]


@dataclass(frozen=True)
class EvalItem:
    qid: str
    question: str
    evidence: tuple[EvidenceUnit, ...]
    qtype: str = "factual"


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[EvalItem]:
    items = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            units = tuple(EvidenceUnit(e["arxiv_id"], tuple(e["quotes"])) for e in d["evidence"])
            items.append(EvalItem(d["qid"], d["question"], units, d.get("type", "factual")))
    return items


_LIGATURES = str.maketrans(
    {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "’": "'", "‘": "'", "“": '"', "”": '"'}
)


def normalize(text: str) -> str:
    """Make quote matching robust to PDF-extraction noise: ligatures, curly quotes,
    line breaks / whitespace runs, and hyphens. A line break after a hyphen can't tell
    "re-\\nstore" (soft hyphen) from "scale-\\nspecific" (real hyphen), so hyphens are
    dropped entirely on both sides of the comparison."""
    text = unicodedata.normalize("NFKC", text).translate(_LIGATURES).lower()
    text = re.sub(r"-\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


class Judge:
    """Maps (question, chunk) -> set of evidence-unit indices the chunk covers."""

    def __init__(self, items: list[EvalItem]):
        self._units = {it.qid: [(u.arxiv_id, [normalize(q) for q in u.quotes]) for u in it.evidence] for it in items}
        self._norm_cache: dict[str, str] = {}

    def covered(self, qid: str, chunk: Chunk) -> set[int]:
        # Keyed on the text, not chunk_id: every chunking numbers its chunks
        # "<arxiv_id>::<i>", so the same id names different text in different configs.
        text = self._norm_cache.get(chunk.text)
        if text is None:
            text = self._norm_cache[chunk.text] = normalize(chunk.text)
        return {
            i
            for i, (aid, quotes) in enumerate(self._units[qid])
            if aid == chunk.arxiv_id and any(q in text for q in quotes)
        }

    def unresolved(self, items: list[EvalItem], chunks: list[Chunk]) -> list[tuple[str, int]]:
        """(qid, unit index) pairs whose quotes appear in no chunk; such labels are broken."""
        found = {(it.qid, i) for it in items for c in chunks for i in self.covered(it.qid, c)}
        return [(it.qid, i) for it in items for i in range(len(it.evidence)) if (it.qid, i) not in found]


# ------------------------------------------------------------------------- metrics
# Each metric takes `ranked`: for each retrieved rank, the set of evidence units that
# chunk covers (empty set = irrelevant), and `n_units`, the number of units required.


def recall_at_k(ranked: list[set[int]], n_units: int, k: int) -> float:
    """Fraction of the question's evidence units covered by the top-k chunks."""
    covered = set().union(*ranked[:k]) if ranked[:k] else set()
    return len(covered) / n_units


def reciprocal_rank(ranked: list[set[int]], cutoff: int = MRR_CUTOFF) -> float:
    """1/rank of the first chunk covering any evidence unit (0 if none in the cutoff)."""
    for r, units in enumerate(ranked[:cutoff], start=1):
        if units:
            return 1.0 / r
    return 0.0


def ndcg_at_k(ranked: list[set[int]], n_units: int, k: int) -> float:
    """Binary nDCG with novelty: a chunk earns gain only for units not already covered
    higher up. A second chunk restating the same fact adds nothing for the generator,
    so it should not add to the score either. The ideal ranking covers one new unit
    per rank; the score is capped at 1 for the rare chunk that covers two units."""
    seen: set[int] = set()
    dcg = 0.0
    for r, units in enumerate(ranked[:k], start=1):
        new = units - seen
        if new:
            dcg += len(new) / math.log2(r + 1)
            seen |= new
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(n_units, k) + 1))
    return min(1.0, dcg / idcg)


def score_question(ranked: list[set[int]], n_units: int, ks=KS) -> dict[str, float]:
    out = {f"recall@{k}": recall_at_k(ranked, n_units, k) for k in ks}
    out[f"mrr@{MRR_CUTOFF}"] = reciprocal_rank(ranked)
    out.update({f"ndcg@{k}": ndcg_at_k(ranked, n_units, k) for k in ks if k > 1})
    return out


def bootstrap_ci(values: list[float], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling questions with replacement."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


# ------------------------------------------------------------------------- harness


def evaluate_retriever(retriever: Retriever, items: list[EvalItem], judge: Judge, depth: int = max(KS)) -> dict:
    per_q = {}
    for it in items:
        hits = retriever.search(it.question, k=depth)
        ranked = [judge.covered(it.qid, h.chunk) for h in hits]
        per_q[it.qid] = score_question(ranked, len(it.evidence))
    metrics = list(next(iter(per_q.values())))
    summary = {}
    for m in metrics:
        vals = [per_q[q][m] for q in per_q]
        lo, hi = bootstrap_ci(vals)
        summary[m] = {"mean": sum(vals) / len(vals), "ci95": [lo, hi]}
    by_type = {}
    for qtype in sorted({it.qtype for it in items}):
        qids = [it.qid for it in items if it.qtype == qtype]
        by_type[qtype] = {"n": len(qids)} | {
            m: sum(per_q[q][m] for q in qids) / len(qids) for m in ("recall@5", "mrr@10", "ndcg@10")
        }
    return {"summary": summary, "by_type": by_type, "per_question": per_q}


def paired_difference(a: dict, b: dict, metric: str) -> dict:
    """Mean and bootstrap CI of (b - a) per question: same questions, so pair them."""
    diffs = [b["per_question"][q][metric] - a["per_question"][q][metric] for q in a["per_question"]]
    lo, hi = bootstrap_ci(diffs)
    return {"mean": sum(diffs) / len(diffs), "ci95": [lo, hi]}


def question_overlap(items: list[EvalItem], chunks: list[Chunk], judge: Judge) -> dict[str, float]:
    """Share of a question's content terms present in its best gold chunk. Questions
    written by someone reading the passage share its vocabulary, which favours BM25;
    this measures how much, so results can be read per overlap band."""
    out = {}
    for it in items:
        q = set(tokenize(it.question))
        gold = [c for c in chunks if judge.covered(it.qid, c)]
        out[it.qid] = max(len(q & set(tokenize(c.text))) / len(q) for c in gold)
    return out


def overlap_breakdown(results: dict, overlap: dict[str, float], configs: list[str]) -> list[dict]:
    """recall@5 / MRR@10 per overlap tertile for the given configs."""
    qs = sorted(overlap, key=lambda q: (overlap[q], q))
    n = len(qs)
    rows = []
    for name, band in zip(("low", "mid", "high"), (qs[: n // 3], qs[n // 3 : 2 * n // 3], qs[2 * n // 3 :])):
        row = {"band": name, "n": len(band), "overlap_range": [overlap[band[0]], overlap[band[-1]]]}
        for cfg in configs:
            pq = results["configs"][cfg]["per_question"]
            row[cfg] = {m: sum(pq[q][m] for q in band) / len(band) for m in ("recall@5", "mrr@10")}
        rows.append(row)
    return rows


def run_all(items: list[EvalItem]) -> dict:
    embedder = Embedder()
    judge = Judge(items)
    results: dict = {"n_questions": len(items), "configs": {}}
    for size, path in CHUNKINGS.items():
        broken = judge.unresolved(items, load_chunks(path))
        if broken:
            raise ValueError(f"{len(broken)} evidence units match no chunk in {path}: {broken[:10]}")
        for mode in MODES:
            name = f"{mode}-{size}"
            res = evaluate_retriever(build_retriever(mode, path, embedder=embedder), items, judge)
            results["configs"][name] = res
            s = res["summary"]
            print(f"{name:12s} R@5={s['recall@5']['mean']:.3f}  MRR={s['mrr@10']['mean']:.3f}  "
                  f"nDCG@10={s['ndcg@10']['mean']:.3f}")
    cfg = results["configs"]
    results["paired_differences"] = {
        f"{b} vs {a}": {m: paired_difference(cfg[a], cfg[b], m) for m in ("recall@5", "mrr@10", "ndcg@10")}
        for a, b in [("dense-500", "hybrid-500"), ("dense-250", "hybrid-250"),
                     ("dense-500", "dense-250"), ("hybrid-500", "hybrid-250"), ("bm25-250", "hybrid-250")]
    }
    overlap = question_overlap(items, load_chunks(CHUNKINGS["250"]), judge)
    results["question_overlap"] = overlap
    results["overlap_breakdown"] = overlap_breakdown(results, overlap, ["dense-250", "bm25-250", "hybrid-250"])
    return results


def to_markdown(results: dict) -> str:
    cols = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@5", "ndcg@10"]
    lines = [f"Retrieval results on {results['n_questions']} labelled questions "
             "(mean, 95% bootstrap CI over questions in brackets).", "",
             "| Config | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, res in results["configs"].items():
        s = res["summary"]
        cells = [f"{s[c]['mean']:.3f} [{s[c]['ci95'][0]:.2f}, {s[c]['ci95'][1]:.2f}]" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Paired differences (row = B vs A, value = B − A):", "",
              "| Comparison | recall@5 | mrr@10 | ndcg@10 |", "|---|---|---|---|"]
    for comp, d in results["paired_differences"].items():
        cells = [f"{d[m]['mean']:+.3f} [{d[m]['ci95'][0]:+.2f}, {d[m]['ci95'][1]:+.2f}]"
                 for m in ("recall@5", "mrr@10", "ndcg@10")]
        lines.append(f"| {comp} | " + " | ".join(cells) + " |")
    cfgs = ["dense-250", "bm25-250", "hybrid-250"]
    lines += ["", "recall@5 / MRR@10 by lexical-overlap tertile (share of question terms found in the best "
              "gold chunk; low overlap is closest to a user paraphrasing):", "",
              "| Band | overlap | n | " + " | ".join(cfgs) + " |", "|---" * (len(cfgs) + 3) + "|"]
    for row in results["overlap_breakdown"]:
        lo, hi = row["overlap_range"]
        cells = [f"{row[c]['recall@5']:.3f} / {row[c]['mrr@10']:.3f}" for c in cfgs]
        lines.append(f"| {row['band']} | {lo:.2f}–{hi:.2f} | {row['n']} | " + " | ".join(cells) + " |")
    types = list(next(iter(results["configs"].values()))["by_type"])
    lines += ["", "recall@5 / MRR@10 by question type (named = question names the method or paper; "
              "descriptive = paraphrased, no name; multi = evidence from two papers):", "",
              "| Config | " + " | ".join(f"{t} (n={results['configs']['dense-500']['by_type'][t]['n']})"
                                       for t in types) + " |", "|---" * (len(types) + 1) + "|"]
    for name, res in results["configs"].items():
        bt = res["by_type"]
        cells = [f"{bt[t]['recall@5']:.3f} / {bt[t]['mrr@10']:.3f}" for t in types]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def show_misses(config: str, items: list[EvalItem], k: int = 5) -> None:
    mode, size = config.split("-")
    retriever = build_retriever(mode, CHUNKINGS[size])
    judge = Judge(items)
    for it in items:
        hits = retriever.search(it.question, k=k)
        ranked = [judge.covered(it.qid, h.chunk) for h in hits]
        if recall_at_k(ranked, len(it.evidence), k) < 1:
            print(f"\n[{it.qid}] {it.question}\n  want: {[u.arxiv_id for u in it.evidence]}")
            for h, units in zip(hits, ranked):
                print(f"  {h.rank}. {'*' if units else ' '} {h.chunk.chunk_id}  {h.chunk.text[:90]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET_PATH)
    parser.add_argument("--misses", metavar="CONFIG", help="e.g. hybrid-250: print failures at k=5 instead")
    args = parser.parse_args()
    eval_items = load_eval_set(args.eval_set)
    if args.misses:
        show_misses(args.misses, eval_items)
    else:
        out = run_all(eval_items)
        (EVAL_DIR / "results.json").write_text(json.dumps(out, indent=2))
        (EVAL_DIR / "results.md").write_text(to_markdown(out))
        print(f"\nWrote {EVAL_DIR / 'results.json'} and {EVAL_DIR / 'results.md'}")
