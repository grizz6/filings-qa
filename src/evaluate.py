"""Measure quality on the test set and gate CI on it.

Metrics, on eval/test_set.jsonl:
- hit_rate_at_k: share of the 40 answerable questions where a top-k chunk contains the gold
  quote (quote matching keeps the test set valid when chunk sizes change chunk IDs).
- mrr: mean of 1/rank of the first chunk containing the quote (0 if none in the top k).
- hit_rate_at_5: same as hit rate but always at 5, to compare runs with different k.
- refusal_rate: share of the 10 unanswerable questions answered "Not found in the filings."
  (needs the LLM; skipped with --no-llm).

With --answers, the 40 answerable questions are also answered by the LLM and scored:
- answer_rate: share answered instead of refused.
- citation_rate: share whose answer cites at least one retrieved passage.
- cited_quote_rate: share where a passage the answer cites contains the gold quote: the answer
  rests on the passage that holds the fact. The closest automatic check of a correct answer.

Retrieval runs in memory with the same embedding model and company filter as the app, over
chunks rebuilt from data/sections/ with each experiment's chunk size. That makes chunk-size
experiments cheap and keeps them away from the live Supabase index.

An experiment may also set embedding_model, rerank_model and rerank_candidates; anything it
leaves out comes from config.yaml.

Run:
  python -m src.evaluate --all            # every experiment in eval/experiments.yaml -> MLflow
  python -m src.evaluate --experiment B C # some experiments
  python -m src.evaluate --all --answers  # also answer and score the 40 answerable questions
  python -m src.evaluate --gate           # config.yaml settings vs eval/thresholds.yaml, no LLM
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from src.chunk import chunk_filing
from src.config import ROOT, load_config
from src.retrieve import retrieve
from src.testset import TEST_SET_PATH, contains_quote, load_jsonl

EXPERIMENTS_PATH = ROOT / "eval" / "experiments.yaml"
THRESHOLDS_PATH = ROOT / "eval" / "thresholds.yaml"
RESULTS_DIR = ROOT / "data" / "eval"
SECTIONS_DIR = ROOT / "data" / "sections"
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.json"
GATED_METRICS = ("hit_rate_at_k", "mrr")
ANSWER_METRICS = ("answer_rate", "citation_rate", "cited_quote_rate")


def build_chunks(size: int, overlap: int, sections_dir=SECTIONS_DIR, manifest=MANIFEST_PATH):
    filings = json.loads(Path(manifest).read_text())
    chunks = []
    for filing in filings:
        section = (Path(sections_dir) / f"{filing['ticker']}.txt").read_text()
        chunks.extend(chunk_filing(section, filing, size, overlap))
    return chunks


class MemoryIndex:
    """Exact cosine search over normalized vectors, with the same signature as index.query."""

    def __init__(self, chunks: list[dict], vectors, embed_fn):
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.tickers = np.array([c["ticker"] for c in chunks])
        self.embed_fn = embed_fn

    def search(self, question: str, index_name: str, tickers: list[str] | None, k: int):
        scores = self.vectors @ np.asarray(self.embed_fn(question), dtype=np.float32)
        if tickers:
            scores = np.where(np.isin(self.tickers, tickers), scores, -np.inf)
        order = [i for i in np.argsort(-scores)[:k] if np.isfinite(scores[i])]
        return [
            (
                self.chunks[i]["id"],
                self.chunks[i]["ticker"],
                self.chunks[i]["text"],
                self.chunks[i]["source_url"],
                float(scores[i]),
            )
            for i in order
        ]


def first_hit_rank(retrieved: list[dict], quote: str) -> int | None:
    """1-based rank of the first retrieved chunk that contains the quote, or None."""
    for rank, chunk in enumerate(retrieved, 1):
        if contains_quote(chunk["text"], quote):
            return rank
    return None


def retrieval_metrics(ranks: list[int | None], k: int) -> dict[str, float]:
    n = len(ranks)
    return {
        "hit_rate_at_k": sum(r is not None and r <= k for r in ranks) / n,
        "hit_rate_at_5": sum(r is not None and r <= 5 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r is not None and r <= k) / n,
    }


def cites_quote(result: dict, quote: str) -> bool:
    """True when a passage the answer cites contains the quote."""
    texts = {c["id"]: c["text"] for c in result["chunks"]}
    return any(contains_quote(texts.get(c["id"], ""), quote) for c in result["citations"])


def answer_metrics(scored: list[dict]) -> dict[str, float]:
    n = len(scored)
    return {
        "answer_rate": sum(not r["refused"] for r in scored) / n,
        "citation_rate": sum(r["n_citations"] > 0 for r in scored) / n,
        "cited_quote_rate": sum(r["cited_quote"] for r in scored) / n,
    }


def evaluate(
    items: list[dict], cfg: dict, search_fn, k: int, answer_fn=None, score_answers=False
) -> tuple[dict, list[dict]]:
    """Score retrieval on answerable items and, with answer_fn, refusals on the others.

    With score_answers (needs answer_fn), the answerable items are answered and scored too.
    """
    rows, ranks, scored = [], [], []
    for it in items:
        if not it["answerable"]:
            continue
        retrieved = retrieve(it["question"], k=k, cfg=cfg, search_fn=search_fn)
        rank = first_hit_rank(retrieved, it["quote"])
        ranks.append(rank)
        row = {"id": it["id"], "rank": rank, "retrieved": [c["id"] for c in retrieved]}
        if score_answers and answer_fn is not None:
            result = answer_fn(it["question"])
            row.update(
                refused=result["refused"],
                n_citations=len(result["citations"]),
                cited_quote=cites_quote(result, it["quote"]),
                answer=result["answer"],
            )
            scored.append(row)
        rows.append(row)
    metrics = retrieval_metrics(ranks, k)
    if scored:
        metrics.update(answer_metrics(scored))
    if answer_fn is not None:
        refused = []
        for it in items:
            if it["answerable"]:
                continue
            result = answer_fn(it["question"])
            refused.append(result["refused"])
            rows.append({"id": it["id"], "refused": result["refused"], "answer": result["answer"]})
        if refused:
            metrics["refusal_rate"] = sum(refused) / len(refused)
    return metrics, rows


RETRIEVAL_OVERRIDES = ("embedding_model", "rerank_model", "rerank_candidates")


def experiment_config(params: dict, cfg: dict) -> dict:
    """cfg with the experiment's retrieval settings (embedding model, reranker) applied."""
    retrieval = {**cfg["retrieval"], **{p: params[p] for p in RETRIEVAL_OVERRIDES if p in params}}
    return {**cfg, "retrieval": retrieval}


def run_experiment(
    name: str, params: dict, cfg: dict, use_llm: bool, score_answers: bool = False
) -> tuple[dict, list[dict]]:
    from src.index import embed, load_embedder

    cfg = experiment_config(params, cfg)
    model = load_embedder(cfg["retrieval"]["embedding_model"])
    chunks = build_chunks(params["size_words"], params["overlap_words"])
    vectors = embed(model, [c["text"] for c in chunks])
    index = MemoryIndex(chunks, vectors, lambda q: embed(model, [q])[0])
    k = params["top_k"]

    answer_fn = None
    if use_llm:
        from src.answer import answer

        def answer_fn(q):
            return answer(
                q,
                cfg,
                retrieve_fn=lambda q2, cfg: retrieve(q2, k=k, cfg=cfg, search_fn=index.search),
            )

    metrics, rows = evaluate(
        load_jsonl(TEST_SET_PATH), cfg, index.search, k, answer_fn, score_answers=score_answers
    )
    metrics["n_chunks"] = len(chunks)
    return metrics, rows


def log_to_mlflow(name: str, params: dict, cfg: dict, metrics: dict, rows_path: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlflow.db"))
    mlflow.set_experiment("filings-qa-retrieval")
    retrieval = experiment_config(params, cfg)["retrieval"]
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                **params,
                "embedding_model": retrieval["embedding_model"],
                "rerank_model": retrieval.get("rerank_model") or "none",
                "llm_model": cfg["llm"]["model"],
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(rows_path))


def markdown_table(results: dict[str, dict]) -> str:
    cols = ["hit_rate_at_k", "hit_rate_at_5", "mrr", "refusal_rate"]
    cols += [c for c in ANSWER_METRICS if any(c in m for m in results.values())]
    cols.append("n_chunks")
    lines = ["| run | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in results.items():
        cells = [
            "" if c not in m else (str(m[c]) if c == "n_chunks" else f"{m[c]:.3f}") for c in cols
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gate_failures(metrics: dict, thresholds: dict) -> list[str]:
    return [
        f"{name} {metrics[name]:.3f} is below the threshold {thresholds[name]:.3f}"
        for name in GATED_METRICS
        if name in thresholds and metrics[name] < thresholds[name]
    ]


def _write_summary(text: str) -> None:
    print(text)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--experiment", nargs="+", metavar="NAME")
    g.add_argument("--gate", action="store_true")
    p.add_argument("--no-llm", action="store_true", help="skip the refusal metric")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument(
        "--answers", action="store_true", help="also answer and score the answerable questions"
    )
    args = p.parse_args()
    cfg = load_config()

    if args.gate:
        params = {
            "size_words": cfg["chunking"]["size_words"],
            "overlap_words": cfg["chunking"]["overlap_words"],
            "top_k": cfg["retrieval"]["top_k"],
        }
        metrics, _ = run_experiment("gate", params, cfg, use_llm=False)
        thresholds = yaml.safe_load(THRESHOLDS_PATH.read_text())
        _write_summary(
            f"### Eval gate (config.yaml: {params})\n\n" + markdown_table({"current": metrics})
        )
        _write_summary(f"\nThresholds: {thresholds}")
        failures = gate_failures(metrics, thresholds)
        if failures:
            print("\n".join(f"GATE FAILED: {f}" for f in failures), file=sys.stderr)
            sys.exit(1)
        print("Gate passed.")
        return

    experiments = yaml.safe_load(EXPERIMENTS_PATH.read_text())["experiments"]
    names = list(experiments) if args.all else args.experiment
    if unknown := [n for n in names if n not in experiments]:
        sys.exit(f"Unknown experiment(s): {unknown}. Known: {list(experiments)}")
    if args.answers and args.no_llm:
        sys.exit("--answers needs the LLM; drop --no-llm")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        params = experiments[name]
        metrics, rows = run_experiment(
            name, params, cfg, use_llm=not args.no_llm, score_answers=args.answers
        )
        rows_path = RESULTS_DIR / f"{name}.json"
        rows_path.write_text(
            json.dumps({"params": params, "metrics": metrics, "rows": rows}, indent=2)
        )
        if not args.no_mlflow:
            log_to_mlflow(name, params, cfg, metrics, rows_path)
        results[name] = metrics
        print(f"{name}: {metrics}", flush=True)
    _write_summary("### Experiments\n\n" + markdown_table(results))


if __name__ == "__main__":
    main()
