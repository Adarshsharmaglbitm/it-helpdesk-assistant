"""Evaluate retrieval quality (and optionally answer quality) on a labeled question set.

Usage:
    python -m eval.evaluate            # retrieval metrics for dense, BM25 and hybrid
    python -m eval.evaluate --llm      # also run the LLM: citation and refusal metrics

Retrieval metrics (questions that have an expected source):
    Hit@k  - share of questions where the correct article is in the top-k results
    MRR@5  - mean reciprocal rank of the correct article (1.0 = always ranked first)

LLM metrics:
    citation rate      - in-scope answers that cite at least one passage
    source accuracy    - in-scope answers that cite the correct article
    refusal accuracy   - out-of-scope questions where the bot declines instead of guessing
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from src import config
from src.metrics import summarize
from src.retrieval import MODES, HybridRetriever

EVAL_DIR = Path(__file__).resolve().parent
KS = (1, 3, 5)


def load_questions() -> list[dict]:
    return json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))


def evaluate_retrieval(retriever: HybridRetriever, questions: list[dict]) -> dict:
    in_scope = [q for q in questions if q["expected_source"]]
    results = {}
    for mode in MODES:
        per_question, latencies = [], []
        for q in in_scope:
            start = time.perf_counter()
            # Retrieve extra chunks so we can score the top-5 *distinct* articles.
            docs = retriever.retrieve(q["question"], k=max(KS) * 3, mode=mode)
            latencies.append((time.perf_counter() - start) * 1000)
            per_question.append(
                {"retrieved": [d.metadata["source"] for d in docs], "expected": q["expected_source"]}
            )
        results[mode] = summarize(per_question, KS)
        results[mode]["latency_ms"] = statistics.median(latencies)
    return results


def is_refusal(answer: str) -> bool:
    text = answer.lower().replace("’", "'")
    return "couldn't find this" in text or "could not find this" in text


def evaluate_llm(retriever: HybridRetriever, questions: list[dict]) -> dict:
    from src.rag import RAGPipeline

    pipeline = RAGPipeline(retriever=retriever)
    if pipeline.llm is None:
        raise SystemExit("LLM_PROVIDER is 'none'; set a provider in .env to run --llm.")

    cited_any = cited_right = in_scope = refused = out_scope = 0
    latencies, rows = [], []
    for q in questions:
        start = time.perf_counter()
        result = pipeline.answer(q["question"])
        latencies.append(time.perf_counter() - start)
        cited_sources = {d.metadata["source"] for d in result["cited"]}
        if q["expected_source"]:
            in_scope += 1
            cited_any += bool(cited_sources) and not is_refusal(result["answer"])
            cited_right += q["expected_source"] in cited_sources
        else:
            out_scope += 1
            refused += is_refusal(result["answer"])
        rows.append({**q, "answer": result["answer"], "cited": sorted(cited_sources)})
        print(".", end="", flush=True)
    print()

    (EVAL_DIR / "llm_answers.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {
        "citation_rate": cited_any / in_scope if in_scope else 0.0,
        "source_accuracy": cited_right / in_scope if in_scope else 0.0,
        "refusal_accuracy": refused / out_scope if out_scope else 0.0,
        "median_latency_s": statistics.median(latencies),
    }


def print_table(retrieval: dict) -> None:
    metrics = list(next(iter(retrieval.values())).keys())
    print("\n| Mode | " + " | ".join(metrics) + " |")
    print("|---" * (len(metrics) + 1) + "|")
    for mode, vals in retrieval.items():
        cells = [f"{vals[m]:.1f}" if m == "latency_ms" else f"{vals[m]:.3f}" for m in metrics]
        print(f"| {mode} | " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="also evaluate generated answers")
    args = parser.parse_args()

    questions = load_questions()
    n_in = sum(1 for q in questions if q["expected_source"])
    print(f"{len(questions)} questions ({n_in} in-scope, {len(questions) - n_in} out-of-scope)")
    print(f"Embeddings: {config.EMBEDDING_MODEL}")

    retriever = HybridRetriever()
    report = {"retrieval": evaluate_retrieval(retriever, questions)}
    print_table(report["retrieval"])

    if args.llm:
        print(f"\nRunning LLM ({config.LLM_PROVIDER}) on all questions", end="")
        report["llm"] = evaluate_llm(retriever, questions)
        print(json.dumps(report["llm"], indent=2))

    (EVAL_DIR / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved results to {EVAL_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
