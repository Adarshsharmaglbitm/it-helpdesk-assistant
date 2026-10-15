"""Retrieval evaluation metrics. A question counts as a hit if a chunk from the
expected source document is retrieved."""

from typing import Sequence

from src.utils import dedupe_preserve_order


def hit_at_k(retrieved_sources: Sequence[str], expected: str, k: int) -> float:
    """1.0 if the expected document appears among the top-k distinct retrieved documents."""
    return 1.0 if expected in dedupe_preserve_order(retrieved_sources)[:k] else 0.0


def reciprocal_rank(retrieved_sources: Sequence[str], expected: str, k: int) -> float:
    """1/rank of the expected document among the top-k distinct documents, else 0."""
    for rank, source in enumerate(dedupe_preserve_order(retrieved_sources)[:k], start=1):
        if source == expected:
            return 1.0 / rank
    return 0.0


def summarize(per_question: list[dict], ks: Sequence[int] = (1, 3, 5)) -> dict:
    """Aggregate per-question retrieved-source lists into mean Hit@k and MRR@max(k)."""
    n = len(per_question)
    if n == 0:
        return {}
    max_k = max(ks)
    result = {
        f"hit@{k}": sum(hit_at_k(q["retrieved"], q["expected"], k) for q in per_question) / n
        for k in ks
    }
    result[f"mrr@{max_k}"] = (
        sum(reciprocal_rank(q["retrieved"], q["expected"], max_k) for q in per_question) / n
    )
    return result
