"""Pure helper functions with no heavy dependencies, so they are easy to unit test."""

import re
from collections import defaultdict
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "the", "to", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your", "me", "should",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics and drop stopwords (used for BM25)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Merge several ranked lists of ids with Reciprocal Rank Fusion.

    score(id) = sum_i weight_i / (k + rank_i(id)), with ranks starting at 1.
    RRF only uses ranks, so it can combine retrievers whose raw scores are on
    different scales (cosine similarity vs. BM25).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have the same length as ranked_lists")

    scores: dict[str, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def format_context(docs: Iterable) -> str:
    """Render retrieved chunks as numbered passages the LLM can cite as [1], [2], ..."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        label = meta.get("title", meta.get("source", "unknown"))
        if meta.get("section"):
            label += f" > {meta['section']}"
        blocks.append(f"[{i}] ({label})\n{meta.get('text', doc.page_content)}")
    return "\n\n".join(blocks)


_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer: str, num_sources: int) -> list[int]:
    """Return the sorted, valid source numbers cited in an answer, e.g. "[1][3]" or "[1, 3]"."""
    cited = set()
    for group in _CITATION_RE.findall(answer):
        for n in group.split(","):
            n = int(n.strip())
            if 1 <= n <= num_sources:
                cited.add(n)
    return sorted(cited)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
