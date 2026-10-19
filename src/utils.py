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
_FULLWIDTH_CITATION_RE = re.compile(r"【\s*(\d+)(?:\s*[,†][^】]*)?\s*】")


def normalize_citations(answer: str) -> str:
    """Some models (e.g. gpt-oss) cite as 【1】 or 【1†source】; rewrite those to the [1] style we parse."""
    return _FULLWIDTH_CITATION_RE.sub(r"[\1]", answer)


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


# --- conversation helpers ------------------------------------------------------

_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|hiya|yo|good (morning|afternoon|evening)|greetings)\b[\s!.,?]*(there|team|bot)?[\s!.,?]*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(r"^\s*(thanks?|thank you|thx|ty|cheers|great|perfect|awesome|ok(ay)?|cool)\b[\w\s!.,]*$", re.IGNORECASE)
_BYE_RE = re.compile(r"^\s*(bye|goodbye|see you|cya|that'?s all)\b[\w\s!.,]*$", re.IGNORECASE)


def small_talk_reply(text: str) -> str | None:
    """Return a canned reply for greetings, thanks and goodbyes, or None for a real question."""
    if len(text) > 60:
        return None
    if _GREETING_RE.match(text):
        return (
            "Hi! 👋 I'm the IT Helpdesk Assistant. Describe your issue, for example "
            "*\"my VPN won't connect\"* or *\"how do I set up MFA on a new phone?\"*, "
            "and I'll find the answer in the IT knowledge base."
        )
    if _THANKS_RE.match(text):
        return "You're welcome! Anything else I can help you with?"
    if _BYE_RE.match(text):
        return "Glad I could help. If the issue comes back, you can always raise a ticket from the **Tickets** tab. 👋"
    return None


def format_history(messages: Sequence[dict], max_turns: int = 3) -> str:
    """Render the last few user/assistant turns as plain text for the follow-up rewriter."""
    turns = [m for m in messages if m.get("role") in ("user", "assistant")][-max_turns * 2:]
    lines = []
    for m in turns:
        content = _CITATION_RE.sub("", m["content"]).strip()
        lines.append(f"{m['role'].capitalize()}: {content[:600]}")
    return "\n".join(lines)


_PRIORITY_KEYWORDS = [
    ("P1 Critical", ("phishing", "hacked", "virus", "malware", "ransomware", "security", "breach",
                     "company-wide", "everyone", "whole company", "all users", "outage")),
    ("P2 High", ("team", "department", "whole floor", "cannot work", "can't work", "blocked")),
    ("P4 Low", ("request", "install", "new software", "license", "licence", "how do i", "question",
                "peripheral", "monitor", "keyboard", "mouse")),
]
PRIORITIES = ("P1 Critical", "P2 High", "P3 Medium", "P4 Low")


def guess_priority(text: str) -> str:
    """Suggest a ticket priority from the issue text, using the priority table in the SLA article."""
    lowered = text.lower()
    for priority, words in _PRIORITY_KEYWORDS:
        if any(w in lowered for w in words):
            return priority
    return "P3 Medium"


_FOLLOWUP_RE = re.compile(
    r"\b(it|its|that|this|those|these|they|them|there|still|again|also|else|instead|same|"
    r"didn'?t|doesn'?t|won'?t|isn'?t|other|another|next|more)\b|^\s*(and|but|what about|how about|ok|so)\b",
    re.IGNORECASE,
)


def looks_like_followup(question: str) -> bool:
    """Cheap check for questions that depend on earlier turns, so we only pay for an LLM rewrite when needed."""
    return len(question.split()) <= 4 or bool(_FOLLOWUP_RE.search(question))
