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

DEFAULT_BOT_NAME = "IT Helpdesk Assistant"
DEFAULT_FOUNDER = "Adarsh Sharma"

_GREETING_RE = re.compile(
    r"^(hi+|hello+|hey+|hiya|yo|howdy|namaste|hola|good (morning|afternoon|evening|day)|greetings)"
    r"(\s+(there|team|bot|assistant|everyone|all|helpdesk))?$",
)
_HOW_ARE_YOU_RE = re.compile(
    r"^((hi+|hey+|hello+)[\s,]+)?(how are you|how r u|how are u|how'?s it going|how are things|how do you do|"
    r"what'?s up|wh?assup|sup)(\s+(doing|today|there))*$"
)
# Whole-message thanks only: "ok but my VPN still fails" must reach the knowledge base, not get "You're welcome".
_THANKS_WORDS = (r"(thanks?|thank\s*you|thankyou|thx|ty|cheers|great|perfect|awesome|ok(ay)?|cool|nice|got it|"
                 r"understood|that (worked|helped|fixed it)|it (works|worked)( now)?|(all )?(sorted|solved|fixed))")
_THANKS_TAIL = r"(\s+(so much|very much|a lot|a ton|again|man|mate|buddy|bro|team|for (the|your) help|now))*"
_THANKS_RE = re.compile(rf"^{_THANKS_WORDS}{_THANKS_TAIL}([\s,]+{_THANKS_WORDS}{_THANKS_TAIL})?$")
_BYE_RE = re.compile(
    r"^(bye+|bye bye|goodbye|good night|see (you|ya)( later| soon)?|cya|later|"
    r"that'?s all( for (now|today))?|nothing else|no(pe)?[\s,]+(thanks|thank you|that'?s all))$"
)
# "who made you?", "who is your founder", "who built this app", "founder of this bot"
_ABOUT_YOU = r"(\byou\b|\bu\b|\byourself\b|\b(this|the)\s+(app|bot|assistant|chatbot|tool|website|site|project|helpdesk)\b)"
_CREATOR_RE = re.compile(
    rf"\bwho\b.*\b(made|make|created|create|built|build|developed|develop|founded|designed|programmed|owns|wrote|"
    rf"invented|trained)\b.*{_ABOUT_YOU}"
    r"|\b(your|ur)\s+(founder|co-?founder|creator|developer|maker|owner|author|father|ceo|boss)\b"
    rf"|\b(founder|co-?founder|creator|developer|maker|owner|author)\s+(of|behind)\s+{_ABOUT_YOU}"
    r"|^(who'?s|who is|who are)?\s*(the\s+)?(founder|creator|developer|owner|maker)s?$"
)
_IDENTITY_RE = re.compile(
    r"^(who|what)\s+(are|r)\s+(you|u)$|\byour\s+name\b|^introduce\s+yourself$|^tell me about (yourself|you)$"
    r"|^are you (a )?(bot|human|robot|ai|real)$"
)
_CAPABILITIES_RE = re.compile(r"^(help|menu|options|what can (you|u) do|what do you do|how can (you|u) help( me)?|"
                              r"what can i ask( you)?|how does this work)$")


def _clean(text: str) -> str:
    """Lowercase, straighten quotes and drop trailing punctuation/emoji, so 'Thanks!! 🙏' matches like 'thanks'."""
    text = text.strip().lower().replace("’", "'")
    text = re.sub(r"[\s!.,?;:~)(\-🙏👍😊🙂😀😃❤️♥]+$", "", text)
    return re.sub(r"^[\s!.,?;:]+", "", text)


def small_talk_reply(text: str, name: str = "", founder: str = DEFAULT_FOUNDER,
                     bot: str = DEFAULT_BOT_NAME, model: str = "") -> str | None:
    """Return a friendly canned reply for greetings, thanks, goodbyes and questions about the assistant itself
    (who it is, who built it, what it can do), or None for a real IT question."""
    if len(text) > 90:
        return None
    t = _clean(text)
    first = name.split()[0] if name.strip() else ""
    hi = f"Hi {first}! 👋" if first else "Hi! 👋"
    powered = f" Under the hood I run on the **{model}** language model." if model else ""

    if _CREATOR_RE.search(t):
        return (
            f"I was created by **{founder}**, the founder of the {bot}. 🛠️ He built me so employees get fast, "
            f"step-by-step IT answers from the company knowledge base, with a source for every step, and a "
            f"pre-filled ticket when something needs a human.{powered}\n\nWhat can I help you fix today?"
        )
    if _IDENTITY_RE.search(t):
        return (
            f"I'm the **{bot}**, created by **{founder}**. I answer IT questions from the company knowledge base "
            f"(VPN, passwords, MFA, Outlook, printers, BitLocker, phishing and more) and cite the article behind "
            f"every step.{powered}\n\nTell me what's going on and I'll walk you through it."
        )
    if _CAPABILITIES_RE.match(t):
        return (
            "Here's what I can do for you:\n\n"
            "- **Fix common IT issues** with cited, step-by-step answers (VPN, passwords, MFA, Outlook, Wi-Fi, "
            "printers, BitLocker, Teams, slow laptops, software requests)\n"
            "- **Read what you attach**: a screenshot of an error, a log file or a PDF, or a voice note\n"
            "- **Remember the conversation**, so you can ask follow-ups like *\"what if that fails?\"*\n"
            "- **Raise a ticket** pre-filled from our chat when you need a person\n\n"
            "Pick an answer style below the chat (⚡ Quick, 🧭 Step-by-step or 📖 Detailed) to match how much "
            "detail you want."
        )
    if _GREETING_RE.match(t):
        return (
            f"{hi} I'm the {bot}. Describe your issue, for example *\"my VPN won't connect\"* or "
            "*\"how do I set up MFA on a new phone?\"*, and I'll find the answer in the IT knowledge base."
        )
    if _HOW_ARE_YOU_RE.match(t):
        return (f"I'm doing great, thanks for asking{', ' + first if first else ''}! 😊 Ready when you are. "
                "What IT issue can I help with?")
    if _THANKS_RE.match(t):
        return f"You're welcome{', ' + first if first else ''}! 🙌 Anything else I can help you with?"
    if _BYE_RE.match(t):
        return (f"Glad I could help{', ' + first if first else ''}. If the issue comes back, you can always "
                "raise a ticket from the **Tickets** tab. 👋")
    return None


# --- answer styles -----------------------------------------------------------------

ANSWER_STYLES = {
    "Quick": (
        "⚡",
        "Answer in 2 to 4 short bullet points or sentences with only the essential steps. No background.",
    ),
    "Step-by-step": (
        "🧭",
        "Start with one short sentence saying what is going on or what you'll do together. Then give numbered "
        "steps, one action per step, with exact menu names and buttons in **bold**. If the passages give a next "
        "step when this doesn't work (often raising a ticket), end with a short line starting 'If that doesn't work'.",
    ),
    "Detailed": (
        "📖",
        "Give a thorough answer: first briefly explain the likely cause, then numbered steps with exact menu names "
        "and buttons in **bold**, then useful tips or common pitfalls from the passages, and finally when to raise "
        "a ticket (and its priority, if the passages say).",
    ),
}
DEFAULT_STYLE = "Step-by-step"


def style_instructions(style: str) -> str:
    return ANSWER_STYLES.get(style, ANSWER_STYLES[DEFAULT_STYLE])[1]


def group_by_day(chats: Sequence[dict], today) -> list[tuple[str, list[dict]]]:
    """Split chats (newest first, with an ISO 'updated' time) into Today / Yesterday / Previous 7 days / Older."""
    from datetime import datetime

    groups: dict[str, list[dict]] = {"Today": [], "Yesterday": [], "Previous 7 days": [], "Older": []}
    for chat in chats:
        try:
            day = datetime.fromisoformat(chat.get("updated", "")).astimezone().date()
        except ValueError:
            day = today
        age = (today - day).days
        name = "Today" if age <= 0 else "Yesterday" if age == 1 else "Previous 7 days" if age < 7 else "Older"
        groups[name].append(chat)
    return [(name, items) for name, items in groups.items() if items]


_MD_SPECIAL_RE = re.compile(r"([\\`*_{}\[\]()#+\-.!|~<>:$])")


def escape_markdown(text: str) -> str:
    """Show text literally in a Streamlit label (button labels are Markdown)."""
    return _MD_SPECIAL_RE.sub(r"\\\1", text)


def history_messages(messages: Sequence[dict], max_turns: int = 3, max_chars: int = 700) -> list[tuple[str, str]]:
    """The last few turns as (role, text) pairs for the answering prompt. Citations are removed because the
    passage numbers change every turn and would point the model at the wrong source."""
    turns = [m for m in messages if m.get("role") in ("user", "assistant")][-max_turns * 2:]
    out = []
    for m in turns:
        content = _CITATION_RE.sub("", m.get("content", "")).strip()
        if content:
            out.append(("human" if m["role"] == "user" else "ai", content[:max_chars]))
    return out


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
