"""Unit tests for the dependency-free helpers. Run with: pytest"""

from types import SimpleNamespace

import pytest

from src.metrics import hit_at_k, reciprocal_rank, summarize
from src.utils import extract_citations, format_context, reciprocal_rank_fusion, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("How do I reset my VPN password?") == ["reset", "vpn", "password"]


def test_rrf_rewards_items_ranked_well_by_both_lists():
    dense = ["a", "b", "c"]
    sparse = ["b", "c", "a"]
    fused = [item for item, _ in reciprocal_rank_fusion([dense, sparse], k=60)]
    assert fused[0] == "b"  # ranks 2 and 1 beat ranks 1 and 3
    assert set(fused) == {"a", "b", "c"}


def test_rrf_includes_items_from_only_one_list():
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 61)


def test_rrf_weights_must_match():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_extract_citations_handles_formats_and_invalid_numbers():
    answer = "Restart the router [1]. Then reset your password [2][3] or see [2, 4]. Ignore [9]."
    assert extract_citations(answer, num_sources=4) == [1, 2, 3, 4]
    assert extract_citations("No citations here.", num_sources=4) == []


def test_format_context_numbers_passages():
    docs = [
        SimpleNamespace(page_content="x", metadata={"title": "VPN", "section": "Errors", "text": "Restart."}),
        SimpleNamespace(page_content="y", metadata={"title": "MFA", "section": "", "text": "Install app."}),
    ]
    ctx = format_context(docs)
    assert ctx.startswith("[1] (VPN > Errors)\nRestart.")
    assert "[2] (MFA)\nInstall app." in ctx


def test_hit_and_reciprocal_rank_use_distinct_documents():
    retrieved = ["a.md", "a.md", "a.md", "b.md"]  # three chunks from the same article
    assert hit_at_k(retrieved, "b.md", k=2) == 1.0
    assert reciprocal_rank(retrieved, "b.md", k=5) == 0.5
    assert reciprocal_rank(retrieved, "c.md", k=5) == 0.0


def test_summarize():
    rows = [
        {"retrieved": ["a", "b"], "expected": "a"},
        {"retrieved": ["a", "b"], "expected": "b"},
    ]
    s = summarize(rows, ks=(1, 3))
    assert s["hit@1"] == 0.5
    assert s["hit@3"] == 1.0
    assert s["mrr@3"] == pytest.approx(0.75)


# --- conversation helpers ------------------------------------------------------

from src.utils import format_history, guess_priority, small_talk_reply


@pytest.mark.parametrize("text", ["hi", "Hello!", "hey there", "good morning", "thanks", "Thank you so much", "bye"])
def test_small_talk_is_detected(text):
    assert small_talk_reply(text)


@pytest.mark.parametrize("text", ["hi, my VPN is broken", "how do I reset my password?", "outlook keeps crashing"])
def test_real_questions_are_not_small_talk(text):
    assert small_talk_reply(text) is None


def test_format_history_keeps_last_turns_and_strips_citations():
    msgs = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    msgs.append({"role": "assistant", "content": "Reset it in the portal [1][2]."})
    text = format_history(msgs, max_turns=1)
    assert text == "User: q9\nAssistant: Reset it in the portal ."


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I clicked a phishing link", "P1 Critical"),
        ("My whole team cannot access SAP", "P2 High"),
        ("Please install Visio", "P4 Low"),
        ("My printer is jammed", "P3 Medium"),
    ],
)
def test_guess_priority(text, expected):
    assert guess_priority(text) == expected


def test_normalize_citations_handles_fullwidth_brackets():
    from src.utils import normalize_citations

    text = normalize_citations("Call the desk【1】【2】 or see 【3†source】.")
    assert text == "Call the desk[1][2] or see [3]."
    assert extract_citations(text, 3) == [1, 2, 3]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("what if that still doesn't work?", True),
        ("and on my phone?", True),
        ("Outlook too", True),
        ("How do I get my BitLocker recovery key from the portal?", False),
        ("My printer on floor 3 shows an offline status", False),
    ],
)
def test_looks_like_followup(text, expected):
    from src.utils import looks_like_followup

    assert looks_like_followup(text) is expected


# --- chat attachments ------------------------------------------------------------

from src import media


@pytest.mark.parametrize("name, kind", [("shot.PNG", "image"), ("err.jpeg", "image"), ("vpn.log", "text"),
                                        ("report.pdf", "pdf"), ("noext", "text")])
def test_attachment_kind(name, kind):
    assert media.kind_of(name) == kind


def test_extract_text_truncates_long_files():
    text = media.extract_text("big.log", ("x" * 50).encode(), limit=10)
    assert text == "x" * 10 + "\n…(truncated)"


def test_prompt_and_search_include_attachment_content():
    atts = [{"name": "s.png", "kind": "image", "summary": "Error 0x80090308 in GlobalProtect"},
            {"name": "vpn.log", "kind": "text", "summary": "TLS handshake failed"}]
    prompt = media.build_prompt("Why can't I connect?", atts)
    assert "Attached screenshot (s.png) shows:\nError 0x80090308" in prompt
    assert "Attached file vpn.log:\n```\nTLS handshake failed\n```" in prompt
    assert media.search_text("VPN down", atts) == "VPN down Error 0x80090308 in GlobalProtect TLS handshake failed"
