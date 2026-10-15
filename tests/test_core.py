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
