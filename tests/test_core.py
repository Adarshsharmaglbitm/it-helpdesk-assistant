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


# --- assistant identity, small talk and answer styles ------------------------------

from src.utils import ANSWER_STYLES, DEFAULT_STYLE, history_messages, style_instructions


@pytest.mark.parametrize("text", [
    "who is your founder?", "Who made you", "who created this app?", "who built you?", "founder of this bot",
    "Who's the founder", "who developed this assistant", "tell me about your creator",
])
def test_creator_questions_name_the_founder(text):
    reply = small_talk_reply(text, founder="Adarsh Sharma")
    assert reply and "Adarsh Sharma" in reply


@pytest.mark.parametrize("text", ["who are you?", "what's your name", "introduce yourself"])
def test_identity_questions_mention_founder(text):
    assert "Adarsh Sharma" in small_talk_reply(text)


def test_creator_reply_mentions_model_when_known():
    assert "groq / openai/gpt-oss-120b" in small_talk_reply("who made you", model="groq / openai/gpt-oss-120b")


@pytest.mark.parametrize("text", [
    "ok but my VPN still fails", "great, now outlook crashes", "thanks but it still doesn't work",
    "who created my account?", "who can reset my password", "help me reset my password",
])
def test_real_issues_are_not_swallowed_by_small_talk(text):
    assert small_talk_reply(text) is None


@pytest.mark.parametrize("text", ["Thanks!! 🙏", "ok thanks", "that worked, thank you", "perfect thanks a lot"])
def test_more_thanks_variants(text):
    assert small_talk_reply(text).startswith("You're welcome")


def test_greeting_uses_first_name():
    assert small_talk_reply("hello", name="Adarsh Sharma").startswith("Hi Adarsh!")


def test_capabilities_and_how_are_you():
    assert "Raise a ticket" in small_talk_reply("what can you do?")
    assert small_talk_reply("how are you").startswith("I'm doing great")


def test_answer_styles():
    assert DEFAULT_STYLE in ANSWER_STYLES
    assert style_instructions("Quick") != style_instructions("Detailed")
    assert style_instructions("nonsense") == style_instructions(DEFAULT_STYLE)


def test_history_messages_maps_roles_and_strips_citations():
    msgs = [{"role": "user", "content": "vpn fails"}, {"role": "assistant", "content": "Restart it [1][2]."}]
    assert history_messages(msgs) == [("human", "vpn fails"), ("ai", "Restart it .")]
    many = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    assert len(history_messages(many, max_turns=2)) == 4


# --- accounts and the ticket store -------------------------------------------------

from src import auth, config, store


@pytest.fixture
def tmp_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "s3cret-pass")
    return tmp_path


def test_register_and_sign_in(tmp_logs):
    user = auth.register("Priya Verma", "Priya.V", "hunter22", "hunter22")
    assert user == {"username": "priya.v", "name": "Priya Verma", "role": "user"}
    assert "hunter22" not in (tmp_logs / "users.json").read_text()  # only the hash is stored
    assert auth.authenticate("priya.v", "hunter22")["role"] == "user"
    with pytest.raises(auth.AuthError):
        auth.authenticate("priya.v", "wrong")
    with pytest.raises(auth.AuthError, match="admin access"):
        auth.authenticate("priya.v", "hunter22", "admin")


@pytest.mark.parametrize("name, username, password, confirm, problem", [
    ("", "abc", "hunter22", "hunter22", "name"),
    ("A", "a!", "hunter22", "hunter22", "Username"),
    ("A", "abc", "123", "123", "at least"),
    ("A", "abc", "hunter22", "hunter23", "match"),
    ("A", "admin", "hunter22", "hunter22", "taken"),
])
def test_register_validation(tmp_logs, name, username, password, confirm, problem):
    with pytest.raises(auth.AuthError, match=problem):
        auth.register(name, username, password, confirm)


def test_builtin_admin_and_promotion(tmp_logs):
    assert auth.authenticate("ADMIN", "s3cret-pass", "admin")["role"] == "admin"
    with pytest.raises(auth.AuthError):
        auth.authenticate("admin", "admin123", "admin")
    auth.register("Ravi", "ravi", "hunter22")
    auth.set_role("ravi", "admin")
    ravi = auth.authenticate("ravi", "hunter22", "admin")
    assert auth.still_valid(ravi)
    auth.set_role("ravi", "user")
    assert not auth.still_valid(ravi)  # demoted admins lose the admin console straight away
    auth.delete_user("ravi")
    assert not auth.still_valid({"username": "ravi", "name": "Ravi", "role": "user"})
    with pytest.raises(auth.AuthError):
        auth.delete_user("admin")


def test_remembered_sign_in_expires(tmp_logs):
    sid = "a" * 32
    store.save_auth(sid, {"username": "admin", "name": "Adarsh Sharma", "role": "admin"}, hours=1)
    assert store.load_auth(sid)["name"] == "Adarsh Sharma"
    store.save_auth(sid, {"username": "admin", "name": "Adarsh Sharma", "role": "admin"}, hours=-1)
    assert store.load_auth(sid) is None
    assert store.load_auth("../../etc/passwd") is None


def test_ticket_status_update(tmp_logs):
    store.append(store.TICKETS, {"id": "INC1", "status": "Open"})
    store.append(store.TICKETS, {"id": "INC2", "status": "Open"})
    assert store.update(store.TICKETS, {"id": "INC2"}, {"status": "Resolved"})
    assert [t["status"] for t in store.read(store.TICKETS)] == ["Open", "Resolved"]
    assert not store.update(store.TICKETS, {"id": "nope"}, {"status": "Closed"})


# --- website colors (Admin console → Website colors) ------------------------------------

from src import theme


def test_palette_has_light_and_dark_values():
    p = theme.palette("#059669")
    assert p["accent"].startswith("light-dark(#059669")
    assert p["on-accent"] == "#ffffff"
    assert theme.palette("#fde047")["on-accent"] == "#111827"  # light accents get dark text
    assert theme.palette("not-a-color") == theme.palette(theme.DEFAULTS["user"])
    assert theme.palette("#2563eb")["slider-filter"] == "hue-rotate(0deg) saturate(1.0)"


def test_color_math():
    assert theme.mix("#000000", "#ffffff", 0.5) == "#808080"
    assert theme.tint("#2563eb", 1) == "#ffffff" and theme.shade("#2563eb", 1) == "#000000"
    assert theme.alpha("#2563eb", 0.5) == "rgba(37,99,235,0.5)"
    assert theme.preset_name("#7C3AED") == "🟣 Royal violet" and theme.preset_name("#123456") == theme.CUSTOM


def test_colors_save_load_reset(tmp_logs):
    assert theme.load() == theme.DEFAULTS
    theme.save({"user": "#059669", "admin": "#E11D48", "junk": "x"}, "admin")
    assert theme.load() == {"user": "#059669", "admin": "#e11d48"}
    theme.save({"user": "nope"})
    assert theme.load()["user"] == theme.DEFAULTS["user"]  # invalid colors fall back to the default
    theme.reset()
    assert theme.load() == theme.DEFAULTS


# --- saved chats (left sidebar) --------------------------------------------------------

from datetime import date


def test_save_list_load_delete_chats(tmp_logs):
    a, b = "a" * 32, "b" * 32
    msgs = [{"role": "user", "content": "My   VPN says\nauthentication failed when I connect from home"},
            {"role": "assistant", "content": "Try this [1]."}]
    store.save_chat("Priya.V", a, msgs)
    store.save_chat("Priya.V", b, [{"role": "user", "content": "printer offline"}])
    chats = store.list_chats("priya.v")
    assert [c["id"] for c in chats] == [b, a] or {c["id"] for c in chats} == {a, b}
    assert next(c for c in chats if c["id"] == a)["title"].startswith("My VPN says authentication failed")
    assert store.load_chat("priya.v", a) == msgs
    assert store.list_chats("someone-else") == []
    store.delete_chat("priya.v", a)
    assert [c["id"] for c in store.list_chats("priya.v")] == [b]
    assert store.load_chat("priya.v", a) is None


def test_chat_ids_and_usernames_cannot_escape_the_logs_folder(tmp_logs):
    store.save_chat("../../evil", "../../etc/passwd", [{"role": "user", "content": "x"}])
    assert store.load_chat("x", "../index") is None
    assert all(p.resolve().is_relative_to(tmp_logs.resolve()) for p in tmp_logs.rglob("*"))


def test_chat_title():
    assert store.chat_title([]) == "New chat"
    assert store.chat_title([{"role": "user", "content": "x" * 100}]).endswith("…")
    assert store.chat_title([{"role": "user", "content": "", "attachments": [{"name": "err.png"}]}]) == "Attachment: err.png"


def test_group_by_day():
    from src.utils import group_by_day
    today = date(2026, 10, 20)
    chats = [{"id": "1", "updated": "2026-10-20T08:00:00+00:00"}, {"id": "2", "updated": "2026-10-19T08:00:00+00:00"},
             {"id": "3", "updated": "2026-10-15T08:00:00+00:00"}, {"id": "4", "updated": "2026-09-01T08:00:00+00:00"}]
    assert [(g, [c["id"] for c in items]) for g, items in group_by_day(chats, today)] == [
        ("Today", ["1"]), ("Yesterday", ["2"]), ("Previous 7 days", ["3"]), ("Older", ["4"])]


def test_escape_markdown():
    from src.utils import escape_markdown
    assert escape_markdown("**bold** :smile: [x](y)") == r"\*\*bold\*\* \:smile\: \[x\]\(y\)"
