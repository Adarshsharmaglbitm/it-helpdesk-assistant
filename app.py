"""Streamlit UI for the IT Helpdesk RAG assistant.

Run with:  streamlit run app.py
"""

import html
import re
import time
import uuid
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import config, media, store
from src.llm import describe
from src.retrieval import MODES
from src.utils import (
    PRIORITIES, extract_citations, format_history, guess_priority, looks_like_followup, normalize_citations,
    small_talk_reply,
)

st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="🛠️", layout="centered")

CHAT, KB, TICKETS, INSIGHTS = "💬 Chat", "📚 Knowledge base", "🎫 Tickets", "📊 Insights"
VIEWS = [CHAT, KB, TICKETS, INSIGHTS]
USER_AVATAR, BOT_AVATAR = "🧑‍💻", "🛠️"

TOPICS = [
    ("🔐", "VPN", "Connection and sign-in errors", "My VPN says authentication failed, what should I do?"),
    ("🔑", "Passwords", "Reset, expiry and lockouts", "How do I reset my password?"),
    ("📱", "MFA", "Authenticator and new phones", "How do I set up MFA on a new phone?"),
    ("📧", "Outlook", "Email and sync problems", "Outlook keeps asking for my password"),
    ("🖨️", "Printing", "Add printers, fix print queues", "How do I add an office printer?"),
    ("💾", "BitLocker", "Recovery keys and encryption", "How do I get my BitLocker recovery key?"),
    ("🎣", "Phishing", "Report suspicious email", "I clicked a link in a phishing email"),
    ("🐢", "Slow laptop", "Speed up a sluggish device", "My laptop is very slow, what can I do?"),
]

ARTICLE_ICONS = {
    "email_outlook.md": "📧", "hardware_replacement.md": "🖥️", "laptop_encryption.md": "💾", "mfa_setup.md": "📱",
    "new_joiner_access.md": "👋", "password_reset.md": "🔑", "phishing_reporting.md": "🎣", "printer_setup.md": "🖨️",
    "slow_laptop.md": "🐢", "software_request.md": "📦", "teams_meetings.md": "🎧", "ticket_priority_sla.md": "🎫",
    "vpn_setup.md": "🔐", "wifi_access.md": "📶",
}

# From data/kb/ticket_priority_sla.md
SLA = {
    "P1 Critical": ("15 minutes", "4 hours"),
    "P2 High": ("1 hour", "8 business hours"),
    "P3 Medium": ("4 business hours", "3 business days"),
    "P4 Low": ("1 business day", "5 business days"),
}
PRIORITY_CLASS = {"P1 Critical": "p1", "P2 High": "p2", "P3 Medium": "p3", "P4 Low": "p4"}

# ---- styling -----------------------------------------------------------------
# Streamlit does not expose its theme colors to CSS, so the palette is chosen here from the active theme.

DARK = (st.context.theme.type or "light") == "dark"
TOKENS = {
    "surface": "#131c2f" if DARK else "#ffffff",
    "surface-2": "#0f1729" if DARK else "#f6f8fc",
    "border": "rgba(148,163,184,.16)" if DARK else "rgba(15,23,42,.08)",
    "border-strong": "rgba(148,163,184,.28)" if DARK else "rgba(15,23,42,.16)",
    "muted": "#94a3b8" if DARK else "#64748b",
    "accent": "#3b82f6" if DARK else "#2563eb",
    "accent-soft": "rgba(59,130,246,.16)" if DARK else "rgba(37,99,235,.08)",
    "accent-text": "#93c5fd" if DARK else "#1d4ed8",
    "shadow": "0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.22)" if DARK
              else "0 1px 2px rgba(15,23,42,.04), 0 6px 20px rgba(15,23,42,.06)",
}
st.markdown(
    "<style>:root{" + "".join(f"--hd-{k}:{v};" for k, v in TOKENS.items()) + "}</style>",
    unsafe_allow_html=True,
)
st.markdown(
    """
<style>
html, body, .stApp, .stMarkdown, button, input, textarea, [data-testid="stChatInput"] textarea {
  font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif !important;
}
.block-container {padding-top: 3.4rem; padding-bottom: 7rem; max-width: 880px;}
@keyframes hd-in {from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: none;}}
@keyframes hd-blink {0%, 80%, 100% {opacity: .25; transform: scale(.85);} 40% {opacity: 1; transform: scale(1);}}

/* header */
.hd-hero {
  position: relative; isolation: isolate; overflow: hidden; color: #fff; border-radius: 22px;
  display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
  padding: 1.9rem 2rem 1.7rem; margin-bottom: 1rem; cursor: default;
  background: radial-gradient(700px 260px at 0% 0%, rgba(56,189,248,.35), transparent 60%),
              linear-gradient(135deg, #172554 0%, #1e40af 45%, #2563eb 100%);
  box-shadow: 0 10px 30px -12px rgba(37,99,235,.55);
  transition: transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s;
}
/* hover palette: fades in over the base gradient (gradients themselves can't be transitioned) */
.hd-hero:before {
  content: ""; position: absolute; inset: 0; z-index: -1; opacity: 0; transition: opacity .5s ease;
  background: radial-gradient(600px 240px at 100% 100%, rgba(34,211,238,.45), transparent 60%),
              linear-gradient(135deg, #312e81 0%, #6d28d9 50%, #0ea5e9 100%);
}
.hd-hero .orb {position: absolute; z-index: -1; border-radius: 50%; pointer-events: none;
               transition: transform .8s cubic-bezier(.2,.8,.2,1), opacity .5s;}
.hd-hero .orb.a {width: 260px; height: 260px; right: -70px; top: -90px; background: rgba(255,255,255,.08);}
.hd-hero .orb.b {width: 140px; height: 140px; right: 180px; bottom: -80px; background: rgba(255,255,255,.06);}
.hd-hero .orb.c {width: 340px; height: 340px; left: -160px; bottom: -220px;
                 background: radial-gradient(circle, rgba(147,197,253,.25), transparent 70%);}
.hd-hero:hover {transform: translateY(-3px); box-shadow: 0 22px 44px -14px rgba(109,40,217,.6);}
.hd-hero:hover:before {opacity: 1;}
.hd-hero:hover .orb.a {transform: translate(-30px, 24px) scale(1.12);}
.hd-hero:hover .orb.b {transform: translate(-50px, -18px) scale(1.25); opacity: .8;}
.hd-hero:hover .orb.c {transform: translate(40px, -30px);}
.hd-hero .copy {position: relative; min-width: 0;}
.hd-hero .eyebrow {display: inline-block; font-size: .7rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
                   opacity: .75; margin-bottom: .45rem;}
.hd-hero h1 {color: #fff; font-size: 2rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: -.03em; line-height: 1.15;}
.hd-hero h1 [data-testid="stHeaderActionElements"], .hd-hero h1 a {display: none !important;}
.hd-hero h1 em {font-style: normal; background: linear-gradient(90deg, #93c5fd, #e0f2fe); -webkit-background-clip: text;
                background-clip: text; color: transparent; transition: filter .5s;}
.hd-hero:hover h1 em {filter: hue-rotate(110deg) saturate(1.6);}
.hd-hero p {margin: .55rem 0 0; opacity: .88; font-size: .98rem; max-width: 32rem; line-height: 1.5;}
.hd-hero .pills {display: flex; flex-wrap: wrap; gap: .45rem; margin-top: 1rem;}
.hd-hero .pills .hd-pill {margin: 0; border: 1px solid rgba(255,255,255,.18); backdrop-filter: blur(6px);
                          transition: background .3s, border-color .3s;}
.hd-hero:hover .pills .hd-pill {background: rgba(255,255,255,.22); border-color: rgba(255,255,255,.32);}
.hd-hero .art {position: relative; flex: none; width: 150px; height: 120px;}
.hd-hero .tile {position: absolute; inset: 14px 22px; display: grid; place-items: center; font-size: 2.6rem;
                border-radius: 24px; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
                backdrop-filter: blur(10px); box-shadow: 0 12px 30px -10px rgba(0,0,0,.4);
                transition: transform .5s cubic-bezier(.2,.8,.2,1);}
.hd-hero .badge {position: absolute; display: grid; place-items: center; width: 34px; height: 34px; border-radius: 11px;
                 font-size: 1rem; background: rgba(255,255,255,.92); box-shadow: 0 6px 16px -6px rgba(0,0,0,.45);
                 transition: transform .5s cubic-bezier(.2,.8,.2,1);}
.hd-hero .badge.b1 {left: 0; top: 4px;} .hd-hero .badge.b2 {right: 0; top: 30px;} .hd-hero .badge.b3 {left: 30px; bottom: 0;}
.hd-hero:hover .tile {transform: rotate(-6deg) scale(1.06);}
.hd-hero:hover .badge.b1 {transform: translate(-6px, -6px) rotate(-10deg);}
.hd-hero:hover .badge.b2 {transform: translate(6px, -4px) rotate(8deg);}
.hd-hero:hover .badge.b3 {transform: translate(-4px, 6px) rotate(-6deg);}
@keyframes hd-pulse {0% {box-shadow: 0 0 0 0 rgba(74,222,128,.6);} 70% {box-shadow: 0 0 0 7px rgba(74,222,128,0);}
                     100% {box-shadow: 0 0 0 0 rgba(74,222,128,0);}}
.hd-hero .hd-dot:not(.off) {animation: hd-pulse 2s infinite;}
@media (prefers-reduced-motion: reduce) {.hd-hero, .hd-hero *, .hd-hero:before {transition: none !important; animation: none !important;}}
.hd-bar {
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding: .7rem 1rem; border-radius: 14px; margin-bottom: .9rem;
  background: var(--hd-surface); border: 1px solid var(--hd-border); box-shadow: var(--hd-shadow);
}
.hd-bar .t {font-weight: 700; font-size: 1.02rem; letter-spacing: -.01em;}
.hd-pill {
  display: inline-flex; align-items: center; gap: .45rem; font-size: .76rem; font-weight: 500;
  padding: .25rem .7rem; border-radius: 999px; background: rgba(255,255,255,.16); margin-top: .9rem;
}
.hd-bar .hd-pill {margin: 0; background: var(--hd-accent-soft); color: var(--hd-accent-text);}
.hd-dot {width: 8px; height: 8px; border-radius: 50%; background: #4ade80; box-shadow: 0 0 0 3px rgba(74,222,128,.25);}
.hd-dot.off {background: #fbbf24; box-shadow: 0 0 0 3px rgba(251,191,36,.25);}

/* navigation */
[data-testid="stButtonGroup"] button {border-radius: 999px !important; padding: .3rem .95rem !important;}

/* section labels and steps */
.hd-label {font-size: .74rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
           color: var(--hd-muted); margin: 1.1rem 0 .5rem;}
.hd-steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin-top: .4rem;}
.hd-step {background: var(--hd-surface); border: 1px solid var(--hd-border); border-radius: 14px; padding: .8rem .9rem;}
.hd-step b {display: block; font-size: .88rem; margin-bottom: .15rem;}
.hd-step span {font-size: .8rem; color: var(--hd-muted);}
.hd-step i {font-style: normal; display: inline-grid; place-items: center; width: 24px; height: 24px; border-radius: 8px;
            background: var(--hd-accent-soft); color: var(--hd-accent-text); font-size: .78rem; font-weight: 700;
            margin-bottom: .45rem;}
@media (max-width: 640px) {
  .hd-steps {grid-template-columns: 1fr;}
  .hd-hero {padding: 1.4rem 1.3rem;} .hd-hero h1 {font-size: 1.5rem;} .hd-hero .art {display: none;}
}

/* topic cards */
.st-key-topics [data-testid="stButton"] button {
  min-height: 92px; width: 100%; justify-content: flex-start; align-items: flex-start; text-align: left;
  padding: .85rem .95rem; border-radius: 16px; border: 1px solid var(--hd-border);
  background: var(--hd-surface); box-shadow: var(--hd-shadow); transition: transform .12s, border-color .12s;
}
.st-key-topics [data-testid="stButton"] button:hover {transform: translateY(-2px); border-color: var(--hd-accent);}
.st-key-topics [data-testid="stButton"] button p {text-align: left; font-size: .8rem; color: var(--hd-muted); margin: 0;}
.st-key-topics [data-testid="stButton"] button p strong {display: block; font-size: .95rem; color: inherit;
  margin-bottom: .1rem; filter: none;}

/* chat */
[data-testid="stChatMessage"] {animation: hd-in .25s ease-out; background: transparent; padding: .35rem 0;}
[data-testid="stChatMessage"]:has(> [aria-label="Chat message from user"]) {flex-direction: row-reverse;}
[aria-label="Chat message from user"] {
  flex: 0 1 auto; max-width: 78%; margin-left: auto; padding: .6rem 1rem; border-radius: 18px 18px 4px 18px;
  background: var(--hd-accent); color: #fff;
}
[aria-label="Chat message from user"] p {color: #fff;}
[aria-label="Chat message from assistant"] {
  padding: .85rem 1.1rem; border-radius: 4px 18px 18px 18px;
  background: var(--hd-surface); border: 1px solid var(--hd-border); box-shadow: var(--hd-shadow);
}
[data-testid="stChatInput"] {border-radius: 16px !important; box-shadow: var(--hd-shadow);}
.hd-typing {display: inline-flex; align-items: center; gap: .55rem; color: var(--hd-muted); font-size: .86rem;}
.hd-typing span.d {width: 7px; height: 7px; border-radius: 50%; background: var(--hd-accent); display: inline-block;
                   animation: hd-blink 1.2s infinite;}
.hd-typing span.d:nth-child(2) {animation-delay: .15s;} .hd-typing span.d:nth-child(3) {animation-delay: .3s;}
.cite {
  display: inline-block; font-size: .66rem; font-weight: 700; line-height: 1; vertical-align: super;
  padding: .17rem .38rem; margin: 0 .06rem; border-radius: 6px; background: var(--hd-accent-soft); color: var(--hd-accent-text);
}
.hd-thumb {display: block; max-width: 260px; max-height: 200px; border-radius: 12px; margin: .45rem 0 .1rem auto;
           border: 1px solid rgba(255,255,255,.35);}
.hd-attach {display: flex; flex-wrap: wrap; gap: .35rem; justify-content: flex-end; margin-top: .4rem;}
.hd-attach span {font-size: .74rem; padding: .18rem .55rem; border-radius: 999px; background: rgba(255,255,255,.2); color: #fff;}
.hd-voice {font-size: .72rem; font-weight: 600; letter-spacing: .02em; opacity: .85; margin-bottom: .15rem; color: #fff;}
.hd-rewrite {font-size: .78rem; color: var(--hd-muted); margin-bottom: .45rem;}
.hd-meta {display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin-top: .55rem; font-size: .74rem; color: var(--hd-muted);}
.hd-chip {display: inline-flex; align-items: center; gap: .3rem; padding: .18rem .55rem; border-radius: 999px;
          border: 1px solid var(--hd-border); background: var(--hd-surface-2); font-size: .74rem;}
.hd-chip.cited {border-color: transparent; background: var(--hd-accent-soft); color: var(--hd-accent-text);}
.src-card {border: 1px solid var(--hd-border); border-radius: 12px; padding: .65rem .85rem; margin-bottom: .5rem;
           background: var(--hd-surface-2);}
.src-card.cited {border-left: 3px solid var(--hd-accent);}
.src-head {font-weight: 600; font-size: .88rem;}
.src-meta {font-size: .72rem; color: var(--hd-muted); margin-top: .1rem;}
.src-body {font-size: .82rem; opacity: .9; margin-top: .35rem; white-space: pre-wrap;}
.st-key-followups [data-testid="stButton"] button {
  border-radius: 999px; padding: .2rem .8rem; min-height: 0; font-size: .8rem;
  border: 1px solid var(--hd-border); background: var(--hd-surface);
}
.st-key-followups [data-testid="stButton"] button:hover {border-color: var(--hd-accent); color: var(--hd-accent-text);}
.hd-foot {text-align: center; font-size: .72rem; color: var(--hd-muted); margin-top: 1.2rem;}

/* knowledge base */
.hd-card {background: var(--hd-surface); border: 1px solid var(--hd-border); border-radius: 16px; padding: 1rem 1.05rem .6rem;
          box-shadow: var(--hd-shadow); min-height: 150px;}
.hd-card .ic {display: inline-grid; place-items: center; width: 38px; height: 38px; border-radius: 12px;
              background: var(--hd-accent-soft); font-size: 1.15rem; margin-bottom: .55rem;}
.hd-card .tt {font-weight: 600; font-size: .96rem; line-height: 1.3;}
.hd-card .ss {font-size: .76rem; color: var(--hd-muted); margin-top: .35rem; line-height: 1.45;}

/* tickets */
.hd-ticket {display: flex; gap: .8rem; align-items: flex-start; background: var(--hd-surface); border: 1px solid var(--hd-border);
            border-radius: 14px; padding: .75rem .95rem; margin-bottom: .55rem;}
.hd-ticket .id {font-family: ui-monospace, monospace; font-size: .78rem; color: var(--hd-muted);}
.hd-ticket .sm {font-weight: 600; font-size: .92rem; margin: .1rem 0;}
.hd-ticket .mt {font-size: .74rem; color: var(--hd-muted);}
.hd-prio {flex: none; font-size: .7rem; font-weight: 700; padding: .25rem .55rem; border-radius: 8px; white-space: nowrap;}
.hd-prio.p1 {background: rgba(239,68,68,.15); color: #ef4444;}
.hd-prio.p2 {background: rgba(249,115,22,.15); color: #f97316;}
.hd-prio.p3 {background: var(--hd-accent-soft); color: var(--hd-accent-text);}
.hd-prio.p4 {background: rgba(100,116,139,.15); color: var(--hd-muted);}

/* insights */
[data-testid="stMetric"] {background: var(--hd-surface); border: 1px solid var(--hd-border); border-radius: 16px;
                          padding: .85rem 1rem; box-shadow: var(--hd-shadow);}
[data-testid="stMetricLabel"] p {font-size: .78rem; color: var(--hd-muted);}

/* sidebar */
[data-testid="stSidebar"] .hd-brand {display: flex; align-items: center; gap: .6rem; margin-bottom: .6rem;}
[data-testid="stSidebar"] .hd-brand .lg {display: grid; place-items: center; width: 36px; height: 36px; border-radius: 11px;
  background: linear-gradient(135deg, #1e3a8a, #3b82f6); font-size: 1.05rem;}
[data-testid="stSidebar"] .hd-brand b {font-size: 1rem; display: block; line-height: 1.1;}
[data-testid="stSidebar"] .hd-brand span {font-size: .74rem; color: var(--hd-muted);}
.hd-urgent {font-size: .78rem; border-radius: 12px; padding: .65rem .75rem; background: rgba(239,68,68,.1);
            border: 1px solid rgba(239,68,68,.25);}
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {border-radius: 12px;}
</style>
""",
    unsafe_allow_html=True,
)


# ---- data loading ------------------------------------------------------------


@st.cache_resource(show_spinner="Starting the assistant (first load only)...")
def load_pipeline():
    from src.rag import RAGPipeline

    # The index is gitignored, so build it on first start (e.g. on a fresh cloud deploy).
    if not config.CHUNKS_FILE.exists():
        from src.ingest import build_index

        build_index()
    pipeline = RAGPipeline()
    pipeline.retrieve("warm up")  # the first ONNX run is slow; pay it here instead of on the first question
    return pipeline


@st.cache_data
def load_articles() -> list[dict]:
    articles = []
    for path in sorted(config.KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        body = text.split("\n", 1)[1] if "\n" in text else ""
        sections = re.findall(r"^## (.+)$", body, flags=re.MULTILINE)
        articles.append({"file": path.name, "title": title, "body": body, "sections": sections})
    return articles


# ---- state & callbacks -------------------------------------------------------

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("view", CHAT)
ss.setdefault("last_view", CHAT)
ss.setdefault("ticket_prefill", {})
ss.setdefault("ticket_created", None)

# The session id lives in the URL, so reloading the page (e.g. after a theme switch) restores the chat.
if "sid" not in ss:
    sid = st.query_params.get("sid", "")
    if not re.fullmatch(r"[0-9a-f]{32}", sid):
        sid = uuid.uuid4().hex
        st.query_params["sid"] = sid
    ss.sid = sid
    ss.messages = store.load_session(sid)


def keep_view_selected():
    # A segmented control can be clicked off; fall back to the previous view.
    if ss.view is None:
        ss.view = ss.last_view
    ss.last_view = ss.view


def ask(question: str):
    ss.pending = question
    ss.view = ss.last_view = CHAT


def new_chat():
    ss.messages = []
    store.save_session(ss.sid, [])


def switch_theme():
    # Streamlit reads the theme choice from the browser's localStorage on page load, so save it and reload.
    store.save_session(ss.sid, ss.messages)
    ss.pending_theme = "Dark" if ss.dark_mode else "Light"


def save_feedback(msg: dict):
    rating = ss.get(f"fb_{msg['id']}")
    if rating is None:
        return
    store.append(store.FEEDBACK, {
        "message_id": msg["id"],
        "question": msg.get("question", ""),
        "rating": "up" if rating == 1 else "down",
    })


def start_ticket(msg: dict):
    question = msg.get("question", "")
    transcript = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in ss.messages[-6:])
    top = msg.get("sources", [{}])[0].get("title", "") if msg.get("sources") else ""
    ss.ticket_prefill = {
        "summary": question[:120],
        "category": top,
        "priority": guess_priority(question),
        "description": f"{question}\n\nWhat I already tried (from the assistant chat):\n{transcript}",
    }
    if msg.get("attachments"):
        ss.ticket_prefill["description"] += "\n\nAttachments:\n" + "\n".join(
            f"- {a['name']}: {a['summary'][:300]}" for a in msg["attachments"]
        )
    ss.ticket_created = None
    ss.view = ss.last_view = TICKETS


def export_chat() -> str:
    lines = [f"# IT Helpdesk chat, {datetime.now():%Y-%m-%d %H:%M}\n"]
    for m in ss.messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"**{who}:** {m['content']}\n")
        for a in m.get("attachments", []):
            lines.append(f"> 📎 {a['name']}: {a['summary'][:300]}")
        for i, s in enumerate(m.get("sources", []), start=1):
            if i in m.get("cited", []):
                lines.append(f"> [{i}] {s['title']} > {s['section']} (`{s['source']}`)")
        lines.append("")
    return "\n".join(lines)


# ---- rendering helpers -------------------------------------------------------

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
TYPING = '<div class="hd-typing"><span class="d"></span><span class="d"></span><span class="d"></span>{}</div>'


def badgify(answer: str) -> str:
    """Escape raw HTML from the model, then turn [1] / [1, 2] citations into small badges."""
    safe = html.escape(normalize_citations(answer), quote=False)
    return _CITE_RE.sub(
        lambda m: "".join(f'<span class="cite">{n.strip()}</span>' for n in m.group(1).split(",")), safe
    )


def render_sources(sources: list[dict], cited: list[int]) -> None:
    if not sources:
        return
    with st.expander(f"View {len(sources)} source passages"):
        for i, src in enumerate(sources, start=1):
            heading = html.escape(src["title"] + (f" › {src['section']}" if src["section"] else ""))
            plain = re.sub(r"\*\*|`", "", src["text"])  # cards show plain text, not raw markdown
            body = html.escape(plain[:420] + ("…" if len(plain) > 420 else ""))
            st.markdown(
                f'<div class="src-card{" cited" if i in cited else ""}">'
                f'<div class="src-head"><span class="cite">{i}</span> {heading}</div>'
                f'<div class="src-meta">{html.escape(src["source"])} · relevance {src["score"]:.3f}'
                f'{" · cited" if i in cited else ""}</div>'
                f'<div class="src-body">{body}</div></div>',
                unsafe_allow_html=True,
            )


def followup_suggestions(msg: dict) -> list[str]:
    """Related sections the answer did not cite, offered as one-click next questions (no LLM call needed)."""
    seen, out = set(), []
    for i, src in enumerate(msg.get("sources", []), start=1):
        key = (src["title"], src["section"])
        if i in msg.get("cited", []) or not src["section"] or key in seen:
            continue
        seen.add(key)
        out.append(f"{src['title'].split(' and ')[0]}: {src['section']}")
    return out[:3]


def render_assistant(msg: dict, is_last: bool) -> None:
    if msg.get("rewritten"):
        st.markdown(f'<div class="hd-rewrite">🔄 Understood as: <i>{html.escape(msg["rewritten"])}</i></div>',
                    unsafe_allow_html=True)
    st.markdown(badgify(msg["content"]), unsafe_allow_html=True)
    if msg.get("kind") == "smalltalk":
        return

    sources, cited = msg.get("sources", []), msg.get("cited", [])
    by_article: dict[str, list[str]] = {}
    for i in cited:
        by_article.setdefault(sources[i - 1]["title"], []).append(str(i))
    chips = [f'<span class="hd-chip cited">{", ".join(nums)} · {html.escape(title)}</span>'
             for title, nums in by_article.items()]
    if msg.get("latency") is not None:
        chips.append(f'<span>⏱ {msg["latency"]:.1f}s</span>')
    if chips:
        st.markdown(f'<div class="hd-meta">{"".join(chips)}</div>', unsafe_allow_html=True)
    render_sources(sources, cited)

    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        st.feedback("thumbs", key=f"fb_{msg['id']}", on_change=save_feedback, args=(msg,))
        unanswered = msg.get("fallback")
        st.button(
            "🎫 Raise a ticket" if unanswered else "Still stuck? Raise a ticket",
            key=f"tk_{msg['id']}", on_click=start_ticket, args=(msg,),
            type="primary" if unanswered else "tertiary",
        )

    if is_last and not msg.get("fallback"):
        suggestions = followup_suggestions(msg)
        if suggestions:
            st.markdown('<div class="hd-label" style="margin-top:.4rem">Related</div>', unsafe_allow_html=True)
            with st.container(key="followups", horizontal=True):
                for j, s in enumerate(suggestions):
                    st.button(s, key=f"fu_{msg['id']}_{j}", on_click=ask, args=(f"Tell me about {s}",))


# ---- sidebar -----------------------------------------------------------------

articles = load_articles()
llm_on = config.LLM_PROVIDER != "none"

with st.sidebar:
    st.markdown(
        '<div class="hd-brand"><div class="lg">🛠️</div><div><b>IT Helpdesk</b>'
        '<span>Self-service assistant</span></div></div>',
        unsafe_allow_html=True,
    )
    st.toggle("🌙 Dark mode", value=DARK, key="dark_mode", on_change=switch_theme)
    st.button("➕ New chat", on_click=new_chat, width="stretch", type="primary")
    st.download_button(
        "⬇️ Export chat",
        data=export_chat(),
        file_name=f"it-helpdesk-chat-{datetime.now():%Y%m%d-%H%M}.md",
        mime="text/markdown",
        width="stretch",
        disabled=not ss.messages,
    )

    with st.expander("⚙️ Answer settings"):
        mode = st.selectbox(
            "Retrieval mode",
            MODES,
            help="hybrid = semantic search + BM25 keyword search, merged with Reciprocal Rank Fusion",
        )
        top_k = st.slider("Passages to retrieve (k)", 1, 8, config.TOP_K)
        memory = st.toggle(
            "Understand follow-up questions",
            value=True,
            help="Uses the conversation so far to turn follow-ups like 'what if that fails?' into a full question.",
        )

    with st.expander("ℹ️ System"):
        st.caption(f"**LLM:** {describe()}")
        st.caption(f"**Embeddings:** {config.EMBEDDING_MODEL}")
        st.caption(f"**Knowledge base:** {len(articles)} articles")
        if st.button("Rebuild index", width="stretch"):
            from src.ingest import build_index

            with st.spinner("Re-indexing knowledge base..."):
                n = build_index()
            load_pipeline.clear()
            load_articles.clear()
            st.success(f"Indexed {n} chunks")

    st.markdown(
        '<div class="hd-urgent">🚨 <b>Urgent or a security incident?</b><br>'
        "Call the IT Service Desk, available 24/7.</div>",
        unsafe_allow_html=True,
    )

# ---- header ------------------------------------------------------------------

status = f"Online · {describe()}" if llm_on else "Retrieval only · no LLM configured"
pill = f'<span class="hd-pill"><span class="hd-dot{"" if llm_on else " off"}"></span>{html.escape(status)}</span>'
if ss.messages and ss.view == CHAT:
    st.markdown(f'<div class="hd-bar"><span class="t">🛠️ IT Helpdesk Assistant</span>{pill}</div>',
                unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="hd-hero">'
        '<span class="orb a"></span><span class="orb b"></span><span class="orb c"></span>'
        '<div class="copy"><span class="eyebrow">IT self-service · 24/7</span>'
        "<h1>How can we <em>help</em> today?</h1>"
        "<p>Ask any IT question. Answers come from the company knowledge base, with a source for every step.</p>"
        f'<div class="pills">{pill}<span class="hd-pill">📚 {len(articles)} articles</span>'
        '<span class="hd-pill">🔗 Cited answers</span></div></div>'
        '<div class="art" aria-hidden="true"><div class="tile">🛠️</div>'
        '<span class="badge b1">🔐</span><span class="badge b2">📧</span><span class="badge b3">🖨️</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )

view = st.segmented_control(
    "View", VIEWS, key="view", on_change=keep_view_selected, label_visibility="collapsed"
) or CHAT

try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()
except Exception as e:  # e.g. missing API key or Ollama not running
    st.error(f"Could not start the assistant: {e}")
    st.stop()


# ---- chat view ---------------------------------------------------------------


def unpack_submission(submission) -> tuple[str, object, list]:
    """chat_input returns a plain string (topic buttons) or a ChatInputValue with text, files and audio."""
    if isinstance(submission, str):
        return submission.strip(), None, []
    files = list(getattr(submission, "files", None) or [])
    return (submission.text or "").strip(), getattr(submission, "audio", None), files


def read_attachments(text: str, audio, files: list) -> dict:
    """Transcribe the voice note and read each attached screenshot or file into text the RAG chain can use."""
    status = st.empty()
    attachments, problems = [], []
    if audio is not None:
        status.markdown(TYPING.format("Transcribing your voice note…"), unsafe_allow_html=True)
        try:
            spoken = media.transcribe(audio.getvalue(), audio.name or "voice.wav")
            text = f"{text} {spoken}".strip()
        except Exception as e:
            problems.append(f"The voice note could not be transcribed: {e}")
    for f in files:
        kind = media.kind_of(f.name)
        status.markdown(TYPING.format(f"Reading {html.escape(f.name)}…"), unsafe_allow_html=True)
        try:
            data, thumb = f.getvalue(), None
            if kind == "image":
                if not media.groq_available():
                    raise RuntimeError("reading screenshots needs GROQ_API_KEY")
                summary, thumb = media.describe_image(data, f.type), media.thumbnail_data_url(data)
            else:
                summary = media.extract_text(f.name, data)
            if not summary:
                raise ValueError("no readable text found")
            attachments.append({"name": f.name, "kind": kind, "summary": summary, "thumb": thumb})
        except Exception as e:
            problems.append(f"Could not read {f.name}: {e}")
    status.empty()
    if not text and attachments:
        has_image = any(a["kind"] == "image" for a in attachments)
        text = ("Help me fix the problem shown in the attached screenshot." if has_image
                else "Help me with the issue in the attached file.")
    return {"role": "user", "content": text, "attachments": attachments, "voice": audio is not None,
            "problems": problems}


def render_user(msg: dict) -> None:
    if msg.get("voice"):
        st.markdown('<div class="hd-voice">🎙️ Voice note</div>', unsafe_allow_html=True)
    if msg["content"]:
        st.markdown(msg["content"])
    attachments = msg.get("attachments", [])
    for a in attachments:
        if a.get("thumb"):
            st.markdown(f'<img class="hd-thumb" src="{a["thumb"]}" alt="{html.escape(a["name"])}">',
                        unsafe_allow_html=True)
    files = [a for a in attachments if not a.get("thumb")]
    if files:
        chips = "".join(f"<span>📎 {html.escape(a['name'])}</span>" for a in files)
        st.markdown(f'<div class="hd-attach">{chips}</div>', unsafe_allow_html=True)
    for problem in msg.get("problems", []):
        st.warning(problem, icon="⚠️")


def chat_view():
    if not ss.messages:
        st.markdown('<div class="hd-label">Popular topics</div>', unsafe_allow_html=True)
        with st.container(key="topics"):
            for row in (TOPICS[:4], TOPICS[4:]):
                cols = st.columns(4)
                for col, (icon, name, blurb, question) in zip(cols, row):
                    col.button(f"{icon} **{name}**\n\n{blurb}", key=f"topic_{name}", on_click=ask,
                               args=(question,), width="stretch")
        st.markdown(
            '<div class="hd-label">How it works</div><div class="hd-steps">'
            '<div class="hd-step"><i>1</i><b>Describe the issue</b><span>Type it, speak it 🎙️, or attach a screenshot or log file.</span></div>'
            '<div class="hd-step"><i>2</i><b>Get cited steps</b><span>Every step links to the KB article it came from.</span></div>'
            '<div class="hd-step"><i>3</i><b>Still stuck?</b><span>Raise a pre-filled ticket in one click.</span></div>'
            "</div>",
            unsafe_allow_html=True,
        )

    for idx, msg in enumerate(ss.messages):
        with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
            if msg["role"] == "assistant":
                render_assistant(msg, is_last=idx == len(ss.messages) - 1)
            else:
                render_user(msg)

    if ss.messages:
        st.markdown('<div class="hd-foot">Answers are generated from the IT knowledge base and can be wrong. '
                    "Check the sources for anything important.</div>", unsafe_allow_html=True)

    submission = st.chat_input(
        "Describe your IT issue, attach a screenshot, or record a voice note…",
        accept_file="multiple",
        file_type=list(media.FILE_TYPES),
        accept_audio=media.groq_available(),  # Whisper runs on Groq
        max_upload_size=10,
    ) or ss.pop("pending", None)
    if not submission:
        return
    text, audio, files = unpack_submission(submission)

    # Only real Q&A turns give useful context; greetings would just trigger needless rewrites.
    turns = [t for m in ss.messages if m["role"] == "assistant" and m.get("kind") != "smalltalk"
             for t in ({"role": "user", "content": m["question"]}, m)]
    history = format_history(turns) if memory else ""
    with st.chat_message("user", avatar=USER_AVATAR):
        user_msg = read_attachments(text, audio, files)
        render_user(user_msg)
    if not user_msg["content"]:
        return  # nothing usable (e.g. the voice note could not be transcribed); the warning stays on screen
    ss.messages.append(user_msg)
    question, attachments = user_msg["content"], user_msg["attachments"]

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        reply = None if attachments else small_talk_reply(question)
        if reply:
            ss.messages.append({"role": "assistant", "content": reply, "kind": "smalltalk", "id": uuid.uuid4().hex})
            st.rerun()

        start = time.perf_counter()
        placeholder = st.empty()
        placeholder.markdown(TYPING.format("Searching the knowledge base…"), unsafe_allow_html=True)
        # Rewriting costs an extra LLM call, so only do it for questions that lean on the conversation.
        search_q = question
        if history and looks_like_followup(question):
            placeholder.markdown(TYPING.format("Reading the conversation…"), unsafe_allow_html=True)
            search_q = pipeline.condense(question, history)
        docs = pipeline.retrieve(media.search_text(search_q, attachments), k=top_k, mode=mode)
        placeholder.markdown(TYPING.format(f"Found {len(docs)} passages, writing the answer…"), unsafe_allow_html=True)

        answer, last_paint = "", 0.0
        try:
            for chunk in pipeline.stream(media.build_prompt(search_q, attachments), docs):
                answer += chunk
                if time.perf_counter() - last_paint > 0.05:  # repainting on every token makes long answers lag
                    placeholder.markdown(answer + " ▌")
                    last_paint = time.perf_counter()
        except Exception as e:
            answer = f"⚠️ The language model could not be reached: {e}"
        answer = normalize_citations(answer)
        placeholder.markdown(answer)
        latency = time.perf_counter() - start

    sources = [
        {**{k: d.metadata.get(k, "") for k in ("title", "section", "source", "text")},
         "score": d.metadata.get("score", 0.0)}
        for d in docs
    ]
    fallback = pipeline.is_fallback(answer) or not docs
    msg = {
        "id": uuid.uuid4().hex,
        "role": "assistant",
        "content": answer,
        "question": question,
        "rewritten": search_q if search_q.strip().lower() != question.strip().lower() else None,
        "sources": sources,
        "cited": extract_citations(answer, len(docs)),
        "latency": latency,
        "mode": mode,
        "fallback": fallback,
        "attachments": [{"name": a["name"], "summary": a["summary"]} for a in attachments],
    }
    ss.messages.append(msg)
    store.append(store.QUERIES, {
        "question": question,
        "search_query": search_q,
        "attachments": [a["kind"] for a in attachments],
        "answered": not fallback,
        "top_article": sources[0]["title"] if sources and not fallback else None,
        "latency": round(latency, 2),
        "mode": mode,
        "llm": describe(),
    })
    st.rerun()


# ---- knowledge base view -----------------------------------------------------


@st.dialog("Knowledge base article", width="large")
def show_article(a: dict):
    st.markdown(f"### {ARTICLE_ICONS.get(a['file'], '📄')} {a['title']}")
    st.caption(f"`{a['file']}` · {len(a['sections'])} sections")
    st.markdown(re.sub(r"^## ", "#### ", a["body"], flags=re.MULTILINE))
    if st.button(f"💬 Ask about {a['title']}", type="primary", width="stretch"):
        ask(f"Give me an overview of {a['title']}")
        st.rerun()


def kb_view():
    query = st.text_input("Search articles", placeholder="🔍  Search articles, e.g. printer, recovery key, Teams audio",
                          label_visibility="collapsed")
    q = query.lower().strip()
    shown = [a for a in articles if not q or q in a["title"].lower() or q in a["body"].lower()]
    st.markdown(f'<div class="hd-label">{len(shown)} of {len(articles)} articles</div>', unsafe_allow_html=True)
    if not shown:
        st.info("No articles match. Try another word, or ask the assistant in the Chat tab.", icon="🔍")
    for row in range(0, len(shown), 2):
        cols = st.columns(2)
        for col, a in zip(cols, shown[row:row + 2]):
            with col:
                sections = " · ".join(a["sections"][:3]) + (" …" if len(a["sections"]) > 3 else "")
                st.markdown(
                    f'<div class="hd-card"><div class="ic">{ARTICLE_ICONS.get(a["file"], "📄")}</div>'
                    f'<div class="tt">{html.escape(a["title"])}</div>'
                    f'<div class="ss">{html.escape(sections)}</div></div>',
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                if c1.button("Read", key=f"read_{a['file']}", width="stretch"):
                    show_article(a)
                c2.button("Ask", key=f"ask_{a['file']}", on_click=ask, width="stretch",
                          args=(f"Give me an overview of {a['title']}",))


# ---- tickets view ------------------------------------------------------------


def tickets_view():
    if ss.ticket_created:
        t = ss.ticket_created
        first, resolve = SLA[t["priority"]]
        st.success(f"Ticket **{t['id']}** created ({t['priority']}). First response within **{first}**, "
                   f"target resolution **{resolve}**.", icon="✅")

    pre = ss.ticket_prefill
    titles = [a["title"] for a in articles] + ["Other"]
    with st.form("ticket", clear_on_submit=True, border=True):
        st.markdown("**Raise a ticket**")
        st.caption("Demo mode: tickets are saved locally to `logs/tickets.jsonl`, not sent to a real service desk.")
        summary = st.text_input("Summary", value=pre.get("summary", ""), placeholder="e.g. VPN fails with 'Authentication failed'")
        c1, c2 = st.columns(2)
        category = c1.selectbox(
            "Category", titles, index=titles.index(pre["category"]) if pre.get("category") in titles else len(titles) - 1
        )
        priority = c2.selectbox(
            "Priority", PRIORITIES, index=PRIORITIES.index(pre.get("priority", "P3 Medium")),
            help="The Service Desk may change the priority based on actual impact.",
        )
        affected = st.radio("Who is affected?", ["Just me", "My team", "Many users"], horizontal=True)
        description = st.text_area(
            "Description", value=pre.get("description", ""), height=160,
            placeholder="What were you trying to do, the exact error, when it started, and what you already tried.",
        )
        submitted = st.form_submit_button("Submit ticket", type="primary", width="stretch")

    if submitted:
        if not summary.strip():
            st.error("Please add a short summary.")
        else:
            ss.ticket_created = store.append(store.TICKETS, {
                "id": store.new_ticket_id(), "summary": summary.strip(), "category": category,
                "priority": priority, "affected": affected, "description": description.strip(), "status": "Open",
            })
            ss.ticket_prefill = {}
            st.rerun()

    with st.expander("Priority guide and response times"):
        st.markdown(
            "| Priority | Meaning | First response | Target resolution |\n|---|---|---|---|\n"
            "| P1 Critical | Business stopped for many users, or a security incident | 15 minutes | 4 hours |\n"
            "| P2 High | One team blocked, or a key user fully blocked | 1 hour | 8 business hours |\n"
            "| P3 Medium | One user affected, workaround exists | 4 business hours | 3 business days |\n"
            "| P4 Low | Requests and questions | 1 business day | 5 business days |"
        )

    tickets = store.read(store.TICKETS)
    st.markdown(f'<div class="hd-label">Your tickets ({len(tickets)})</div>', unsafe_allow_html=True)
    if not tickets:
        st.caption("No tickets yet.")
        return
    for t in tickets[::-1][:20]:
        created = datetime.fromisoformat(t["ts"]).astimezone().strftime("%d %b %Y, %H:%M")
        first, _ = SLA.get(t["priority"], ("", ""))
        st.markdown(
            f'<div class="hd-ticket"><span class="hd-prio {PRIORITY_CLASS.get(t["priority"], "p3")}">'
            f'{html.escape(t["priority"].split()[0])}</span><div>'
            f'<div class="id">{html.escape(t["id"])} · {html.escape(t.get("status", "Open"))}</div>'
            f'<div class="sm">{html.escape(t["summary"])}</div>'
            f'<div class="mt">{html.escape(t["category"])} · {created} · first response within {first}</div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )


# ---- insights view -----------------------------------------------------------


def insights_view():
    queries = store.read(store.QUERIES)
    feedback = store.read(store.FEEDBACK)
    if not queries:
        st.info("No questions logged yet. Ask something in the Chat tab and usage stats will appear here.", icon="📊")
        return

    answered = [q for q in queries if q.get("answered")]
    ups = sum(f["rating"] == "up" for f in feedback)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions", len(queries))
    c2.metric("Answered from KB", f"{len(answered) / len(queries):.0%}")
    c3.metric("Avg response", f"{sum(q.get('latency', 0) for q in queries) / len(queries):.1f}s")
    c4.metric("Rated helpful", f"{ups / len(feedback):.0%}" if feedback else "–",
              help=f"{len(feedback)} ratings" if feedback else "No ratings yet")

    topics = Counter(q["top_article"] for q in answered if q.get("top_article"))
    if topics:
        st.markdown('<div class="hd-label">Most asked topics</div>', unsafe_allow_html=True)
        df = pd.DataFrame(topics.most_common(8), columns=["Topic", "Questions"])
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config={"Questions": st.column_config.ProgressColumn(
                "Questions", format="%d", min_value=0, max_value=int(df["Questions"].max()))},
        )

    gaps = [q for q in queries if not q.get("answered")]
    st.markdown('<div class="hd-label">Knowledge gaps</div>', unsafe_allow_html=True)
    if gaps:
        st.caption("Questions the knowledge base could not answer. Good candidates for new articles.")
        for q in gaps[-8:][::-1]:
            st.markdown(f"- {q['question']}")
    else:
        st.caption("Every question so far was answered from the knowledge base. 🎉")

    downs = [f for f in feedback if f["rating"] == "down"]
    if downs:
        st.markdown('<div class="hd-label">Answers rated unhelpful</div>', unsafe_allow_html=True)
        for f in downs[-8:][::-1]:
            st.markdown(f"- {f['question']}")


if theme := ss.pop("pending_theme", None):
    components.html(
        f"""<script>
        const p = window.parent;
        p.localStorage.setItem(`stActiveTheme-${{p.location.pathname}}-v2`, JSON.stringify("{theme}"));
        p.location.reload();
        </script>""",
        height=0,
    )

{CHAT: chat_view, KB: kb_view, TICKETS: tickets_view, INSIGHTS: insights_view}[view]()

store.save_session(ss.sid, ss.messages)
