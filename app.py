"""Streamlit UI for the IT Helpdesk RAG assistant.

Run with:  streamlit run app.py
"""

import html
import os
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import streamlit as st

# On Streamlit Community Cloud the API keys live in the app's Secrets. Copy top-level ones into the
# environment before src.config reads it, so the same code works with a local .env file.
try:
    for _key, _value in st.secrets.items():
        # Also accept keys pasted under a heading such as [general] or [secrets].
        _items = _value.items() if hasattr(_value, "items") else [(_key, _value)]
        for _k, _v in _items:
            if isinstance(_v, (str, int, float, bool)):
                os.environ.setdefault(str(_k).strip().upper(), str(_v).strip())
except Exception:  # no secrets.toml (e.g. running locally with .env)
    pass

from src import auth, config, media, store, theme  # noqa: E402
from src.llm import describe
from src.utils import (
    ANSWER_STYLES, DEFAULT_STYLE, PRIORITIES, escape_markdown, extract_citations, format_history, group_by_day,
    guess_priority, history_messages, looks_like_followup, normalize_citations, small_talk_reply,
)

st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="🛠️", layout="wide", initial_sidebar_state="auto")

CHAT, KB, TICKETS, INSIGHTS, ADMIN = "💬 Chat", "📚 Knowledge base", "🎫 Tickets", "📊 Insights", "🛡️ Admin"
USER_VIEWS = [CHAT, KB, TICKETS]
ADMIN_VIEWS = [CHAT, KB, TICKETS, INSIGHTS, ADMIN]
BOT_AVATAR = "🛠️"
FOUNDER = config.FOUNDER_NAME
MODES = config.RETRIEVAL_MODES

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
TICKET_STATUSES = ("Open", "In progress", "Resolved", "Closed")
STATUS_CLASS = {"Open": "open", "In progress": "prog", "Resolved": "done", "Closed": "closed"}


# ---- assistant start-up, in the background ------------------------------------
# Loading the search index, the embedding model and the LLM client takes a few seconds. It starts on the very first
# page view (usually the login page), so by the time someone has signed in it is ready.


def _build_pipeline():
    from src.rag import RAGPipeline

    # The index is gitignored, so build it on first start (e.g. on a fresh cloud deploy).
    if not config.CHUNKS_FILE.exists():
        from src.ingest import build_index

        build_index()
    try:
        pipeline = RAGPipeline()
        pipeline.setup_error = None
    except RuntimeError as e:  # e.g. LLM_PROVIDER=groq but GROQ_API_KEY missing: still answer from retrieval
        pipeline = RAGPipeline(provider="none")
        pipeline.setup_error = str(e)
    pipeline.retrieve("warm up")  # the first ONNX run is slow; pay it here instead of on the first question
    return pipeline


@st.cache_resource(show_spinner=False)
def pipeline_future():
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="hd-warmup").submit(_build_pipeline)


def get_pipeline():
    future = pipeline_future()
    try:
        if future.done():
            return future.result()
        with st.spinner("Starting the assistant (first load only)…"):
            return future.result()
    except Exception:
        pipeline_future.clear()  # try again on the next run instead of caching the failure
        raise


pipeline_future()  # kick off the warm-up now


# ---- session & sign-in -------------------------------------------------------

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("view", CHAT)
ss.setdefault("last_view", CHAT)
ss.setdefault("ticket_prefill", {})
ss.setdefault("ticket_created", None)
ss.setdefault("style", DEFAULT_STYLE)
ss.setdefault("last_style", DEFAULT_STYLE)
ss.setdefault("chat_id", None)
ss.setdefault("attachments", [])  # files waiting in the Attach menu, sent with the next message
ss.setdefault("upload_nonce", 0)

# The session id lives in the URL, so reloading the page restores the sign-in; the open chat is in the URL too.
if "sid" not in ss:
    sid = st.query_params.get("sid", "")
    if not re.fullmatch(r"[0-9a-f]{32}", sid):
        sid = uuid.uuid4().hex
        st.query_params["sid"] = sid
    ss.sid = sid
    ss.user = store.load_auth(sid)
    chat_id = st.query_params.get("chat", "")
    saved = store.load_chat(ss.user["username"], chat_id) if ss.user and chat_id else None
    ss.messages, ss.chat_id = (saved, chat_id) if saved else ([], None)

# An admin may have removed or demoted this account since it signed in.
if ss.get("user") and not auth.still_valid(ss.user):
    store.clear_auth(ss.sid)
    ss.user = None

USER = ss.get("user")
IS_ADMIN = bool(USER) and USER["role"] == "admin"
FIRST_NAME = USER["name"].split()[0] if USER and USER["name"].strip() else ""
COLORS = theme.load()  # set by admins in Admin console → Appearance
ACCENT = COLORS["admin" if IS_ADMIN else "user"]


# ---- styling -----------------------------------------------------------------
# Every color is a CSS variable built from the portal's accent color. Each one holds a light and a dark value
# (CSS light-dark()), so switching the theme recolors the page instantly, without a reload.

st.markdown(f"<style>{theme.css_variables(ACCENT)}</style>", unsafe_allow_html=True)
st.markdown(
    """
<style>
/* ---------- page chrome: our own top bar replaces Streamlit's toolbar and menu ---------- */
/* Streamlit's header stays (see-through, not clickable) only for its "open sidebar" button; its menu stays in the
   page, hidden, because the Theme / Record screen / Print items drive it. */
[data-testid="stHeader"] {background: transparent !important; z-index: 999 !important;}
[data-testid="stHeader"], [data-testid="stHeader"] * {pointer-events: none !important;}
[data-testid="stToolbar"] {background: transparent !important;}
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
[data-testid="stStatusWidget"], [data-testid="stDecoration"] {display: none !important;}
[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {pointer-events: auto !important; border-radius: 9px !important;}
@media (max-width: 1240px) {.stApp:has([data-testid="stExpandSidebarButton"]) .st-key-topbar {padding-left: 2.8rem !important;}}
[data-testid="stMainMenuPopover"] {opacity: 0 !important; pointer-events: none !important;}
.stApp {background: var(--hd-bg) !important;}
[data-testid="stAppScrollToBottomContainer"] {overflow-x: hidden;}
.block-container {padding: 0 2rem 6rem !important; max-width: 1160px;}
[data-testid="stBottom"] > div {background: var(--hd-bg) !important;}
[data-testid="stBottomBlockContainer"] {max-width: 880px !important; padding: .5rem 1rem 1.3rem !important;}
[data-testid="stElementContainer"]:has(.hd-js),
[data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] > style) {display: none;}
.hd-mono {font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;}
@keyframes hd-in {from {opacity: 0; transform: translateY(4px);} to {opacity: 1; transform: none;}}
@keyframes hd-blink {0%, 80%, 100% {opacity: .25;} 40% {opacity: 1;}}

/* ---------- top bar: one row, pinned to the top, full width ---------- */
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar), [data-testid="stLayoutWrapper"]:has(> .st-key-loginbar) {
  position: sticky; top: 0; z-index: 990; isolation: isolate; margin-bottom: 1.6rem;}
.st-key-topbar, .st-key-loginbar {min-height: 64px; padding: .55rem 0; margin: 0 !important; gap: .5rem !important;
                                  flex-wrap: nowrap !important;}
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar)::before, [data-testid="stLayoutWrapper"]:has(> .st-key-loginbar)::before {
  content: ""; position: absolute; z-index: -1; top: 0; bottom: 0; left: calc(50% - 50vw); right: calc(50% - 50vw);
  background: var(--hd-bar); backdrop-filter: saturate(1.5) blur(14px); -webkit-backdrop-filter: saturate(1.5) blur(14px);
  border-bottom: 1px solid var(--hd-border);
}
.st-key-topbar > *, .st-key-loginbar > * {flex-shrink: 0;}
.hd-brand {display: flex; align-items: center; gap: .65rem; white-space: nowrap; margin-right: .9rem;}
.hd-brand .lg {display: grid; place-items: center; width: 34px; height: 34px; border-radius: 10px; font-size: 1rem;
               background: var(--hd-brand);}
.hd-brand b {display: block; font-size: 1rem; font-weight: 600; line-height: 1.15; color: var(--hd-text);}
.hd-brand span {display: block; font-size: .74rem; color: var(--hd-muted); margin-top: .05rem;}
/* tabs */
.st-key-topbar .st-key-view {flex: 0 1 auto !important; flex-shrink: 1 !important; min-width: 0; overflow-x: auto;
                             scrollbar-width: none;}
.st-key-topbar .st-key-view::-webkit-scrollbar {display: none;}
.st-key-view [data-testid="stButtonGroup"] {flex-wrap: nowrap !important; gap: .15rem;}
.st-key-view [data-testid="stButtonGroup"] button {
  border: none !important; background: transparent !important; border-radius: 9px !important; margin: 0 !important;
  padding: .45rem .75rem !important; min-height: 0 !important; box-shadow: none !important; color: var(--hd-muted) !important;}
.st-key-view [data-testid="stButtonGroup"] button p {font-size: .93rem; font-weight: 500; color: inherit !important;}
.st-key-view [data-testid="stButtonGroup"] button [data-testid="stIconMaterial"] {font-size: 1.2rem; color: inherit;}
.st-key-view [data-testid="stButtonGroup"] button:hover {color: var(--hd-text) !important; background: var(--hd-surface-2) !important;}
.st-key-view [data-testid="stButtonGroup"] button[aria-checked="true"] {color: var(--hd-accent-text) !important;
                                                                       background: var(--hd-accent-soft) !important;}
.st-key-view [data-testid="stButtonGroup"] button[aria-checked="true"] p {font-weight: 600;}
/* actions on the right */
.st-key-new_chat button {border-radius: 999px !important; min-height: 38px !important; padding: .3rem 1rem .3rem .8rem !important;}
.st-key-theme_menu button, .st-key-user_menu button, .st-key-loginbar [data-testid="stPopoverButton"] {
  border-radius: 999px !important; min-height: 38px !important; padding: .3rem .8rem !important;
  background: transparent !important; border: 1px solid var(--hd-border) !important; box-shadow: none !important;}
.st-key-theme_menu button:hover, .st-key-user_menu button:hover, .st-key-loginbar [data-testid="stPopoverButton"]:hover {
  background: var(--hd-surface-2) !important; border-color: var(--hd-border-strong) !important; color: var(--hd-text) !important;}
.st-key-theme_menu [data-testid="stMarkdownContainer"],
.st-key-theme_menu [data-testid="stPopoverButton"] > div > div:last-child {display: none !important;}
.st-key-theme_menu button {padding: .3rem .6rem !important;}
/* The bar folds labels into icons when the content area (not the window: the sidebar takes room) gets narrow */
[data-testid="stMainBlockContainer"] {container-type: inline-size; container-name: hdmain;}
@container hdmain (max-width: 1000px) {
  .st-key-topbar:has(.hd-brand.admin) :is(.st-key-new_chat, .st-key-user_menu) [data-testid="stMarkdownContainer"] {display: none !important;}
  .st-key-topbar:has(.hd-brand.admin) .st-key-new_chat button {padding: .3rem .65rem !important;}
}
@container hdmain (max-width: 870px) {
  .st-key-topbar:has(.hd-brand.admin) .st-key-view [data-testid="stButtonGroup"] button [data-testid="stMarkdownContainer"] {display: none !important;}
  .st-key-topbar:has(.hd-brand.admin) .st-key-view [data-testid="stButtonGroup"] button {padding: .45rem .6rem !important;}
  .st-key-topbar:has(.hd-brand.admin) .hd-brand span {display: none;}
}
@container hdmain (max-width: 800px) {
  .st-key-topbar :is(.st-key-new_chat, .st-key-user_menu) [data-testid="stMarkdownContainer"] {display: none !important;}
  .st-key-topbar .st-key-new_chat button {padding: .3rem .65rem !important;}
}
@container hdmain (max-width: 670px) {
  .st-key-topbar .st-key-view [data-testid="stButtonGroup"] button [data-testid="stMarkdownContainer"] {display: none !important;}
  .st-key-topbar .st-key-view [data-testid="stButtonGroup"] button {padding: .45rem .6rem !important;}
  .hd-brand span {display: none;}
}
@container hdmain (max-width: 480px) {.hd-brand b {display: none;} .hd-brand {margin-right: .2rem;}}
@media (max-width: 640px) {.block-container {padding: 0 1rem 6rem !important;}}

/* ---------- welcome panel: the one bold element ---------- */
.hd-panel {position: relative; isolation: isolate; overflow: hidden; display: grid; grid-template-columns: 1fr auto;
  align-items: center; gap: 2rem; padding: 2.3rem 2.5rem; margin-bottom: 2rem; border-radius: 20px;
  color: var(--hd-on-accent); background: var(--hd-panel); box-shadow: var(--hd-panel-shadow);}
.hd-panel::before {content: ""; position: absolute; inset: 0; z-index: -1; pointer-events: none;
  background-image: linear-gradient(rgba(255,255,255,.08) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.08) 1px, transparent 1px);
  background-size: 28px 28px;
  -webkit-mask-image: linear-gradient(100deg, transparent 35%, #000 90%); mask-image: linear-gradient(100deg, transparent 35%, #000 90%);}
.hd-panel .h1 {font-size: 2.15rem; font-weight: 600; letter-spacing: -.022em; line-height: 1.15; color: inherit; margin: 0;}
.hd-panel p {margin: .65rem 0 0; max-width: 36rem; font-size: 1.03rem; line-height: 1.55; color: inherit; opacity: .9;}
.hd-panel .status {display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.4rem;}
.hd-panel .status > span {display: inline-flex; align-items: center; gap: .45rem; font-size: .8rem; padding: .3rem .75rem;
  border-radius: 999px; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.2);}
.hd-panel .glyph {display: grid; place-items: center; width: 104px; height: 104px; border-radius: 26px; font-size: 2.8rem;
  background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.24);}
.hd-dot {width: 7px; height: 7px; border-radius: 50%; background: #4ade80; box-shadow: 0 0 0 3px rgba(74,222,128,.28);}
.hd-dot.off {background: #fbbf24; box-shadow: 0 0 0 3px rgba(251,191,36,.28);}
@media (max-width: 760px) {
  .hd-panel {grid-template-columns: 1fr; padding: 1.5rem 1.4rem;} .hd-panel .glyph {display: none;}
  .hd-panel .h1 {font-size: 1.6rem;}
}
.hd-role {display: inline-block; font-size: .72rem; font-weight: 600; padding: .12rem .5rem; border-radius: 6px;
          background: var(--hd-accent-soft); color: var(--hd-accent-text);}

/* ---------- headings ---------- */
.hd-page {display: flex; align-items: center; gap: .9rem; margin: .1rem 0 1.4rem;}
.hd-page .ic {display: grid; place-items: center; width: 44px; height: 44px; border-radius: 12px; font-size: 1.3rem;
              background: var(--hd-accent-soft);}
.hd-page .tt {font-size: 1.55rem; font-weight: 600; letter-spacing: -.018em; line-height: 1.2; color: var(--hd-text);}
.hd-page .st {font-size: .9rem; color: var(--hd-muted); margin-top: .15rem;}
.hd-section {margin: 2rem 0 .8rem;}
.hd-section .t {font-size: 1.08rem; font-weight: 600; color: var(--hd-text); letter-spacing: -.005em;}
.hd-section .s {font-size: .87rem; color: var(--hd-muted); margin-top: .1rem;}
.hd-label {font-size: .82rem; font-weight: 600; color: var(--hd-muted); margin: 1rem 0 .45rem;}

/* ---------- chat home ---------- */
.st-key-topics [data-testid="stButton"] button {
  min-height: 86px; width: 100%; justify-content: flex-start; align-items: flex-start; text-align: left;
  padding: .9rem 1rem; border-radius: 14px; border: 1px solid var(--hd-border); background: var(--hd-surface);
  box-shadow: none; transition: border-color .15s, background-color .15s;
}
.st-key-topics [data-testid="stButton"] button:hover {border-color: var(--hd-accent); background: var(--hd-surface-2);}
.st-key-topics [data-testid="stButton"] button p {text-align: left; font-size: .84rem; color: var(--hd-muted); margin: 0; line-height: 1.45;}
.st-key-topics [data-testid="stButton"] button p strong {display: block; font-size: .98rem; font-weight: 600;
  color: var(--hd-text); margin-bottom: .15rem;}
.hd-steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem;}
.hd-step {background: var(--hd-surface); border: 1px solid var(--hd-border); border-radius: 14px; padding: 1rem 1.1rem;}
.hd-step b {display: block; font-size: .95rem; font-weight: 600; margin: .55rem 0 .2rem; color: var(--hd-text);}
.hd-step span {font-size: .86rem; color: var(--hd-muted); line-height: 1.5;}
.hd-step i {font-style: normal; display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 8px;
            background: var(--hd-accent-soft); color: var(--hd-accent-text); font-size: .82rem; font-weight: 600;}
.hd-urgent {margin-top: 1rem; font-size: .9rem; padding: .75rem 1rem; border-radius: 12px; color: var(--hd-text);
            background: rgba(239,68,68,.07); border: 1px solid rgba(239,68,68,.22);}
@media (max-width: 760px) {.hd-steps {grid-template-columns: 1fr;}}

/* ---------- chat: one reading column, your own messages on the right ---------- */
[data-testid="stChatMessage"] {animation: hd-in .2s ease-out; background: transparent; padding: .3rem 0;
                               width: 100%; max-width: 880px; margin: 0 auto;}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {justify-content: flex-end;}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > div:first-child {display: none;}
[aria-label="Chat message from user"] {
  flex: 0 1 auto; max-width: 75%; margin-left: auto !important; margin-right: 0 !important;
  padding: .6rem 1rem; border-radius: 18px 18px 4px 18px; background: var(--hd-accent); color: var(--hd-on-accent);
}
[aria-label="Chat message from user"] p, [aria-label="Chat message from user"] li {color: var(--hd-on-accent);}
[aria-label="Chat message from assistant"] {padding: .95rem 1.2rem; border-radius: 4px 16px 16px 16px;
  background: var(--hd-surface); border: 1px solid var(--hd-border);}
[data-testid="stChatInput"] {border-radius: 16px !important; box-shadow: var(--hd-shadow);}
[data-testid="stChatInput"] > div {background: var(--hd-surface) !important;}
[data-testid="stChatInput"]:focus-within > div {border-color: var(--hd-accent) !important;}
.hd-typing {display: inline-flex; align-items: center; gap: .55rem; color: var(--hd-muted); font-size: .88rem;}
.hd-typing span.d {width: 6px; height: 6px; border-radius: 50%; background: var(--hd-accent); display: inline-block;
                   animation: hd-blink 1.2s infinite;}
.hd-typing span.d:nth-child(2) {animation-delay: .15s;} .hd-typing span.d:nth-child(3) {animation-delay: .3s;}
.cite {display: inline-block; font-size: .66rem; font-weight: 600; line-height: 1; vertical-align: super;
  padding: .17rem .38rem; margin: 0 .06rem; border-radius: 5px; background: var(--hd-accent-soft); color: var(--hd-accent-text);}
.hd-thumb {display: block; max-width: 260px; max-height: 200px; border-radius: 12px; margin: .45rem 0 .1rem auto;
           border: 1px solid rgba(255,255,255,.35);}
.hd-attach {display: flex; flex-wrap: wrap; gap: .35rem; justify-content: flex-end; margin-top: .4rem;}
.hd-attach span {font-size: .76rem; padding: .18rem .55rem; border-radius: 999px; background: rgba(255,255,255,.2);}
.hd-voice {font-size: .74rem; font-weight: 600; opacity: .85; margin-bottom: .15rem;}
.hd-rewrite {font-size: .82rem; color: var(--hd-muted); margin-bottom: .45rem;}
.hd-meta {display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin-top: .7rem; font-size: .76rem; color: var(--hd-muted);}
.hd-chip {display: inline-flex; align-items: center; gap: .35rem; padding: .2rem .6rem; border-radius: 999px;
          border: 1px solid var(--hd-border); background: var(--hd-surface-2); font-size: .76rem; color: var(--hd-muted);}
.hd-chip.cited {border-color: transparent; background: var(--hd-accent-soft); color: var(--hd-accent-text);}
.hd-chip.cited b {font-weight: 600;}
.src-card {border: 1px solid var(--hd-border); border-radius: 12px; padding: .7rem .9rem; margin-bottom: .5rem;
           background: var(--hd-surface-2);}
.src-card.cited {border-left: 3px solid var(--hd-accent);}
.src-head {font-weight: 600; font-size: .9rem;}
.src-meta {font-size: .74rem; color: var(--hd-muted); margin-top: .1rem;}
.src-body {font-size: .84rem; opacity: .9; margin-top: .35rem; white-space: pre-wrap;}
.st-key-followups [data-testid="stButton"] button {border-radius: 999px; padding: .2rem .85rem; min-height: 0;
  font-size: .82rem; border: 1px solid var(--hd-border); background: var(--hd-surface);}
.st-key-followups [data-testid="stButton"] button:hover {border-color: var(--hd-accent); color: var(--hd-accent-text);}
.hd-foot {text-align: center; font-size: .78rem; color: var(--hd-muted); margin: 1.6rem auto 0; max-width: 620px; line-height: 1.6;}
.hd-foot b, .hd-credit b {color: var(--hd-text); font-weight: 600;}
.st-key-stylebar {gap: .6rem; margin-bottom: .25rem;}
.st-key-stylebar .hd-stylelbl {font-size: .84rem; font-weight: 600; color: var(--hd-muted); white-space: nowrap;}
.st-key-stylebar [data-testid="stButtonGroup"] button {padding: .15rem .75rem !important; min-height: 0 !important;
                                                       font-size: .84rem !important;}

/* ---------- knowledge base ---------- */
[class*="st-key-kbcard_"] {background: var(--hd-surface); border-radius: 14px !important; min-height: 196px;
                           justify-content: space-between;
                           border-color: var(--hd-border) !important; padding: 1rem 1.1rem .8rem !important;}
[class*="st-key-kbcard_"]:hover {border-color: var(--hd-border-strong) !important;}
.hd-card .ic {display: inline-grid; place-items: center; width: 38px; height: 38px; border-radius: 11px;
              background: var(--hd-accent-soft); font-size: 1.15rem; margin-bottom: .6rem;}
.hd-card .tt {font-weight: 600; font-size: 1rem; line-height: 1.3; color: var(--hd-text);}
.hd-card .ss {font-size: .82rem; color: var(--hd-muted); margin-top: .35rem; line-height: 1.5;}
[class*="st-key-kbcard_"] [data-testid="stButton"] button {min-height: 0; padding: .25rem .1rem; font-size: .86rem;}

/* ---------- tickets ---------- */
.hd-ticket {display: flex; gap: .85rem; align-items: flex-start; background: var(--hd-surface); border: 1px solid var(--hd-border);
            border-radius: 14px; padding: .85rem 1rem; margin-bottom: .55rem;}
.hd-ticket .id {font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .78rem; color: var(--hd-muted);}
.hd-ticket .sm {font-weight: 600; font-size: .95rem; margin: .15rem 0; color: var(--hd-text);}
.hd-ticket .mt {display: flex; flex-wrap: wrap; gap: .15rem 1.1rem; font-size: .8rem; color: var(--hd-muted);}
.hd-prio {flex: none; font-size: .72rem; font-weight: 600; padding: .25rem .55rem; border-radius: 8px; white-space: nowrap;}
.hd-prio.p1 {background: rgba(239,68,68,.14); color: #ef4444;}
.hd-prio.p2 {background: rgba(249,115,22,.14); color: #f97316;}
.hd-prio.p3 {background: var(--hd-accent-soft); color: var(--hd-accent-text);}
.hd-prio.p4 {background: rgba(100,116,139,.14); color: var(--hd-muted);}
.hd-status {display: inline-block; font-family: inherit; font-size: .7rem; font-weight: 600; padding: .08rem .5rem;
            border-radius: 999px; margin-left: .4rem; vertical-align: 1px;}
.hd-status.open {background: rgba(59,130,246,.13); color: #3b82f6;}
.hd-status.prog {background: rgba(245,158,11,.15); color: #d97706;}
.hd-status.done {background: rgba(34,197,94,.14); color: #16a34a;}
.hd-status.closed {background: rgba(100,116,139,.14); color: var(--hd-muted);}

/* ---------- metrics, menus, admin ---------- */
[data-testid="stMetric"] {background: var(--hd-surface); border: 1px solid var(--hd-border); border-radius: 14px;
                          padding: .9rem 1.1rem;}
[data-testid="stMetricLabel"] p {font-size: .84rem; color: var(--hd-muted);}
.hd-me {display: flex; align-items: center; gap: .7rem; padding: .15rem 0 .75rem; margin-bottom: .35rem;
        border-bottom: 1px solid var(--hd-border);}
.hd-me .av {flex: none; display: grid; place-items: center; width: 40px; height: 40px; border-radius: 50%; color: #fff;
            font-weight: 600; font-size: .85rem; background: var(--hd-brand);}
.hd-me .who {min-width: 0; flex: 1;}
.hd-me .who b {display: block; font-size: .95rem; font-weight: 600; line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.hd-me .who span {font-size: .78rem; color: var(--hd-muted);}
.hd-credit {text-align: center; font-size: .78rem; color: var(--hd-muted); margin-top: 1rem;}
.hd-note {font-size: .86rem; color: var(--hd-muted); line-height: 1.55; margin-bottom: .6rem;}
.hd-kv {display: grid; grid-template-columns: max-content 1fr; gap: .4rem 1.2rem; font-size: .9rem;
        background: var(--hd-surface); border: 1px solid var(--hd-border); border-radius: 14px; padding: .9rem 1.1rem;}
.hd-kv span {color: var(--hd-muted);}
.hd-preview {border: 1px solid rgba(15,23,42,.1); border-radius: 14px; padding: .65rem; color: #0f172a; margin: .4rem 0 .2rem;}
.hd-preview .pv-bar {display: flex; align-items: center; gap: .45rem; font-size: .8rem; margin-bottom: .55rem;}
.hd-preview .pv-logo {display: grid; place-items: center; width: 22px; height: 22px; border-radius: 7px; font-size: .7rem;}
.hd-preview .pv-tab {font-size: .72rem; padding: .1rem .5rem; border-radius: 7px; border: 1px solid transparent; color: #5d6677;}
.hd-preview .pv-hero {color: #fff; font-weight: 600; font-size: .98rem; padding: 1rem; border-radius: 12px;}
.hd-preview .pv-row {display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin-top: .6rem;}
.hd-preview .pv-btn {font-size: .74rem; font-weight: 600; padding: .3rem .75rem; border-radius: 9px;}
.hd-preview .pv-chip {font-size: .7rem; font-weight: 600; padding: .15rem .55rem; border-radius: 999px;}
.hd-preview .pv-bubble {font-size: .74rem; padding: .3rem .75rem; border-radius: 12px 12px 3px 12px; margin-left: auto;}
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {border-radius: 10px;}
[data-testid="stPopoverBody"] button[data-testid="stBaseButton-tertiary"] {justify-content: flex-start !important;
  padding: .45rem .7rem !important; border-radius: 9px !important; min-height: 0 !important;}
[data-testid="stPopoverBody"] button[data-testid="stBaseButton-tertiary"] > div {justify-content: flex-start !important;}
[data-testid="stPopoverBody"] button[data-testid="stBaseButton-tertiary"]:hover {background: var(--hd-accent-soft) !important;}

/* ---------- left sidebar: previous chats ---------- */
[data-testid="stSidebar"] {background: var(--hd-surface) !important; border-right: 1px solid var(--hd-border) !important;}
[data-testid="stSidebar"] > div {background: transparent !important;}
[data-testid="stSidebarHeader"] {height: 3rem !important; margin-bottom: 0 !important;}
[data-testid="stSidebarUserContent"] {padding-top: 0 !important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .35rem;}
.hd-side-head {font-size: 1.02rem; font-weight: 600; color: var(--hd-text); margin: 0 0 .25rem .2rem;}
.hd-side-empty {font-size: .84rem; color: var(--hd-muted); line-height: 1.5; margin: .6rem .2rem;}
.hd-side-group {font-size: .76rem; font-weight: 600; color: var(--hd-muted); margin: .9rem 0 0 .6rem;}
.st-key-side_new_chat button {justify-content: flex-start !important; border-radius: 10px !important;}
[class*="st-key-chatrow_"] {border-radius: 9px; gap: 0 !important; min-height: 36px;}
[class*="st-key-chatrow_"]:hover {background: var(--hd-surface-2);}
[class*="st-key-chatrow_on_"], [class*="st-key-chatrow_on_"]:hover {background: var(--hd-accent-soft);}
[class*="st-key-chatrow_"] [class*="st-key-open_"] {flex: 1 1 auto !important; min-width: 0 !important; overflow: hidden;}
[class*="st-key-chatrow_"] [class*="st-key-open_"] button {justify-content: flex-start !important; width: 100% !important;
  padding: .45rem .6rem !important; min-height: 0 !important; color: var(--hd-text) !important; overflow: hidden;}
[class*="st-key-chatrow_"] [class*="st-key-open_"] button * {min-width: 0; max-width: 100%;}
[class*="st-key-chatrow_"] [class*="st-key-open_"] button > div, [class*="st-key-chatrow_"] [class*="st-key-open_"] button span {
  justify-content: flex-start !important; text-align: left !important;}
[class*="st-key-chatrow_"] [class*="st-key-open_"] button p {font-size: .88rem; text-align: left; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;}
[class*="st-key-chatrow_on_"] [class*="st-key-open_"] button p {color: var(--hd-accent-text); font-weight: 600;}
[class*="st-key-more_"] {flex: none !important;}
[class*="st-key-more_"] button {padding: .25rem .4rem !important; min-height: 0 !important; border: none !important;
  background: transparent !important; opacity: 0; transition: opacity .12s;}
[class*="st-key-chatrow_"]:hover [class*="st-key-more_"] button, [class*="st-key-chatrow_on_"] [class*="st-key-more_"] button,
[class*="st-key-more_"] button:focus-visible {opacity: 1;}
@media (hover: none) {[class*="st-key-more_"] button {opacity: 1;}}
[class*="st-key-more_"] [data-testid="stMarkdownContainer"],
[class*="st-key-more_"] [data-testid="stPopoverButton"] > div > div:last-child {display: none !important;}

/* ---------- Attach: a paperclip inside the chat box, on its left edge ---------- */
[data-testid="stBottomBlockContainer"] {position: relative;}
[data-testid="stLayoutWrapper"]:has(> .st-key-attach_slot) {position: absolute !important; z-index: 5;
  left: calc(1rem + 10px); bottom: calc(1.3rem + 11px); width: auto !important; height: auto !important;}
.st-key-attach_slot {position: relative; width: auto !important; gap: 0 !important;}
.st-key-attach_menu button {border: none !important; background: transparent !important; box-shadow: none !important;
  min-height: 34px !important; height: 34px; padding: 0 .55rem !important; border-radius: 10px !important; color: var(--hd-muted) !important;}
.st-key-attach_menu button:hover {background: var(--hd-surface-2) !important; color: var(--hd-text) !important;}
.st-key-attach_menu [data-testid="stPopoverButton"] > div > div:last-child {display: none !important;}
.st-key-attach_menu button [data-testid="stIconMaterial"] {font-size: 1.25rem;}
/* "Attach" stays for screen readers; on screen it's the clip, with a count badge when files are waiting */
.st-key-attach_menu button [data-testid="stMarkdownContainer"] {position: absolute; width: 1px; height: 1px;
  overflow: hidden; clip-path: inset(50%);}
.st-key-attach_slot [data-testid="stElementContainer"]:has(.hd-attach-count) {position: absolute; z-index: 6;
  left: 22px; top: -5px; width: auto !important; pointer-events: none;}
.hd-attach-count {display: grid; place-items: center; min-width: 17px; height: 17px; padding: 0 4px; border-radius: 999px;
  font-size: .68rem; font-weight: 700; background: var(--hd-accent); color: var(--hd-on-accent);}
[data-testid="stChatInput"] textarea {padding-left: 2.25rem !important;}
.hd-pop-title {font-size: 1rem; font-weight: 600; color: var(--hd-text); margin-bottom: .2rem;}
.hd-file {display: flex; align-items: baseline; gap: .5rem; font-size: .88rem; min-width: 0;}
.hd-file b {font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.hd-file span {flex: none; font-size: .78rem; color: var(--hd-muted);}
[class*="st-key-attrow_"] [data-testid="stMarkdownContainer"], [class*="st-key-attrow_"] [data-testid="stElementContainer"] {
  margin: 0 !important;}
[class*="st-key-attrow_"] {padding: .3rem .5rem; border: 1px solid var(--hd-border); border-radius: 10px; background: var(--hd-surface-2);}
[class*="st-key-attrm_"] {flex: none !important;}
[class*="st-key-attrm_"] button {padding: .15rem .3rem !important; min-height: 0 !important;}
[class*="st-key-attrm_"] [data-testid="stMarkdownContainer"] {display: none !important;}
[data-testid="stPopoverBody"]:has(.hd-attach-pop) {min-width: min(420px, 92vw);}
/* the drop zone's own one-line list of every file type is replaced by the grouped list below it */
[data-testid="stPopoverBody"]:has(.hd-attach-pop) [data-testid="stFileUploaderDropzoneInstructions"] span {display: none;}
[data-testid="stPopoverBody"]:has(.hd-attach-pop) [data-testid="stFileUploaderDropzoneInstructions"] > div::after {
  content: "or drag and drop files here"; font-size: .86rem; color: var(--hd-muted);}

/* ---------- Streamlit's own widgets, recolored to the portal's color (config.toml only knows one blue) ---------- */
button[data-testid^="stBaseButton-primary"] {background: var(--hd-accent) !important; border-color: var(--hd-accent) !important;
                                             color: var(--hd-on-accent) !important;}
button[data-testid^="stBaseButton-primary"]:hover {background: var(--hd-accent-hover) !important;
                                                   border-color: var(--hd-accent-hover) !important;}
button[data-testid^="stBaseButton-primary"]:disabled {opacity: .4; cursor: not-allowed;}
button[data-testid="stBaseButton-tertiary"]:hover {color: var(--hd-accent-text) !important;}
button[data-testid="stBaseButton-secondary"] {background: var(--hd-surface) !important;}
button[data-testid="stBaseButton-secondary"]:hover {color: var(--hd-accent-text) !important; border-color: var(--hd-accent) !important;}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"] {
  background: var(--hd-accent-soft) !important; border-color: var(--hd-accent) !important;}
[data-testid="stButtonGroup"] button[aria-checked="true"] p, [data-testid="stButtonGroup"] button[aria-pressed="true"] p {
  color: var(--hd-accent-text) !important;}
[data-testid="stCheckbox"]:has(input:checked) label > div:first-of-type {background: var(--hd-accent) !important;
                                                                        border-color: var(--hd-accent) !important;}
[data-testid="stRadio"] input:checked + div {background: var(--hd-accent) !important; border-color: var(--hd-accent) !important;}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color: var(--hd-accent) !important;}
[data-testid="stTab"][aria-selected="true"] p, [data-testid="stTab"]:hover p {color: var(--hd-accent-text) !important;}
[data-testid="stTab"][aria-selected="true"] > div:not([data-testid]) {background: var(--hd-accent) !important;}
[data-testid="stSlider"] {filter: var(--hd-slider-filter);}
a {color: var(--hd-accent-text);}
button:focus-visible, a:focus-visible, [role="tab"]:focus-visible {outline: 2px solid var(--hd-accent) !important;
                                                                   outline-offset: 2px;}
@media (prefers-reduced-motion: reduce) {* {animation: none !important; transition: none !important;}}
</style>
""",
    unsafe_allow_html=True,
)

# Popups (menus, dialogs) live outside the app's root element, so copy its light/dark color-scheme to the page,
# which keeps light-dark() colors right inside them too.
st.html(
    """<div class="hd-js"></div><script>
(() => {
  if (window.__hdSchemeSync) return; window.__hdSchemeSync = true;
  const sync = () => { const app = document.querySelector('.stApp');
    if (app) document.documentElement.style.colorScheme = getComputedStyle(app).colorScheme; };
  const watch = () => { const app = document.querySelector('.stApp'); if (!app) return setTimeout(watch, 200);
    sync(); new MutationObserver(sync).observe(app, {attributes: true, attributeFilter: ['class', 'style']}); };
  watch();
})();
</script>""",
    unsafe_allow_javascript=True,
)


def esc(text) -> str:
    return html.escape(str(text or ""))


def initials(name: str) -> str:
    return "".join(w[0] for w in name.split()[:2]).upper() or "?"


def set_theme(choice: str):
    ss.pending_theme = choice


def theme_menu():
    """Light / Dark / System, using Streamlit's own theme switch (instant, no page reload)."""
    with st.popover("Theme", icon=":material/contrast:", key="theme_menu", help="Theme"):
        st.markdown('<div class="hd-label" style="margin:0 0 .3rem">Theme</div>', unsafe_allow_html=True)
        for label, icon in (("Light", ":material/light_mode:"), ("Dark", ":material/dark_mode:"),
                            ("System", ":material/computer:")):
            st.button(label, icon=icon, key=f"theme_{label}", on_click=set_theme, args=(label,), width="stretch",
                      type="tertiary")


MENU_ACTIONS = ("recordScreencast", "print")


def set_menu_action(action: str):
    ss.pending_menu_action = action


def apply_pending_menu_action():
    """Record screen and Print come from Streamlit's own (hidden) main menu, so click the item there."""
    action = ss.pop("pending_menu_action", None)
    if action not in MENU_ACTIONS:
        return
    st.html(
        f"""<div class="hd-js" data-n="{uuid.uuid4().hex}"></div><script>
(async () => {{
  const d = document, id = "stMainMenuItem-{action}";
  const item = () => d.querySelector(`[data-testid="${{id}}"]`);
  if (!item()) d.querySelector('[data-testid="stMainMenuButton"]')?.click();
  for (let i = 0; i < 40 && !item(); i++) await new Promise(r => setTimeout(r, 25));
  if (item()) item().click();
  else if (id.endsWith("print")) window.print();
}})();
</script>""",
        unsafe_allow_javascript=True,
    )


def apply_pending_theme():
    apply_pending_menu_action()
    choice = ss.pop("pending_theme", None)
    if choice not in ("Light", "Dark", "System"):
        return
    # Clicks the (hidden) theme item in Streamlit's main menu; falls back to the saved setting + reload.
    st.html(
        f"""<div class="hd-js" data-n="{uuid.uuid4().hex}"></div><script>
(async () => {{
  const want = "{choice}", d = document;
  const item = () => d.querySelector(`[data-testid="stMainMenuItem-theme-${{want}}"]`);
  if (!item()) d.querySelector('[data-testid="stMainMenuButton"]')?.click();
  for (let i = 0; i < 40 && !item(); i++) await new Promise(r => setTimeout(r, 25));
  if (!item()) {{
    localStorage.setItem(`stActiveTheme-${{location.pathname}}-v2`, JSON.stringify(want));
    location.reload(); return;
  }}
  item().click();
  await new Promise(r => setTimeout(r, 120));
  if (d.querySelector('[data-testid="stMainMenuPopover"]'))
    d.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Escape', bubbles: true}}));
}})();
</script>""",
        unsafe_allow_javascript=True,
    )


def brand_html(subtitle: str) -> str:
    role = "admin" if IS_ADMIN else "user"
    return (f'<div class="hd-brand {role}"><div class="lg">🛠️</div><div><b>IT Helpdesk</b>'
            f"<span>{esc(subtitle)}</span></div></div>")


PROVIDERS = {"groq": "Groq", "ollama": "Ollama", "openai": "OpenAI"}


def model_label(llm: str) -> str:
    """'groq / openai/gpt-oss-120b' -> 'gpt-oss-120b on Groq'."""
    if not llm or llm.startswith("none"):
        return "the knowledge base only (no AI model)"
    provider, _, model = llm.partition(" / ")
    return f"{model.split('/')[-1] or provider} on {PROVIDERS.get(provider, provider)}"


NAV_ICONS = {CHAT: ":material/chat_bubble:", KB: ":material/menu_book:", TICKETS: ":material/confirmation_number:",
             INSIGHTS: ":material/monitoring:", ADMIN: ":material/admin_panel_settings:"}


def nav_label(view_name: str) -> str:
    return f"{NAV_ICONS[view_name]} {view_name.split(' ', 1)[1]}"


# ---- login page --------------------------------------------------------------


def sign_in(user: dict) -> None:
    ss.user = user
    store.save_auth(ss.sid, user, config.LOGIN_HOURS)
    ss.messages, ss.chat_id = [], None
    ss.view = ss.last_view = CHAT
    ss.just_signed_in = True
    st.rerun()


def login_page() -> None:
    admin = COLORS["admin"]
    st.markdown(
        f"""
<style>
.block-container {{max-width: 1080px;}}
.st-key-auth {{padding: 1.3rem 1.5rem 1.5rem; border-radius: 20px; background: var(--hd-surface);
              border: 1px solid var(--hd-border); box-shadow: var(--hd-pop-shadow);}}
.st-key-auth [data-testid="stTab"] p {{font-weight: 600;}}
.hd-panel.login {{grid-template-columns: 1fr; min-height: 440px; align-content: center; margin: 0;}}
.hd-panel.login ul {{list-style: none; padding: 0; margin: 1.4rem 0 0; display: grid; gap: .55rem;}}
.hd-panel.login li {{display: flex; gap: .6rem; align-items: baseline; font-size: .98rem; opacity: .95;}}
.hd-panel.login li::before {{content: ""; flex: none; width: 6px; height: 6px; border-radius: 50%; background: currentColor;
                             opacity: .7; transform: translateY(-2px);}}
.hd-authnote {{font-size: .86rem; color: var(--hd-muted); margin: .2rem 0 .5rem; line-height: 1.5;}}
.hd-adminnote {{font-size: .86rem; line-height: 1.5; padding: .65rem .8rem; border-radius: 12px; margin: .2rem 0 .6rem;
               background: {theme.alpha(admin, .1)}; border: 1px solid {theme.alpha(admin, .28)};}}
.st-key-admin_login button[data-testid^="stBaseButton-primary"] {{background: {admin} !important;
  border-color: {admin} !important; color: {"#111827" if theme.luminance(admin) > .45 else "#fff"} !important;}}
.st-key-admin_login button[data-testid^="stBaseButton-primary"]:hover {{background: {theme.shade(admin, .12)} !important;}}
</style>
""",
        unsafe_allow_html=True,
    )
    with st.container(key="loginbar", horizontal=True, vertical_alignment="center", wrap=False):
        st.markdown(brand_html(f"by {FOUNDER}"), unsafe_allow_html=True, width="content")
        st.space("stretch")
        theme_menu()
    apply_pending_theme()

    panel_col, auth_col = st.columns([1.1, 1], gap="large", vertical_alignment="center")
    panel_col.markdown(
        '<div class="hd-panel login"><div><div class="h1">Sign in to IT Helpdesk</div>'
        "<p>Step-by-step fixes for everyday IT problems, each linked to the article it came from.</p>"
        "<ul><li>Answers from the company knowledge base</li><li>Help from screenshots, log files and voice notes</li>"
        "<li>Tickets filled in from your chat when you need a person</li></ul>"
        f'<div class="status"><span>Created by {esc(FOUNDER)}</span></div></div></div>',
        unsafe_allow_html=True,
    )

    with auth_col, st.container(key="auth"):
        user_tab, admin_tab = st.tabs(["👤 User", "🛡️ Admin"])
        with user_tab:
            ss.setdefault("auth_mode", "Sign in")
            mode = st.segmented_control("Account", ["Sign in", "Create account"], key="auth_mode",
                                        required=True, label_visibility="collapsed")
            if mode == "Create account":
                st.markdown('<div class="hd-authnote">New here? Create an account to chat with the assistant, '
                            "browse the knowledge base and track your tickets.</div>", unsafe_allow_html=True)
                with st.form("signup", border=False):
                    name = st.text_input("Full name", placeholder="e.g. Priya Verma")
                    username = st.text_input("Username", placeholder="e.g. priya.verma")
                    c1, c2 = st.columns(2)
                    password = c1.text_input("Password", type="password", help=f"At least {auth.MIN_PASSWORD} characters")
                    confirm = c2.text_input("Confirm password", type="password")
                    if st.form_submit_button("Create account", type="primary", width="stretch"):
                        try:
                            sign_in(auth.register(name, username, password, confirm))
                        except auth.AuthError as e:
                            st.error(str(e), icon="⚠️")
            else:
                st.markdown('<div class="hd-authnote">For employees: get answers, raise tickets and follow their '
                            "status.</div>", unsafe_allow_html=True)
                with st.form("user_login", border=False):
                    username = st.text_input("Username", placeholder="Your username")
                    password = st.text_input("Password", type="password")
                    if st.form_submit_button("Sign in", type="primary", width="stretch"):
                        try:
                            sign_in(auth.authenticate(username, password, "user"))
                        except auth.AuthError as e:
                            st.error(str(e), icon="⚠️")

        with admin_tab, st.container(key="admin_login"):
            st.markdown('<div class="hd-adminnote">🛡️ <b>Admin console</b>: usage insights, every ticket and its '
                        "status, user accounts, website colors, and knowledge base maintenance.</div>",
                        unsafe_allow_html=True)
            with st.form("admin_form", border=False):
                username = st.text_input("Admin username", placeholder="admin")
                password = st.text_input("Password", type="password", key="admin_pw")
                if st.form_submit_button("Sign in to the admin console", type="primary", width="stretch"):
                    try:
                        sign_in(auth.authenticate(username, password, "admin"))
                    except auth.AuthError as e:
                        st.error(str(e), icon="⚠️")



if not USER:
    login_page()
    st.stop()

if ss.pop("just_signed_in", False):
    st.toast(f"Welcome{', ' + FIRST_NAME if FIRST_NAME else ''}! You're signed in"
             f"{' to the admin console' if IS_ADMIN else ''}.", icon="🛡️" if IS_ADMIN else "👋")


# ---- data loading ------------------------------------------------------------


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


# ---- callbacks ---------------------------------------------------------------

VIEWS = ADMIN_VIEWS if IS_ADMIN else USER_VIEWS
if ss.view not in VIEWS:
    ss.view = ss.last_view = CHAT


def keep_view_selected():
    # A segmented control can be clicked off; fall back to the previous view.
    if ss.view is None:
        ss.view = ss.last_view
    ss.last_view = ss.view


def keep_style_selected():
    if ss.style is None:
        ss.style = ss.last_style
    ss.last_style = ss.style


def ask(question: str):
    ss.pending = question
    ss.view = ss.last_view = CHAT


def persist_chat():
    """Save the open chat to this account's history (it gets an id with its first message)."""
    if not ss.messages or not USER:
        return
    if not ss.chat_id:
        ss.chat_id = uuid.uuid4().hex
    st.query_params["chat"] = ss.chat_id
    store.save_chat(USER["username"], ss.chat_id, ss.messages)
    ss._saved = (ss.chat_id, len(ss.messages), ss.messages[-1].get("id"))


def new_chat():
    ss.messages, ss.chat_id = [], None
    st.query_params.pop("chat", None)
    ss.view = ss.last_view = CHAT


def open_chat(chat_id: str):
    messages = store.load_chat(USER["username"], chat_id)
    if messages is None:
        st.toast("That chat couldn't be opened. It may have been deleted.", icon="⚠️")
        return
    ss.messages, ss.chat_id = messages, chat_id
    ss._saved = (chat_id, len(messages), messages[-1].get("id") if messages else None)
    st.query_params["chat"] = chat_id
    ss.view = ss.last_view = CHAT


def remove_chat(chat_id: str):
    store.delete_chat(USER["username"], chat_id)
    if chat_id == ss.chat_id:
        new_chat()
    st.toast("Chat deleted", icon="🗑️")


class PendingFile:
    """A file waiting in the Attach menu; looks like Streamlit's UploadedFile to read_attachments()."""

    def __init__(self, name: str, mime: str, data: bytes):
        self.name, self.type, self.data, self.size = name, mime, data, len(data)

    def getvalue(self) -> bytes:
        return self.data


MAX_ATTACHMENTS = 5


def stash_uploads():
    """Move what was just dropped into the Attach menu into our own list, and give the drop zone a fresh start."""
    for f in ss.get(f"uploader_{ss.upload_nonce}") or []:
        if len(ss.attachments) >= MAX_ATTACHMENTS:
            st.toast(f"You can attach up to {MAX_ATTACHMENTS} files per message.", icon="📎")
            break
        if all(a.name != f.name for a in ss.attachments):
            ss.attachments.append(PendingFile(f.name, f.type, f.getvalue()))
    ss.upload_nonce += 1


def remove_attachment(index: int):
    if 0 <= index < len(ss.attachments):
        ss.attachments.pop(index)


def send_attachments():
    ss.send_attachments = True
    ss.view = ss.last_view = CHAT


def sign_out():
    store.clear_auth(ss.sid)
    # A fresh session id, so the next person on this browser starts with an empty chat.
    ss.sid = uuid.uuid4().hex
    st.query_params.clear()
    st.query_params["sid"] = ss.sid
    ss.user, ss.messages, ss.chat_id, ss.attachments = None, [], None, []
    ss.view = ss.last_view = CHAT
    ss.ticket_prefill, ss.ticket_created = {}, None


def save_feedback(msg: dict):
    rating = ss.get(f"fb_{msg['id']}")
    if rating is None:
        return
    store.append(store.FEEDBACK, {
        "message_id": msg["id"],
        "question": msg.get("question", ""),
        "rating": "up" if rating == 1 else "down",
        "user": USER["username"],
    })


def regenerate(msg: dict):
    """Drop the last answer and its question, then ask again (e.g. after switching the answer style)."""
    i = next((j for j, m in enumerate(ss.messages) if m.get("id") == msg["id"]), None)
    if i is None:
        return
    del ss.messages[i]
    if i > 0 and ss.messages[i - 1]["role"] == "user":
        del ss.messages[i - 1]
    ss.pending = msg["question"]


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


def set_ticket_status(ticket_id: str):
    status = ss.get(f"status_{ticket_id}")
    if status and store.update(store.TICKETS, {"id": ticket_id}, {"status": status, "updated_by": USER["username"]}):
        st.toast(f"{ticket_id} is now {status}", icon="✅")


def export_chat() -> str:
    lines = [f"# IT Helpdesk chat, {datetime.now():%Y-%m-%d %H:%M}\n"]
    for m in ss.messages:
        who = (FIRST_NAME or "You") if m["role"] == "user" else "Assistant"
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


def section(title: str, subtitle: str = "") -> None:
    sub = f'<div class="s">{esc(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="hd-section"><div class="t">{esc(title)}</div>{sub}</div>', unsafe_allow_html=True)


def page_heading(icon: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hd-page"><div class="ic">{icon}</div><div><div class="tt">{esc(title)}</div>'
                f'<div class="st">{esc(subtitle)}</div></div></div>', unsafe_allow_html=True)


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
                f'<div class="src-meta"><span class="hd-mono">{html.escape(src["source"])}</span>, relevance {src["score"]:.3f}'
                f'{", cited in the answer" if i in cited else ""}</div>'
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
        st.markdown(f'<div class="hd-rewrite">Understood as: <i>{html.escape(msg["rewritten"])}</i></div>',
                    unsafe_allow_html=True)
    st.markdown(badgify(msg["content"]), unsafe_allow_html=True)
    if msg.get("kind") == "smalltalk":
        st.markdown('<div class="hd-meta"><span class="hd-chip">⚡ Instant reply</span></div>', unsafe_allow_html=True)
        return

    sources, cited = msg.get("sources", []), msg.get("cited", [])
    by_article: dict[str, list[str]] = {}
    for i in cited:
        by_article.setdefault(sources[i - 1]["title"], []).append(str(i))
    chips = [f'<span class="hd-chip cited"><b>{", ".join(nums)}</b>{html.escape(title)}</span>'
             for title, nums in by_article.items()]
    # Which model answered, in which answer style, and how fast
    if msg.get("llm"):
        model = "Knowledge base only" if msg["llm"].startswith("none") else model_label(msg["llm"])
        chips.append(f'<span class="hd-chip hd-mono" title="Model">{esc(model)}</span>')
    if msg.get("style") in ANSWER_STYLES:
        chips.append(f'<span class="hd-chip">{ANSWER_STYLES[msg["style"]][0]} {esc(msg["style"])}</span>')
    if IS_ADMIN and msg.get("mode"):
        k = f", k={msg['k']}" if msg.get("k") else ""
        chips.append(f'<span class="hd-chip hd-mono" title="Retrieval">{esc(msg["mode"])}{k}</span>')
    if msg.get("latency") is not None:
        chips.append(f'<span>{msg["latency"]:.1f}s</span>')
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
        if is_last and not msg.get("attachments"):
            st.button("🔄 Regenerate", key=f"rg_{msg['id']}", on_click=regenerate, args=(msg,), type="tertiary",
                      help="Ask again, e.g. after picking another answer style below")

    if is_last and not msg.get("fallback"):
        suggestions = followup_suggestions(msg)
        if suggestions:
            st.markdown('<div class="hd-label" style="margin-top:.4rem">Related questions</div>', unsafe_allow_html=True)
            with st.container(key="followups", horizontal=True):
                for j, s in enumerate(suggestions):
                    st.button(s, key=f"fu_{msg['id']}_{j}", on_click=ask, args=(f"Tell me about {s}",))


# ---- left sidebar: previous chats ----------------------------------------------


def chat_sidebar() -> None:
    chats = store.list_chats(USER["username"])
    with st.sidebar:
        st.markdown('<div class="hd-side-head">Your chats</div>', unsafe_allow_html=True)
        st.button("New chat", icon=":material/edit_square:", key="side_new_chat", on_click=new_chat, width="stretch")
        if not chats:
            st.markdown('<div class="hd-side-empty">Your conversations will show up here, so you can pick up '
                        "where you left off.</div>", unsafe_allow_html=True)
            return
        query = st.text_input("Search chats", placeholder="Search chats", key="chat_search",
                              label_visibility="collapsed", icon=":material/search:").strip().lower()
        shown = [c for c in chats if not query or query in c.get("title", "").lower()][:60]
        if not shown:
            st.markdown('<div class="hd-side-empty">No chats match your search.</div>', unsafe_allow_html=True)
        for group, items in group_by_day(shown, datetime.now().astimezone().date()):
            st.markdown(f'<div class="hd-side-group">{group}</div>', unsafe_allow_html=True)
            for chat in items:
                state = "on" if chat["id"] == ss.chat_id else "off"
                with st.container(key=f"chatrow_{state}_{chat['id']}", horizontal=True, vertical_alignment="center",
                                  gap=None, wrap=False):
                    st.button(escape_markdown(chat.get("title") or "New chat"), key=f"open_{chat['id']}",
                              on_click=open_chat, args=(chat["id"],), type="tertiary", width="stretch")
                    with st.popover("More", icon=":material/more_horiz:", key=f"more_{chat['id']}", type="tertiary"):
                        st.button("Delete chat", icon=":material/delete:", key=f"del_{chat['id']}",
                                  on_click=remove_chat, args=(chat["id"],), type="tertiary", width="stretch")


chat_sidebar()


# ---- top bar -----------------------------------------------------------------

articles = load_articles()

with st.container(key="topbar", horizontal=True, vertical_alignment="center", wrap=False):
    st.markdown(brand_html("Admin console" if IS_ADMIN else f"by {FOUNDER}"), unsafe_allow_html=True, width="content")
    st.segmented_control("View", VIEWS, key="view", on_change=keep_view_selected, label_visibility="collapsed",
                         format_func=nav_label)
    st.space("stretch")
    st.button("New chat", icon=":material/add:", on_click=new_chat, type="primary", key="new_chat", help="Start a new chat")
    theme_menu()
    with st.popover(FIRST_NAME or USER["username"], icon=":material/account_circle:", key="user_menu"):
        st.markdown(
            f'<div class="hd-me"><div class="av">{esc(initials(USER["name"]))}</div>'
            f'<div class="who"><b>{esc(USER["name"])}</b><span>@{esc(USER["username"])}</span></div>'
            f'<span class="hd-role">{"🛡️ Admin" if IS_ADMIN else "User"}</span></div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            "Export chat", icon=":material/download:", data=export_chat(),
            file_name=f"it-helpdesk-chat-{datetime.now():%Y%m%d-%H%M}.md", mime="text/markdown",
            width="stretch", disabled=not ss.messages,
        )
        st.button("Record screen", icon=":material/videocam:", key="menu_record", on_click=set_menu_action,
                  args=("recordScreencast",), width="stretch",
                  help="Record a video of your screen, e.g. to show IT what went wrong. To stop, press Esc or click "
                       "this again.")
        st.button("Print page", icon=":material/print:", key="menu_print", on_click=set_menu_action,
                  args=("print",), width="stretch")
        if IS_ADMIN:
            st.markdown('<div class="hd-label" style="margin:.7rem 0 .2rem">Retrieval settings</div>',
                        unsafe_allow_html=True)
            st.selectbox("Retrieval mode", MODES, index=0, key="mode",
                         help="hybrid = semantic search + BM25 keyword search, merged with Reciprocal Rank Fusion")
            st.slider("Passages to retrieve (k)", 1, 8, value=config.TOP_K, key="top_k")
            st.toggle("Understand follow-up questions", value=True, key="memory",
                      help="Uses the conversation so far to understand follow-ups like 'what if that fails?'.")
        st.button("Sign out", icon=":material/logout:", on_click=sign_out, width="stretch")
        st.markdown(f'<div class="hd-credit">Created by <b>{esc(FOUNDER)}</b></div>',
                    unsafe_allow_html=True)
apply_pending_theme()

# Widgets inside a popover are only built when it opens, so they get explicit defaults (not preset session values),
# otherwise opening the menu would reset them.
mode, top_k, memory = ((ss.get("mode", MODES[0]), ss.get("top_k", config.TOP_K), ss.get("memory", True)) if IS_ADMIN
                       else ("hybrid", config.TOP_K, True))
view = ss.view or CHAT

try:
    pipeline = get_pipeline()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()
except Exception as e:  # e.g. the search index could not be built
    st.error(f"Could not start the assistant: {e}")
    st.stop()

llm_on = pipeline.llm is not None
llm_label = describe() if llm_on else "none (retrieval only)"
status_html = (
    (f'<span><span class="hd-dot"></span>Assistant online</span>'
     f'<span class="hd-mono">{esc(model_label(llm_label))}</span>') if llm_on
    else '<span><span class="hd-dot off"></span>Answering from the knowledge base only</span>'
) + f"<span>{len(articles)} articles</span>"

if not llm_on:
    if IS_ADMIN:
        st.warning(
            "**No AI model is connected, so answers show the best matching knowledge base passage.**  \n"
            + (f"{pipeline.setup_error}  \n" if pipeline.setup_error else "")
            + "To fix it on Streamlit Cloud: open the app's **Settings → Secrets**, add "
            '`GROQ_API_KEY = "gsk_..."` on its own line, save, and reboot the app.',
            icon="⚠️",
        )
    else:
        st.info("The AI model is offline right now, so answers show the best matching knowledge base passage.",
                icon="ℹ️")


def hero() -> None:
    if IS_ADMIN:
        title = f"Welcome back, {esc(FIRST_NAME or 'admin')}"
        text = ("This is the employee chat, exactly as your team sees it. Insights, tickets, accounts and website "
                "colors are in the tabs above.")
        glyph, extra = "🛡️", "<span>Admin</span>"
    else:
        title = f"Hi {esc(FIRST_NAME)}, what's not working?" if FIRST_NAME else "What's not working?"
        text = ("Describe the problem in your own words, or start from a common topic below. Every answer shows "
                "the knowledge base article it came from.")
        glyph, extra = "🛠️", ""
    st.markdown(
        f'<div class="hd-panel"><div><div class="h1">{title}</div><p>{text}</p>'
        f'<div class="status">{status_html}{extra}</div></div>'
        f'<div class="glyph" aria-hidden="true">{glyph}</div></div>',
        unsafe_allow_html=True,
    )


# ---- chat view ---------------------------------------------------------------


def unpack_submission(submission) -> tuple[str, object, list]:
    """chat_input returns a plain string (topic buttons) or a ChatInputValue with text, files and audio."""
    if isinstance(submission, str):
        return submission.strip(), None, []
    files = list(getattr(submission, "files", None) or [])
    return (submission.text or "").strip(), getattr(submission, "audio", None), files


def read_attachments(text: str, audio, files: list) -> dict:
    """Transcribe the voice note and read each attached screenshot or file into text the RAG chain can use."""
    status_box = st.empty()
    attachments, problems = [], []
    if audio is not None:
        status_box.markdown(TYPING.format("Transcribing your voice note…"), unsafe_allow_html=True)
        try:
            spoken = media.transcribe(audio.getvalue(), audio.name or "voice.wav")
            text = f"{text} {spoken}".strip()
        except Exception as e:
            problems.append(f"The voice note could not be transcribed: {e}")
    for f in files:
        kind = media.kind_of(f.name)
        status_box.markdown(TYPING.format(f"Reading {html.escape(f.name)}…"), unsafe_allow_html=True)
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
    status_box.empty()
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


def file_icon(name: str) -> str:
    return {"image": "🖼️", "pdf": "📕"}.get(media.kind_of(name), "📄")


def attach_menu() -> None:
    """A dropdown for adding screenshots, logs or PDFs to the next message."""
    count = len(ss.attachments)
    if count:
        st.markdown(f'<span class="hd-attach-count" title="{count} file{"s" if count != 1 else ""} ready to send">'
                    f"{count}</span>", unsafe_allow_html=True)
    with st.popover("Attach", icon=":material/attach_file:", key="attach_menu", help="Attach files"):
        st.markdown('<div class="hd-pop-title hd-attach-pop">Attach files</div><div class="hd-note">Add a screenshot '
                    "of the error, a log file or a PDF (up to 10 MB each). It's sent with your next message.</div>",
                    unsafe_allow_html=True)
        st.file_uploader("Add files", type=list(media.FILE_TYPES), accept_multiple_files=True,
                         key=f"uploader_{ss.upload_nonce}", on_change=stash_uploads, label_visibility="collapsed",
                         max_upload_size=10, disabled=count >= MAX_ATTACHMENTS)
        if ss.attachments:
            st.markdown(f'<div class="hd-label" style="margin:.4rem 0 .2rem">Ready to send ({count})</div>',
                        unsafe_allow_html=True)
            for i, f in enumerate(ss.attachments):
                with st.container(key=f"attrow_{i}", horizontal=True, vertical_alignment="center", gap="small",
                                  wrap=False):
                    st.markdown(f'<div class="hd-file">{file_icon(f.name)} <b>{html.escape(f.name)}</b>'
                                f"<span>{max(1, round(f.size / 1024))} KB</span></div>",
                                unsafe_allow_html=True, width="stretch")
                    st.button("Remove", icon=":material/close:", key=f"attrm_{i}", on_click=remove_attachment,
                              args=(i,), type="tertiary", help=f"Remove {f.name}")
            st.button("Send now", icon=":material/send:", key="attach_send", type="primary", width="stretch",
                      on_click=send_attachments, help="Send the files without typing a message")


def chat_view():
    if not ss.messages:
        # The chat area sticks to the bottom as it renders; start the welcome screen at the top instead.
        st.html(
            """<div class="hd-js"></div><script>
(() => { const box = document.querySelector('[data-testid="stAppScrollToBottomContainer"]'); if (!box) return;
  let n = 0; const up = () => { box.scrollTop = 0; if (++n < 8) setTimeout(up, 120); }; up(); })();
</script>""",
            unsafe_allow_javascript=True,
        )
        hero()
        section("Popular topics", "Pick one to start, or describe your own problem in the box below.")
        with st.container(key="topics"):
            for row in (TOPICS[:4], TOPICS[4:]):
                cols = st.columns(4)
                for col, (icon, name, blurb, question) in zip(cols, row):
                    col.button(f"{icon} **{name}**\n\n{blurb}", key=f"topic_{name}", on_click=ask,
                               args=(question,), width="stretch")
        section("How it works")
        st.markdown(
            '<div class="hd-steps">'
            '<div class="hd-step"><i>1</i><b>Describe the issue</b><span>Type it, record a voice note, or attach a '
            "screenshot or log file.</span></div>"
            '<div class="hd-step"><i>2</i><b>Follow the steps</b><span>Each step links to the article it came from. '
            "Choose a quick, step-by-step or detailed answer.</span></div>"
            '<div class="hd-step"><i>3</i><b>Still stuck?</b><span>Raise a ticket that is already filled in from your '
            "chat.</span></div>"
            '</div><div class="hd-urgent">🚨 <b>Urgent, or a security incident?</b> Call the IT Service Desk, '
            "available 24/7.</div>",
            unsafe_allow_html=True,
        )

    for idx, msg in enumerate(ss.messages):
        with st.chat_message(msg["role"], avatar=BOT_AVATAR if msg["role"] == "assistant" else None):
            if msg["role"] == "assistant":
                render_assistant(msg, is_last=idx == len(ss.messages) - 1)
            else:
                render_user(msg)

    st.markdown(
        f'<div class="hd-foot">{esc(config.BOT_NAME)}, created by <b>{esc(FOUNDER)}</b>. Answers come from the IT '
        f"knowledge base using {esc(model_label(llm_label))}, and can be wrong, so check the sources for anything "
        "important.</div>",
        unsafe_allow_html=True,
    )

    # Answer style and the Attach menu, pinned just above the chat box
    with st.bottom, st.container(key="stylebar", horizontal=True, vertical_alignment="center"):
        st.markdown('<span class="hd-stylelbl">Answer style</span>', unsafe_allow_html=True, width="content")
        st.segmented_control(
            "Answer style", list(ANSWER_STYLES), key="style", on_change=keep_style_selected,
            format_func=lambda s: f"{ANSWER_STYLES[s][0]} {s}", label_visibility="collapsed",
        )

    with st.bottom, st.container(key="attach_slot"):
        attach_menu()

    voice = media.groq_available()  # voice notes are transcribed with Whisper on Groq
    submission = st.chat_input(
        f"Describe your IT issue{', ' + FIRST_NAME if FIRST_NAME else ''}"
        + (", or record a voice note…" if voice else "…"),
        accept_audio=voice,
    )
    pending, files_only = ss.pop("pending", None), ss.pop("send_attachments", False)
    if submission is None and pending is None and not files_only:
        return
    if pending is not None and submission is None and not files_only:
        text, audio, files = pending, None, []  # a topic card, follow-up chip or Regenerate
    else:
        text, audio, _ = unpack_submission(submission) if submission is not None else ("", None, [])
        files, ss.attachments = list(ss.attachments), []  # whatever is waiting in the Attach menu goes along
    if not text and audio is None and not files:
        return

    # Only real Q&A turns give useful context; greetings would just trigger needless rewrites.
    turns = [t for m in ss.messages if m["role"] == "assistant" and m.get("kind") != "smalltalk"
             for t in ({"role": "user", "content": m["question"]}, m)]
    history = format_history(turns) if memory else ""
    chat_history = history_messages(turns) if memory else []
    with st.chat_message("user"):
        user_msg = read_attachments(text, audio, files)
        render_user(user_msg)
    if not user_msg["content"]:
        return  # nothing usable (e.g. the voice note could not be transcribed); the warning stays on screen
    ss.messages.append(user_msg)
    question, attachments = user_msg["content"], user_msg["attachments"]
    style = ss.get("style") or DEFAULT_STYLE

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        reply = None if attachments else small_talk_reply(
            question, name=USER["name"], founder=FOUNDER, bot=config.BOT_NAME, model=llm_label if llm_on else "",
        )
        if reply:
            ss.messages.append({"role": "assistant", "content": reply, "kind": "smalltalk", "id": uuid.uuid4().hex,
                                "question": question})
            persist_chat()
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
        placeholder.markdown(TYPING.format(f"Found {len(docs)} passages, writing a {style.lower()} answer…"),
                             unsafe_allow_html=True)

        answer, last_paint = "", 0.0
        try:
            stream = pipeline.stream(media.build_prompt(search_q, attachments), docs,
                                     history=chat_history, user_name=USER["name"], style=style)
            for chunk in stream:
                answer += chunk
                if time.perf_counter() - last_paint > 0.05:  # repainting on every token makes long answers lag
                    placeholder.markdown(answer + " ▌")
                    last_paint = time.perf_counter()
        except Exception as e:
            answer = f"⚠️ The language model could not be reached: {e}"
            if config.LLM_PROVIDER == "ollama":
                answer += (
                    f"\n\nThe app is set to use **Ollama** at `{config.OLLAMA_BASE_URL}`, which is not running here. "
                    "On a cloud host, add `GROQ_API_KEY` (and `LLM_PROVIDER = \"groq\"`) to the app's secrets."
                )
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
        "k": top_k,
        "llm": llm_label,
        "style": style,
        "fallback": fallback,
        "attachments": [{"name": a["name"], "summary": a["summary"]} for a in attachments],
    }
    ss.messages.append(msg)
    persist_chat()
    store.append(store.QUERIES, {
        "question": question,
        "search_query": search_q,
        "attachments": [a["kind"] for a in attachments],
        "answered": not fallback,
        "top_article": sources[0]["title"] if sources and not fallback else None,
        "latency": round(latency, 2),
        "mode": mode,
        "style": style,
        "llm": llm_label,
        "user": USER["username"],
    })
    st.rerun()


# ---- knowledge base view -----------------------------------------------------


@st.dialog("Knowledge base article", width="large")
def show_article(a: dict):
    st.markdown(f"### {ARTICLE_ICONS.get(a['file'], '📄')} {a['title']}")
    st.caption(f"`{a['file']}`, {len(a['sections'])} sections")
    st.markdown(re.sub(r"^## ", "#### ", a["body"], flags=re.MULTILINE))
    if st.button(f"💬 Ask about {a['title']}", type="primary", width="stretch"):
        ask(f"Give me an overview of {a['title']}")
        st.rerun()


def kb_view():
    page_heading("📚", "Knowledge base", "Every IT article the assistant answers from. Read one, or ask about it.")
    query = st.text_input("Search articles", placeholder="🔍  Search articles, e.g. printer, recovery key, Teams audio",
                          label_visibility="collapsed")
    q = query.lower().strip()
    shown = [a for a in articles if not q or q in a["title"].lower() or q in a["body"].lower()]
    st.markdown(f'<div class="hd-label">{len(shown)} of {len(articles)} articles</div>', unsafe_allow_html=True)
    if not shown:
        st.info("No articles match. Try another word, or ask the assistant in the Chat tab.", icon="🔍")
    for row in range(0, len(shown), 3):
        cols = st.columns(3)
        for col, a in zip(cols, shown[row:row + 3]):
            with col, st.container(border=True, key=f"kbcard_{a['file'].removesuffix('.md')}"):
                sections = ", ".join(a["sections"][:3]) + (" and more" if len(a["sections"]) > 3 else "")
                st.markdown(
                    f'<div class="hd-card"><div class="ic">{ARTICLE_ICONS.get(a["file"], "📄")}</div>'
                    f'<div class="tt">{html.escape(a["title"])}</div>'
                    f'<div class="ss">{html.escape(sections)}</div></div>',
                    unsafe_allow_html=True,
                )
                with st.container(horizontal=True, gap="medium"):
                    if st.button("Read", key=f"read_{a['file']}", type="tertiary", icon=":material/article:"):
                        show_article(a)
                    st.button("Ask about it", key=f"ask_{a['file']}", on_click=ask, type="tertiary",
                              icon=":material/chat_bubble:", args=(f"Give me an overview of {a['title']}",))


# ---- tickets view ------------------------------------------------------------


def ticket_card(t: dict) -> str:
    created = datetime.fromisoformat(t["ts"]).astimezone().strftime("%d %b %Y, %H:%M")
    first, _ = SLA.get(t["priority"], ("", ""))
    status = t.get("status", "Open")
    raised_by = f" by {esc(t.get('created_by_name') or t.get('created_by') or 'unknown')}" if IS_ADMIN else ""
    return (
        f'<div class="hd-ticket"><span class="hd-prio {PRIORITY_CLASS.get(t["priority"], "p3")}">'
        f'{esc(t["priority"].split()[0])}</span><div>'
        f'<div class="id">{esc(t["id"])}<span class="hd-status {STATUS_CLASS.get(status, "open")}">{esc(status)}</span></div>'
        f'<div class="sm">{esc(t["summary"])}</div>'
        f'<div class="mt"><span>{esc(t["category"])}</span><span>Raised {created}{raised_by}</span>'
        f"<span>First response within {first}</span></div>"
        f"</div></div>"
    )


def tickets_view():
    page_heading("🎫", "Tickets", "Every ticket and its status." if IS_ADMIN
                 else "Raise a ticket when you need a person, and follow its status here.")
    if ss.ticket_created:
        t = ss.ticket_created
        first, resolve = SLA[t["priority"]]
        st.success(f"Ticket **{t['id']}** created ({t['priority']}). First response within **{first}**, "
                   f"target resolution **{resolve}**.", icon="✅")

    pre = ss.ticket_prefill
    titles = [a["title"] for a in articles] + ["Other"]
    # Admins mostly work the queue, so their form starts folded unless a chat answer sent them here.
    with st.expander("➕ Raise a ticket", expanded=not IS_ADMIN or bool(pre)):
        with st.form("ticket", clear_on_submit=True, border=False):
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
                "created_by": USER["username"], "created_by_name": USER["name"],
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
    if not IS_ADMIN:
        tickets = [t for t in tickets if t.get("created_by") == USER["username"]]
    section(f"{'All tickets' if IS_ADMIN else 'Your tickets'} ({len(tickets)})",
            "Change a ticket's status on the right." if IS_ADMIN and tickets else "")
    if not tickets:
        st.caption("No tickets yet.")
        return

    if IS_ADMIN:
        counts = Counter(t.get("status", "Open") for t in tickets)
        status_filter = st.pills(
            "Status", ["All", *TICKET_STATUSES], default="All", key="ticket_filter", label_visibility="collapsed",
            format_func=lambda s: f"{s} ({len(tickets) if s == 'All' else counts.get(s, 0)})",
        ) or "All"
        tickets = [t for t in tickets if status_filter == "All" or t.get("status", "Open") == status_filter]
        for t in tickets[::-1][:40]:
            c1, c2 = st.columns([4, 1.25], vertical_alignment="center")
            c1.markdown(ticket_card(t), unsafe_allow_html=True)
            current = t.get("status", "Open")
            c2.selectbox(
                "Status", TICKET_STATUSES, index=TICKET_STATUSES.index(current) if current in TICKET_STATUSES else 0,
                key=f"status_{t['id']}", on_change=set_ticket_status, args=(t["id"],), label_visibility="collapsed",
            )
        if not tickets:
            st.caption("No tickets with this status.")
    else:
        for t in tickets[::-1][:20]:
            st.markdown(ticket_card(t), unsafe_allow_html=True)


# ---- insights view (admin) ---------------------------------------------------


def insights_view():
    import pandas as pd  # only admins open this view, so don't pay for pandas on every page load

    page_heading("📊", "Insights", "What employees ask, how well the assistant answers, and where the gaps are.")
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
        section("Most asked topics")
        df = pd.DataFrame(topics.most_common(8), columns=["Topic", "Questions"])
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config={"Questions": st.column_config.ProgressColumn(
                "Questions", format="%d", min_value=0, max_value=int(df["Questions"].max()))},
        )

    styles = Counter(q["style"] for q in queries if q.get("style"))
    if styles:
        section("Answer styles people pick")
        st.markdown("\n".join(f"- {ANSWER_STYLES.get(s, ('',))[0]} **{s}**: {n}" for s, n in styles.most_common()))

    gaps = [q for q in queries if not q.get("answered")]
    section("Knowledge gaps", "Questions the knowledge base could not answer: good candidates for new articles."
            if gaps else "")
    if gaps:
        for q in gaps[-8:][::-1]:
            st.markdown(f"- {q['question']}")
    else:
        st.caption("Every question so far was answered from the knowledge base. 🎉")

    downs = [f for f in feedback if f["rating"] == "down"]
    if downs:
        section("Answers rated unhelpful")
        for f in downs[-8:][::-1]:
            st.markdown(f"- {f['question']}")


# ---- admin console -----------------------------------------------------------


def _when(ts: str | None) -> str:
    if not ts:
        return "–"
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%d %b %Y, %H:%M")
    except ValueError:
        return ts


def appearance_panel():
    """Website colors for each portal, picked from presets or a color picker. No code needed."""
    st.markdown('<div class="hd-note">Pick the main color of each portal. It sets the header, buttons, chat bubbles, '
                "chips and page tint, in both light and dark mode. Changes apply to everyone as soon as you save.</div>",
                unsafe_allow_html=True)
    saved = theme.load()
    chosen = {}
    cols = st.columns(2, gap="medium")
    for col, role, title, label in zip(cols, ("user", "admin"), ("👤 Employee portal", "🛡️ Admin console"),
                                       ("Hi there, how can we help?", "Welcome back, admin")):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            preset = st.pills(title, [*theme.PRESETS, theme.CUSTOM], default=theme.preset_name(saved[role]),
                              key=f"preset_{role}", required=True, label_visibility="collapsed")
            if preset == theme.CUSTOM:
                color = st.color_picker("Custom color", value=saved[role], key=f"custom_{role}")
            else:
                color = theme.PRESETS.get(preset, saved[role])
            chosen[role] = color.lower()
            st.markdown(theme.preview_html(color, label), unsafe_allow_html=True)

    changed = chosen != saved
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Save colors", type="primary", width="stretch", disabled=not changed, icon=":material/palette:"):
        theme.save(chosen, USER["username"])
        st.toast("Website colors updated for everyone", icon="🎨")
        st.rerun()
    if c2.button("Reset to defaults", width="stretch", disabled=saved == theme.DEFAULTS):
        theme.reset()
        for role in theme.DEFAULTS:
            ss.pop(f"preset_{role}", None)
            ss.pop(f"custom_{role}", None)
        st.toast("Colors reset to the defaults", icon="↩️")
        st.rerun()


def admin_view():
    import pandas as pd

    page_heading("🛡️", "Admin console", "Website colors, accounts and the assistant's setup.")
    if auth.using_default_admin_password():
        st.warning(
            "You're using the **default admin password**. Set `ADMIN_USERNAME` and `ADMIN_PASSWORD` in `.env` "
            "(or the app's **Settings → Secrets** on Streamlit Cloud) before you share this app.",
            icon="🔒",
        )

    users = auth.list_users()
    tickets = store.read(store.TICKETS)
    queries = store.read(store.QUERIES)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts", len(users))
    c2.metric("Admins", sum(u["role"] == "admin" for u in users))
    c3.metric("Open tickets", sum(t.get("status", "Open") in ("Open", "In progress") for t in tickets))
    c4.metric("Questions asked", len(queries))

    colors_tab, people_tab, system_tab = st.tabs(["🎨 Website colors", "👥 People", "⚙️ Assistant"])
    with colors_tab:
        appearance_panel()

    with people_tab:
        asked = Counter(q.get("user") for q in queries if q.get("user"))
        st.dataframe(
            pd.DataFrame([{
                "Name": u["name"], "Username": f"@{u['username']}",
                "Role": "🛡️ Admin" if u["role"] == "admin" else "👤 User",
                "Questions": asked.get(u["username"], 0),
                "Joined": "built-in" if u["builtin"] else _when(u["created"]),
                "Last sign-in": _when(u["last_login"]),
            } for u in users]),
            hide_index=True, width="stretch",
        )
        manageable = [u for u in users if not u["builtin"] and u["username"] != USER["username"]]
        if manageable:
            names = {u["username"]: u for u in manageable}
            with st.container(border=True):
                st.markdown("**Manage an account**")
                c1, c2 = st.columns([2, 1])
                who = c1.selectbox("Account", list(names), format_func=lambda u: f"{names[u]['name']} (@{u})")
                role = c2.selectbox("Role", auth.ROLES, index=auth.ROLES.index(names[who]["role"]),
                                    format_func=str.capitalize, key=f"role_{who}")
                b1, b2, b3 = st.columns([1, 1.2, 1], vertical_alignment="center")
                if b1.button("Save role", type="primary", width="stretch", disabled=role == names[who]["role"]):
                    auth.set_role(who, role)
                    st.toast(f"@{who} is now {'an admin' if role == 'admin' else 'a user'}", icon="✅")
                    st.rerun()
                sure = b2.checkbox(f"Remove @{who}", key=f"sure_{who}")
                if b3.button("Remove account", width="stretch", disabled=not sure):
                    auth.delete_user(who)
                    st.toast(f"Removed @{who}", icon="🗑️")
                    st.rerun()
        else:
            st.caption("No other accounts yet. People appear here after they create an account on the login page.")

    with system_tab:
        st.markdown(
            '<div class="hd-kv">'
            f"<span>Name</span><b>{esc(config.BOT_NAME)}</b>"
            f"<span>Founder</span><b>{esc(FOUNDER)}</b>"
            f"<span>Language model</span><b>{esc(llm_label)}</b>"
            f"<span>Embeddings</span><b>{esc(config.EMBEDDING_MODEL)}</b>"
            f"<span>Knowledge base</span><b>{len(articles)} articles</b>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("🔄 Rebuild search index", help="Re-read data/kb/ after adding or editing articles"):
            from src.ingest import build_index

            with st.spinner("Re-indexing knowledge base..."):
                n = build_index()
            pipeline_future.clear()
            load_articles.clear()
            st.success(f"Indexed {n} chunks")
        st.markdown(
            '<div class="hd-note">Accounts are saved, with hashed passwords, in <code>logs/users.json</code>, and '
            "website colors in <code>logs/appearance.json</code>. The built-in admin comes from "
            "<code>ADMIN_USERNAME</code> / <code>ADMIN_PASSWORD</code>, so it keeps working even when those files "
            "are reset (for example after a redeploy).</div>",
            unsafe_allow_html=True,
        )


{CHAT: chat_view, KB: kb_view, TICKETS: tickets_view, INSIGHTS: insights_view, ADMIN: admin_view}[view]()

# Save the open chat to this account's history only when it changed, instead of on every click.
if ss.messages and ss.get("_saved") != (ss.chat_id, len(ss.messages), ss.messages[-1].get("id")):
    persist_chat()
