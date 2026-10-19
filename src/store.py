"""Tiny append-only JSONL store for the app's query log, answer feedback and demo tickets."""

import json
import re
import uuid
from datetime import datetime, timezone

from src import config

QUERIES = "queries"
FEEDBACK = "feedback"
TICKETS = "tickets"


def _path(name: str):
    return config.LOG_DIR / f"{name}.jsonl"


def append(name: str, record: dict) -> dict:
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _path(name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:  # read-only deploys still work, they just don't keep history
        pass
    return record


def read(name: str) -> list[dict]:
    path = _path(name)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def new_ticket_id() -> str:
    return "INC" + uuid.uuid4().hex[:7].upper()


# --- per-browser chat sessions (so a page reload, e.g. a theme switch, keeps the conversation) ---

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _session_path(session_id: str):
    if not _SESSION_ID_RE.match(session_id or ""):
        return None  # ids come from the URL, so never build a path from anything else
    return config.LOG_DIR / "sessions" / f"{session_id}.json"


def save_session(session_id: str, messages: list[dict]) -> None:
    path = _session_path(session_id)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def load_session(session_id: str) -> list[dict]:
    path = _session_path(session_id)
    if path is None or not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
