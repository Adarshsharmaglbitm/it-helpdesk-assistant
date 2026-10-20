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


def update(name: str, match: dict, changes: dict) -> bool:
    """Apply `changes` to the first record whose fields equal `match` (e.g. set a ticket's status)."""
    rows = read(name)
    for row in rows:
        if all(row.get(k) == v for k, v in match.items()):
            row.update(changes)
            break
    else:
        return False
    try:
        _path(name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    except OSError:
        return False
    return True


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


# --- saved chats, per account (the "previous chats" list on the left) -----------------------------

_UNSAFE_NAME_RE = re.compile(r"[^a-z0-9._@-]")
_SPACE_RE = re.compile(r"\s+")


def _chats_dir(username: str):
    return config.LOG_DIR / "chats" / (_UNSAFE_NAME_RE.sub("_", (username or "").lower()).strip(".") or "_")


def _chat_path(username: str, chat_id: str):
    if not _SESSION_ID_RE.match(chat_id or ""):
        return None  # chat ids come from the URL, so never build a path from anything else
    return _chats_dir(username) / f"{chat_id}.json"


def _index_path(username: str):
    return _chats_dir(username) / "index.json"


def _load_index(username: str) -> dict:
    path = _index_path(username)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    return data if isinstance(data, dict) else {}


def _write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def chat_title(messages: list[dict], limit: int = 60) -> str:
    """The first question, shortened, so the chat list reads like a list of problems."""
    first = next((m for m in messages if m.get("role") == "user"), None)
    if not first:
        return "New chat"
    text = _SPACE_RE.sub(" ", first.get("content") or "").strip()
    if not text and first.get("attachments"):
        text = f"Attachment: {first['attachments'][0].get('name', 'file')}"
    text = text or "New chat"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def save_chat(username: str, chat_id: str, messages: list[dict]) -> None:
    path = _chat_path(username, chat_id)
    if path is None or not messages:
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        _write_json(path, messages)
        index = _load_index(username)
        entry = index.get(chat_id) or {"created": now}
        entry.update(title=chat_title(messages), updated=now, count=len(messages))
        index[chat_id] = entry
        _write_json(_index_path(username), index)
    except OSError:
        pass  # read-only deploys still chat, they just don't keep history


def load_chat(username: str, chat_id: str) -> list[dict] | None:
    path = _chat_path(username, chat_id)
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) else None


def list_chats(username: str) -> list[dict]:
    """This account's chats, newest first: [{id, title, created, updated, count}]."""
    chats = [{"id": cid, **entry} for cid, entry in _load_index(username).items()
             if _SESSION_ID_RE.match(cid) and isinstance(entry, dict)]
    return sorted(chats, key=lambda c: c.get("updated", ""), reverse=True)


def delete_chat(username: str, chat_id: str) -> None:
    path = _chat_path(username, chat_id)
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
        index = _load_index(username)
        if index.pop(chat_id, None) is not None:
            _write_json(_index_path(username), index)
    except OSError:
        pass


# --- remembered sign-ins (so a page reload, e.g. a theme switch, doesn't sign the person out) ---


def _auth_path(session_id: str):
    path = _session_path(session_id)
    return path.with_suffix(".auth.json") if path else None


def save_auth(session_id: str, user: dict, hours: float) -> None:
    path = _auth_path(session_id)
    if path is None:
        return
    expires = datetime.now(timezone.utc).timestamp() + hours * 3600
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**user, "expires": expires}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def load_auth(session_id: str) -> dict | None:
    path = _auth_path(session_id)
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("expires", 0) < datetime.now(timezone.utc).timestamp():
        clear_auth(session_id)
        return None
    return {k: data[k] for k in ("username", "name", "role") if k in data}


def clear_auth(session_id: str) -> None:
    path = _auth_path(session_id)
    try:
        if path and path.exists():
            path.unlink()
    except OSError:
        pass
