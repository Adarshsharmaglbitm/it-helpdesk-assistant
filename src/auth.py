"""Accounts for the login page.

- The admin account comes from ADMIN_USERNAME / ADMIN_PASSWORD (.env or Streamlit secrets), so it survives redeploys.
- Users create an account on the login page. Passwords are salted and hashed with PBKDF2, then saved in
  logs/users.json. Nothing is stored in plain text.
- Admins can promote a user to admin, or remove an account, from the Admin console.
"""

import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timezone

from src import config

ROLES = ("user", "admin")
MIN_PASSWORD = 6
DEFAULT_ADMIN_PASSWORD = "admin123"
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._@-]{2,39}$")
_ITERATIONS = 200_000


class AuthError(ValueError):
    """A problem the login page can show to the person as is."""


def _users_file():
    return config.LOG_DIR / "users.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS).hex()


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def load_users() -> dict[str, dict]:
    path = _users_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_users(users: dict) -> None:
    path = _users_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # write-then-rename, so a crash never leaves a half-written file


def _public(username: str, record: dict, role: str) -> dict:
    """What the app keeps about a signed-in person (never the hash)."""
    return {"username": username, "name": record.get("name") or username, "role": role}


def register(name: str, username: str, password: str, confirm: str | None = None) -> dict:
    """Create a user account and return the signed-in user. Raises AuthError with a readable message."""
    name, username = (name or "").strip(), normalize_username(username)
    if not name:
        raise AuthError("Please enter your name.")
    if not USERNAME_RE.match(username):
        raise AuthError("Username must be 3 to 40 characters: letters, numbers, dots, dashes, underscores or @.")
    if len(password or "") < MIN_PASSWORD:
        raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.")
    if confirm is not None and password != confirm:
        raise AuthError("The two passwords don't match.")
    users = load_users()
    if username in users or username == config.ADMIN_USERNAME:
        raise AuthError("That username is already taken. Try signing in instead.")
    salt = secrets.token_hex(16)
    users[username] = {
        "name": name[:60], "role": "user", "salt": salt, "hash": _hash(password, salt),
        "created": _now(), "last_login": _now(),
    }
    try:
        _save_users(users)
    except OSError as e:
        raise AuthError(f"Could not save the account: {e}") from e
    return _public(username, users[username], "user")


def authenticate(username: str, password: str, as_role: str = "user") -> dict:
    """Sign in to the user portal or the admin console. Returns the signed-in user or raises AuthError.

    Any account can open the user portal; only admin accounts can open the admin console.
    """
    if as_role not in ROLES:
        raise AuthError("Unknown portal.")
    username, password = normalize_username(username), password or ""
    if not username or not password:
        raise AuthError("Enter your username and password.")

    if username == config.ADMIN_USERNAME:
        if hmac.compare_digest(password.encode("utf-8"), config.ADMIN_PASSWORD.encode("utf-8")):
            return {"username": username, "name": config.ADMIN_DISPLAY_NAME, "role": as_role}
        raise AuthError("Wrong username or password.")

    users = load_users()
    record = users.get(username)
    if not record or not record.get("salt") or not hmac.compare_digest(_hash(password, record["salt"]), record["hash"]):
        raise AuthError("Wrong username or password.")
    if as_role == "admin" and record.get("role") != "admin":
        raise AuthError("This account doesn't have admin access. Sign in on the User tab, or ask an admin.")
    record["last_login"] = _now()
    try:
        _save_users(users)
    except OSError:
        pass  # read-only deploys can still sign in
    return _public(username, record, as_role)


def still_valid(user: dict | None) -> bool:
    """A remembered sign-in only counts while the account exists and still has the role it signed in with."""
    if not user or user.get("role") not in ROLES:
        return False
    if user.get("username") == config.ADMIN_USERNAME:
        return True
    record = load_users().get(user.get("username", ""))
    return bool(record) and (user["role"] == "user" or record.get("role") == "admin")


# --- admin console ---------------------------------------------------------------


def list_users() -> list[dict]:
    rows = [{"username": config.ADMIN_USERNAME, "name": config.ADMIN_DISPLAY_NAME, "role": "admin",
             "created": None, "last_login": None, "builtin": True}]
    for username, r in sorted(load_users().items(), key=lambda kv: kv[1].get("created") or ""):
        rows.append({"username": username, "name": r.get("name", username), "role": r.get("role", "user"),
                     "created": r.get("created"), "last_login": r.get("last_login"), "builtin": False})
    return rows


def set_role(username: str, role: str) -> None:
    if role not in ROLES:
        raise AuthError("Unknown role.")
    users = load_users()
    if username not in users:
        raise AuthError("The built-in admin can't be changed here." if username == config.ADMIN_USERNAME
                        else "No such user.")
    users[username]["role"] = role
    _save_users(users)


def delete_user(username: str) -> None:
    users = load_users()
    if username not in users:
        raise AuthError("The built-in admin can't be removed." if username == config.ADMIN_USERNAME
                        else "No such user.")
    del users[username]
    _save_users(users)


def using_default_admin_password() -> bool:
    return config.ADMIN_PASSWORD == DEFAULT_ADMIN_PASSWORD
