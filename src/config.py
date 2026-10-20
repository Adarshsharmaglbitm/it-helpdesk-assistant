"""Central configuration. Values can be overridden with environment variables or a .env file."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = Path(os.getenv("KB_DIR", ROOT_DIR / "data" / "kb"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT_DIR / "index"))
FAISS_DIR = INDEX_DIR / "faiss"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))

# Embeddings (runs locally, free)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_CACHE_DIR = Path(os.getenv("FASTEMBED_CACHE_PATH", ROOT_DIR / ".cache" / "fastembed"))

# Retrieval
RETRIEVAL_MODES = ("hybrid", "dense", "bm25")
TOP_K = int(os.getenv("TOP_K", 4))
RRF_K = int(os.getenv("RRF_K", 60))  # Reciprocal Rank Fusion constant

# LLM provider: "ollama" (local, free), "groq" (free API tier), "openai", or "none" (retrieval only)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_running(url: str) -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 11434), timeout=0.3):
            return True
    except OSError:
        return False


def _default_provider() -> str:
    # Pick whichever provider has a key, so a cloud deploy with only GROQ_API_KEY set still works.
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    # No key: use a local Ollama if one is running, otherwise answer from retrieval only instead of erroring.
    return "ollama" if _ollama_running(OLLAMA_BASE_URL) else "none"


LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or _default_provider()).lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Only used by reasoning models such as openai/gpt-oss-*: "low", "medium" or "high"
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))

# Local logs for the app's Insights and Tickets tabs (query log, answer feedback, demo tickets)
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT_DIR / "logs"))

# Branding: who the assistant says built it
BOT_NAME = os.getenv("BOT_NAME", "IT Helpdesk Assistant")
FOUNDER_NAME = os.getenv("FOUNDER_NAME", "Adarsh Sharma")

# Login. The built-in admin account comes from these settings (set them in .env or the app's secrets).
# Users create their own accounts on the login page; those are saved, hashed, in logs/users.json.
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_DISPLAY_NAME = os.getenv("ADMIN_DISPLAY_NAME", FOUNDER_NAME)
LOGIN_HOURS = float(os.getenv("LOGIN_HOURS", 12))  # how long a sign-in lasts before asking again

# Chat attachments (voice notes and screenshots use Groq, so they need GROQ_API_KEY)
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "")  # empty = auto-detect, or e.g. "en"
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
MAX_ATTACHMENT_CHARS = int(os.getenv("MAX_ATTACHMENT_CHARS", 6000))
