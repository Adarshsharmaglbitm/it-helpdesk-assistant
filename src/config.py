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
TOP_K = int(os.getenv("TOP_K", 4))
RRF_K = int(os.getenv("RRF_K", 60))  # Reciprocal Rank Fusion constant

# LLM provider: "ollama" (local, free), "groq" (free API tier), "openai", or "none" (retrieval only)
def _default_provider() -> str:
    # Pick whichever provider has a key, so a cloud deploy with only GROQ_API_KEY set still works.
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or _default_provider()).lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Only used by reasoning models such as openai/gpt-oss-*: "low", "medium" or "high"
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))

# Local logs for the app's Insights and Tickets tabs (query log, answer feedback, demo tickets)
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT_DIR / "logs"))

# Chat attachments (voice notes and screenshots use Groq, so they need GROQ_API_KEY)
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "")  # empty = auto-detect, or e.g. "en"
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
MAX_ATTACHMENT_CHARS = int(os.getenv("MAX_ATTACHMENT_CHARS", 6000))
