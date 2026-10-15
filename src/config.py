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

# Retrieval
TOP_K = int(os.getenv("TOP_K", 4))
RRF_K = int(os.getenv("RRF_K", 60))  # Reciprocal Rank Fusion constant

# LLM provider: "ollama" (local, free), "groq" (free API tier), "openai", or "none" (retrieval only)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))
