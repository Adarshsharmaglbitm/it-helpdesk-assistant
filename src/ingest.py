"""Build the search indexes from the Markdown knowledge base.

Usage:
    python -m src.ingest

Steps: load .md files -> split by headings -> split long sections into overlapping
chunks -> embed with a local sentence-transformer (ONNX via fastembed) -> save a FAISS index, plus the
raw chunks as JSON for the BM25 keyword index.
"""

import json
import shutil
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src import config


def load_markdown_files(kb_dir: Path) -> list[Document]:
    files = sorted(kb_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")
    return [
        Document(page_content=f.read_text(encoding="utf-8"), metadata={"source": f.name})
        for f in files
    ]


def split_documents(docs: list[Document]) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")],
        strip_headers=True,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in docs:
        sections = header_splitter.split_text(doc.page_content)
        for section in sections:
            section.metadata["source"] = doc.metadata["source"]
        for piece in size_splitter.split_documents(sections):
            text = piece.page_content.strip()
            if not text:
                continue
            meta = piece.metadata
            title = meta.get("title", Path(meta["source"]).stem.replace("_", " ").title())
            section = meta.get("section", "")
            # Prefix the heading path so each chunk carries its context when embedded.
            breadcrumb = f"{title} > {section}" if section else title
            chunk_id = f"{meta['source']}::{len(chunks)}"
            chunks.append(
                Document(
                    page_content=f"{breadcrumb}\n{text}",
                    metadata={
                        "chunk_id": chunk_id,
                        "source": meta["source"],
                        "title": title,
                        "section": section,
                        "text": text,
                    },
                )
            )
    return chunks


class FastEmbeddings(Embeddings):
    """Sentence-transformer embeddings run through ONNX (fastembed).

    Produces the same normalized vectors as sentence-transformers, but loads in about a second
    instead of half a minute because it does not import PyTorch.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        cache_dir = str(config.EMBED_CACHE_DIR)
        try:
            # Use the cached model without asking the Hub for updates, which can stall startup for 20s+.
            self.model = TextEmbedding(model_name, cache_dir=cache_dir, local_files_only=True)
        except Exception:  # not downloaded yet
            self.model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed([text]))).tolist()


def get_embeddings() -> Embeddings:
    return FastEmbeddings(config.EMBEDDING_MODEL)


def build_index() -> int:
    from langchain_community.vectorstores import FAISS

    start = time.time()
    docs = load_markdown_files(config.KB_DIR)
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

    if config.INDEX_DIR.exists():
        shutil.rmtree(config.INDEX_DIR)
    config.INDEX_DIR.mkdir(parents=True)

    print(f"Embedding with {config.EMBEDDING_MODEL} ...")
    store = FAISS.from_documents(chunks, get_embeddings(), ids=[c.metadata["chunk_id"] for c in chunks])
    store.save_local(str(config.FAISS_DIR))

    config.CHUNKS_FILE.write_text(
        json.dumps(
            [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved index to {config.INDEX_DIR} in {time.time() - start:.1f}s")
    return len(chunks)


if __name__ == "__main__":
    build_index()
