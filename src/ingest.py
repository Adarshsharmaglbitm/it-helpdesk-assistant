"""Build the search indexes from the Markdown knowledge base.

Usage:
    python -m src.ingest

Steps: load .md files -> split by headings -> split long sections into overlapping
chunks -> embed with a local sentence-transformer -> save a FAISS index, plus the
raw chunks as JSON for the BM25 keyword index.
"""

import json
import shutil
import time
from pathlib import Path

from langchain_core.documents import Document
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


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


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
