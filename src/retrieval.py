"""Dense (FAISS), sparse (BM25) and hybrid (Reciprocal Rank Fusion) retrieval."""

import json

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src import config
from src.utils import reciprocal_rank_fusion, tokenize

MODES = ("hybrid", "dense", "bm25")


class HybridRetriever:
    def __init__(self, embeddings=None):
        from langchain_community.vectorstores import FAISS

        from src.ingest import get_embeddings

        if not config.CHUNKS_FILE.exists():
            raise FileNotFoundError("Index not found. Run `python -m src.ingest` first.")

        self.embeddings = embeddings or get_embeddings()
        # The index is built locally by src/ingest.py, so loading its pickle is safe.
        self.vectorstore = FAISS.load_local(
            str(config.FAISS_DIR), self.embeddings, allow_dangerous_deserialization=True
        )

        raw = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
        self.chunks = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in raw]
        self.by_id = {c.metadata["chunk_id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in self.chunks])

    # --- individual retrievers -------------------------------------------------

    def dense(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Semantic search. Returns (chunk, cosine similarity)."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # FAISS returns squared L2 distance; for unit vectors cos = 1 - d / 2.
        return [(self.by_id[doc.metadata["chunk_id"]], 1 - float(d) / 2) for doc, d in results]

    def sparse(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Keyword search with BM25. Returns (chunk, BM25 score)."""
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]

    def hybrid(self, query: str, k: int, candidates: int = 20) -> list[tuple[Document, float]]:
        """Fuse dense and BM25 rankings with Reciprocal Rank Fusion. Returns (chunk, RRF score)."""
        dense_ids = [d.metadata["chunk_id"] for d, _ in self.dense(query, candidates)]
        sparse_ids = [d.metadata["chunk_id"] for d, _ in self.sparse(query, candidates)]
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=config.RRF_K)
        return [(self.by_id[cid], score) for cid, score in fused[:k]]

    # --- entry point -------------------------------------------------------------

    def retrieve(self, query: str, k: int = config.TOP_K, mode: str = "hybrid") -> list[Document]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        fn = {"hybrid": self.hybrid, "dense": self.dense, "bm25": self.sparse}[mode]
        docs = []
        for doc, score in fn(query, k):
            # Copy so the cached chunk metadata is not mutated between queries.
            docs.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "score": score}))
        return docs
