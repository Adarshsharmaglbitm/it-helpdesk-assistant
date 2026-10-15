"""The RAG pipeline: retrieve relevant chunks, then ask the LLM to answer from them with citations."""

from typing import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src import config
from src.llm import get_llm
from src.retrieval import HybridRetriever
from src.utils import extract_citations, format_context

FALLBACK_ANSWER = (
    "I couldn't find this in the IT knowledge base. "
    "Please raise a ticket in the IT self-service portal or call the IT Service Desk."
)

SYSTEM_PROMPT = """You are an IT helpdesk assistant for company employees.
Answer the question using ONLY the numbered knowledge base passages below.

Rules:
- Cite every fact with the passage number in square brackets, like [1] or [2][3].
- Give clear, step-by-step instructions when the user is trying to fix or set up something.
- Be concise. Do not mention the passages themselves, only cite them.
- If the passages do not contain the answer, reply exactly with:
  "{fallback}"
  Do not guess or use outside knowledge.

Knowledge base passages:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
).partial(fallback=FALLBACK_ANSWER)


class RAGPipeline:
    def __init__(self, provider: str | None = None, retriever: HybridRetriever | None = None):
        self.retriever = retriever or HybridRetriever()
        self.llm = get_llm(provider)
        self.chain = PROMPT | self.llm | StrOutputParser() if self.llm else None

    def retrieve(self, question: str, k: int = config.TOP_K, mode: str = "hybrid") -> list[Document]:
        return self.retriever.retrieve(question, k=k, mode=mode)

    def _inputs(self, question: str, docs: list[Document]) -> dict:
        return {"question": question, "context": format_context(docs)}

    def stream(self, question: str, docs: list[Document]) -> Iterator[str]:
        """Stream the answer token by token (used by the Streamlit app)."""
        if not docs:
            yield FALLBACK_ANSWER
            return
        if self.chain is None:
            yield self._retrieval_only_answer(docs)
            return
        yield from self.chain.stream(self._inputs(question, docs))

    def answer(self, question: str, k: int = config.TOP_K, mode: str = "hybrid") -> dict:
        """Full question -> answer call. Returns the answer, all retrieved sources and the cited ones."""
        docs = self.retrieve(question, k=k, mode=mode)
        text = "".join(self.stream(question, docs))
        cited = extract_citations(text, len(docs))
        return {
            "question": question,
            "answer": text,
            "sources": docs,
            "cited": [docs[i - 1] for i in cited],
        }

    @staticmethod
    def _retrieval_only_answer(docs: list[Document]) -> str:
        top = docs[0].metadata
        return (
            "_No LLM configured, showing the best matching passage._\n\n"
            f"**{top['title']}{' > ' + top['section'] if top.get('section') else ''}** [1]\n\n"
            f"{top['text']}"
        )


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "How do I reset my password?"
    result = RAGPipeline().answer(question)
    print(f"Q: {question}\n\nA: {result['answer']}\n\nSources:")
    for i, doc in enumerate(result["sources"], start=1):
        print(f"  [{i}] {doc.metadata['source']} > {doc.metadata['section']}")
