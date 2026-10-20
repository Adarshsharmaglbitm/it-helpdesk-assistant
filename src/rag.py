"""The RAG pipeline: retrieve relevant chunks, then ask the LLM to answer from them with citations."""

from typing import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src import config
from src.llm import get_llm
from src.retrieval import HybridRetriever
from src.utils import DEFAULT_STYLE, extract_citations, format_context, normalize_citations, style_instructions

FALLBACK_ANSWER = (
    "I couldn't find this in the IT knowledge base. "
    "Please raise a ticket in the IT self-service portal or call the IT Service Desk."
)

SYSTEM_PROMPT = """Your name is {bot_name}. You are a friendly and patient IT helpdesk assistant for company employees.
You were created by {founder}, the founder of this assistant.{user_line}

How to answer:
- Use ONLY the numbered knowledge base passages below for facts and steps. Cite every fact with its passage \
number in square brackets, like [1] or [2][3].
- Use the earlier conversation only to understand what the person means (which device, app or error), \
never as a source of facts.
- Talk to the person directly and warmly, like a helpful colleague: plain language, no jargon, no filler. \
Do not mention "passages" or "the knowledge base text", just cite.
- If the message is too vague to choose a fix (for example "it's not working"), ask one short clarifying \
question and offer the 2 or 3 most likely options from the passages instead of guessing.
- If the issue is urgent or a security risk (phishing, a lost or stolen device, a suspected hack), say so \
first and put the most urgent step first.
- Answer style: {style}
- If someone asks who made, built or founded you, say you were created by {founder}.
- If the passages do not contain the answer, reply exactly with:
  "{fallback}"
  Do not guess or use outside knowledge.

Knowledge base passages:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), MessagesPlaceholder("history", optional=True), ("human", "{question}")]
).partial(fallback=FALLBACK_ANSWER, bot_name=config.BOT_NAME, founder=config.FOUNDER_NAME)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest message as a single standalone IT support question, "
            "using the conversation only to fill in missing context (which device, app or error). "
            "If it is already standalone, return it unchanged. Reply with the question only.",
        ),
        ("human", "Conversation:\n{history}\n\nLatest message: {question}"),
    ]
)


class RAGPipeline:
    def __init__(self, provider: str | None = None, retriever: HybridRetriever | None = None):
        self.retriever = retriever or HybridRetriever()
        self.llm = get_llm(provider)
        self.chain = PROMPT | self.llm | StrOutputParser() if self.llm else None
        self.condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser() if self.llm else None

    def condense(self, question: str, history: str) -> str:
        """Turn a follow-up like "what if that doesn't work?" into a standalone question for retrieval."""
        if not history or self.condense_chain is None:
            return question
        try:
            rewritten = self.condense_chain.invoke({"history": history, "question": question}).strip()
        except Exception:  # never block an answer because the rewrite failed
            return question
        return rewritten.strip('"') or question

    @staticmethod
    def is_fallback(answer: str) -> bool:
        return FALLBACK_ANSWER[:40] in answer

    def retrieve(self, question: str, k: int = config.TOP_K, mode: str = "hybrid") -> list[Document]:
        return self.retriever.retrieve(question, k=k, mode=mode)

    @staticmethod
    def _inputs(question: str, docs: list[Document], history: list[tuple[str, str]] | None = None,
                user_name: str = "", style: str = DEFAULT_STYLE) -> dict:
        name = " ".join(user_name.split())[:60]
        user_line = (f"\nYou are talking to {name}. Use their first name naturally, at most once per answer."
                     if name else "")
        return {
            "question": question,
            "context": format_context(docs),
            "history": history or [],
            "user_line": user_line,
            "style": style_instructions(style),
        }

    def stream(self, question: str, docs: list[Document], history: list[tuple[str, str]] | None = None,
               user_name: str = "", style: str = DEFAULT_STYLE) -> Iterator[str]:
        """Stream the answer token by token (used by the Streamlit app).

        history:   earlier (role, text) turns, so follow-ups are answered in context
        user_name: the signed-in person's name, so the answer can address them
        style:     one of utils.ANSWER_STYLES ("Quick", "Step-by-step", "Detailed")
        """
        if not docs:
            yield FALLBACK_ANSWER
            return
        if self.chain is None:
            yield self._retrieval_only_answer(docs)
            return
        yield from self.chain.stream(self._inputs(question, docs, history, user_name, style))

    def answer(self, question: str, k: int = config.TOP_K, mode: str = "hybrid") -> dict:
        """Full question -> answer call. Returns the answer, all retrieved sources and the cited ones."""
        docs = self.retrieve(question, k=k, mode=mode)
        text = normalize_citations("".join(self.stream(question, docs)))
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

    sys.stdout.reconfigure(encoding="utf-8")  # LLM answers can contain characters cp1252 cannot print
    question = " ".join(sys.argv[1:]) or "How do I reset my password?"
    result = RAGPipeline().answer(question)
    print(f"Q: {question}\n\nA: {result['answer']}\n\nSources:")
    for i, doc in enumerate(result["sources"], start=1):
        print(f"  [{i}] {doc.metadata['source']} > {doc.metadata['section']}")
