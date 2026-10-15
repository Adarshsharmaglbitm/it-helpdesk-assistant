"""Streamlit chat UI for the IT Helpdesk RAG assistant.

Run with:  streamlit run app.py
"""

import streamlit as st

from src import config
from src.llm import describe
from src.retrieval import MODES
from src.utils import extract_citations

st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="🛠️", layout="centered")

EXAMPLES = [
    "My VPN says authentication failed, what should I do?",
    "How do I get my BitLocker recovery key?",
    "I clicked a link in a phishing email",
    "How long does a new software request take?",
]


@st.cache_resource(show_spinner="Loading search index and model...")
def load_pipeline():
    from src.rag import RAGPipeline

    # The index is gitignored, so build it on first start (e.g. on a fresh cloud deploy).
    if not config.CHUNKS_FILE.exists():
        from src.ingest import build_index

        build_index()
    return RAGPipeline()


def render_sources(sources: list[dict], cited: list[int]) -> None:
    if not sources:
        return
    label = f"Sources ({len(cited)} cited)" if cited else "Sources"
    with st.expander(label):
        for i, src in enumerate(sources, start=1):
            mark = "✅ " if i in cited else ""
            heading = src["title"] + (f" › {src['section']}" if src["section"] else "")
            st.markdown(f"{mark}**[{i}] {heading}**  \n`{src['source']}` · score {src['score']:.3f}")
            st.caption(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))


# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Retrieval mode",
        MODES,
        help="hybrid = semantic search + BM25 keyword search, merged with Reciprocal Rank Fusion",
    )
    top_k = st.slider("Passages to retrieve (k)", 1, 8, config.TOP_K)
    st.caption(f"**LLM:** {describe()}")
    st.caption(f"**Embeddings:** {config.EMBEDDING_MODEL}")
    if st.button("Rebuild index"):
        from src.ingest import build_index

        with st.spinner("Re-indexing knowledge base..."):
            n = build_index()
        load_pipeline.clear()
        st.success(f"Indexed {n} chunks")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- main --------------------------------------------------------------------
st.title("🛠️ IT Helpdesk Assistant")
st.caption("Ask an IT question. Answers come only from the company knowledge base, with sources.")

try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()
except Exception as e:  # e.g. missing API key or Ollama not running
    st.error(f"Could not start the assistant: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []), msg.get("cited", []))

question = st.chat_input("Describe your IT issue...") or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            docs = pipeline.retrieve(question, k=top_k, mode=mode)
        try:
            answer = st.write_stream(pipeline.stream(question, docs))
        except Exception as e:
            answer = f"⚠️ The language model could not be reached: {e}"
            st.error(answer)

        sources = [
            {**{k: d.metadata.get(k, "") for k in ("title", "section", "source", "text")},
             "score": d.metadata.get("score", 0.0)}
            for d in docs
        ]
        cited = extract_citations(answer, len(docs))
        render_sources(sources, cited)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "cited": cited}
    )
