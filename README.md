---
title: IT Helpdesk Assistant
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# IT Helpdesk Assistant (RAG)

Created by **Adarsh Sharma**.

A chatbot that answers employee IT questions (VPN, passwords, MFA, Outlook, printers, BitLocker, phishing and more) using **Retrieval-Augmented Generation**. Every answer is grounded in an internal knowledge base and cites its sources. If the answer is not in the knowledge base, the bot says so instead of guessing.

**Stack:** Python · LangChain · FAISS · BM25 · Sentence-Transformers (ONNX via fastembed) · Ollama / Groq / OpenAI · Streamlit

## Features

- **Hybrid retrieval**: semantic search (FAISS + sentence-transformer embeddings) combined with keyword search (BM25), merged with **Reciprocal Rank Fusion**.
- **Heading-aware chunking**: Markdown articles are split by section first, then into overlapping chunks, and each chunk keeps its `Title > Section` path for context.
- **Cited, grounded answers**: the LLM must cite passages like `[1][2]`; the UI shows which sources were used.
- **Hallucination guard**: out-of-scope questions get a fixed "not in the knowledge base" reply.
- **Swappable LLM**: local Ollama (free), Groq (free API tier), OpenAI, or retrieval-only mode.
- **Evaluation suite**: 48 labeled questions measure Hit@k and MRR for each retrieval mode, plus citation and refusal accuracy for the LLM.
- **Login page with two portals**: employees sign in (or create an account) on the **User** tab; IT staff use the **Admin** tab. Passwords are salted and hashed (PBKDF2), and a sign-in survives page reloads for `LOGIN_HOURS`.
- **Streamlit app** with role-based views:
  - **Chat** (everyone): streaming answers with citation badges, source cards, popular-topic shortcuts, 👍/👎 feedback, chat export, and a 🔄 Regenerate button. Every answer shows which model wrote it and in which **answer style**: ⚡ Quick, 🧭 Step-by-step or 📖 Detailed, picked just above the chat box.
  - **Knowledge base** (everyone): search and read every article, or jump straight into a chat about it.
  - **Tickets**: raise a ticket pre-filled from the conversation, with a suggested priority and SLA times (demo, saved locally). Employees see their own tickets; admins see every ticket and set its status (Open, In progress, Resolved, Closed).
  - **Insights** (admin): question volume, answer rate, response time, helpfulness, answer styles, most asked topics and knowledge gaps.
  - **Admin console** (admin): accounts (promote to admin, remove), open tickets, assistant details and index rebuild.
- **Website colors without code**: in **Admin → Website colors**, admins pick the color of the employee portal and of the admin console (presets or any custom color) with a live preview; one color drives the header, buttons, chat bubbles and page tint in light and dark mode. Saved in `logs/appearance.json`; `USER_THEME_COLOR` / `ADMIN_THEME_COLOR` set the defaults.
- **Clean layout**: IBM Plex Sans throughout, and a pinned one-row top bar (logo, tabs, New chat, Theme, account menu) that collapses to icons on small screens instead of a sidebar; your own messages sit on the right, and Light / Dark / System switch instantly without reloading the page.
- **Previous chats**: a left sidebar lists each person's earlier conversations (Today, Yesterday, Previous 7 days, Older), with search, reopen and delete. Saved per account in `logs/chats/`.
- **Attach menu**: screenshots, log files and PDFs are added from an Attach dropdown above the chat box, listed before sending, and sent with the next message (or on their own with Send now).
- **Fast start**: the search index, embedding model and LLM client load in the background from the first page view, so the app is ready by the time someone signs in.
- **Conversational answers**: the model sees the last few turns, addresses the person by first name, asks a clarifying question when a message is too vague, and puts urgent security steps first. Questions like *"who made you?"* get an instant answer naming the founder.
- **Follow-up questions**: the conversation is used to rewrite follow-ups like "what if that fails?" into a standalone question before searching.
- **Voice, screenshots and files in the chat**: record a voice note (transcribed with Whisper on Groq), attach a screenshot of an error (read by a vision model), or attach a log, text file or PDF. Their content is used both to search the knowledge base and to answer.
- **Small talk handling**: greetings and thanks get a friendly reply instead of a "not in the knowledge base" answer.

## Architecture

```mermaid
flowchart LR
    A[Markdown KB articles] --> B[Header + recursive chunking]
    B --> C[Sentence-Transformer embeddings]
    C --> D[(FAISS index)]
    B --> E[(BM25 index)]
    Q[User question] --> D & E
    D --> F[Reciprocal Rank Fusion]
    E --> F
    F --> G[Top-k passages]
    G --> H[LLM with citation prompt]
    H --> I[Answer + sources in Streamlit]
```

## Project structure

```
it-helpdesk-rag/
├── app.py                # Streamlit chat UI
├── data/kb/              # 14 IT knowledge base articles (Markdown)
├── src/
│   ├── auth.py           # accounts: sign-up, sign-in, roles (hashed passwords)
│   ├── theme.py          # website colors: full palette from one accent color
│   ├── config.py         # settings (env-overridable)
│   ├── ingest.py         # load -> chunk -> embed -> save FAISS + chunks
│   ├── retrieval.py      # dense, BM25 and hybrid (RRF) retrieval
│   ├── llm.py            # Ollama / Groq / OpenAI factory
│   ├── media.py          # voice transcription, screenshot reading, file text extraction
│   ├── rag.py            # prompt + chain + citations
│   ├── utils.py          # RRF, tokenizer, citation parsing
│   └── metrics.py        # Hit@k, MRR
├── eval/
│   ├── questions.json    # 43 in-scope + 5 out-of-scope labeled questions
│   └── evaluate.py       # retrieval + LLM evaluation
└── tests/test_core.py    # unit tests
```

## Setup

**1. Create a virtual environment and install packages** (Python 3.10+)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**2. Choose an LLM**

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

- **Ollama (free, local, default):** install from [ollama.com](https://ollama.com), then run `ollama pull llama3.2`.
- **Groq (free API, fastest setup on a low-RAM laptop):** get a key at [console.groq.com](https://console.groq.com), then in `.env` set `LLM_PROVIDER=groq` and `GROQ_API_KEY=...`.

**Set the admin password.** In `.env` (or the app's **Secrets** on Streamlit Cloud) set `ADMIN_USERNAME` and `ADMIN_PASSWORD`. Without them the admin account is `admin` / `admin123`, and the admin console shows a warning until you change it.

**3. Build the index** (downloads the ~90 MB embedding model the first time, into `.cache/`)

```bash
python -m src.ingest
```

**4. Run the app**

```bash
streamlit run app.py
```

Open http://localhost:8501, create an account on the **User** tab (or sign in on the **Admin** tab), and ask something like *"My VPN says authentication failed"*.

You can also query from the terminal:

```bash
python -m src.rag "How do I get my BitLocker recovery key?"
```

## Evaluation

```bash
python -m eval.evaluate          # retrieval: Hit@1/3/5 and MRR@5 for dense, BM25, hybrid
python -m eval.evaluate --llm    # + citation rate, source accuracy, refusal accuracy
pytest                           # unit tests
```

Results are saved to `eval/results.json`. Paste your numbers here after running:

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---|---|---|---|
| dense | | | | |
| bm25 | | | | |
| hybrid | | | | |

**Ideas to improve the scores:** try `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`, tune `CHUNK_SIZE`, add a cross-encoder re-ranker, or add more labeled questions.

## Using your own documents

Drop `.md` files into `data/kb/` (use `#` for the title and `##` for sections), then re-run `python -m src.ingest` or click **Rebuild index** in the app.

## Possible extensions

- Cross-encoder re-ranking (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- PDF and Confluence/SharePoint loaders
- A FastAPI endpoint and Docker image
- Auto-create a support ticket when the bot cannot answer
