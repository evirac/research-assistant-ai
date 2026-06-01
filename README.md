<div align="center">

# 🔬 Research Assistant AI

**A fully local, privacy-first RAG system for querying your academic documents**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=for-the-badge)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📽️ Demo

<video controls width="800">
    <source src="demo.mp4" type="video/mp4">
</video>

---

## What is this?

Research Assistant AI is a **Retrieval-Augmented Generation (RAG)** pipeline that lets you have multi-turn conversations with your own PDF library — entirely on your local hardware. Drop academic papers into a folder, ingest them once, and then query them with natural language. The assistant retrieves the most relevant passages, reasons over them, and answers with precise inline citations showing which document and page each claim comes from.

Everything runs locally: the LLM via [Ollama](https://ollama.com), the vector database via [ChromaDB](https://www.trychroma.com), and the embeddings via [nomic-embed-text](https://ollama.com/library/nomic-embed-text). There is no network call to any external API at runtime.

---

## ✨ Features

### Core RAG Pipeline
- **Hybrid retrieval** — fetches a large candidate pool (40 chunks), filters by relevance score, and enforces a per-source diversity cap so no single large document floods the context
- **Layout-aware PDF ingestion** — uses PyMuPDF's content-stream order to correctly parse two-column academic preprints that PyPDFLoader scrambles
- **Smart chunk filtering** — drops bibliography lists, figure captions, base64 noise, and header-only lines; preserves math-heavy chunks using Unicode/LaTeX marker detection
- **Structured citations** — every answer includes deduplicated `{file, page, preview}` citations rendered as cards in the UI

### Conversation Memory
- **Multi-turn Q&A** — each conversation has its own ChromaDB collection storing Q&A pairs as vector embeddings
- **Semantic memory retrieval** — past exchanges are retrieved by relevance to the current question, not just recency
- **Auto-summarization** — long conversation histories are summarized by the LLM before being injected into the prompt to avoid context bloat

### Reasoning & Streaming
- **Thinking token support** — uses Ollama's `/api/chat` endpoint with `think: true`, surfacing the model's internal reasoning chain in a collapsible expander
- **Live streaming** — tokens stream directly from the LLM to the UI with a blinking cursor; thinking and answer streams are separated in real time

### Model Management
- **Switch models from the UI** — dropdown in the sidebar pulls the live model list from Ollama; one click switches with no server restart
- **Model-tagged responses** — each assistant message records which model generated it, visible in the citations expander

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  Sidebar: model selector · chat list · knowledge base       │
│  Main: streaming chat · thinking expander · citation cards  │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP (localhost:8000)
┌───────────────────────▼─────────────────────────────────────┐
│                    FastAPI Backend                           │
│  /ask/stream  /models  /conversations  /documents /ingest   │
└──────┬──────────────────────────────────────┬───────────────┘
       │                                      │
┌──────▼──────────┐                  ┌────────▼────────────┐
│  HybridRetriever│                  │   ConversationMemory│
│                 │                  │                     │
│  ChromaDB       │                  │  ChromaDB           │
│  (documents)    │                  │  (memory_{conv_id}) │
│                 │                  │                     │
│  nomic-embed-   │                  │  nomic-embed-text   │
│  text           │                  │  + similarity filter│
└──────┬──────────┘                  └────────┬────────────┘
       │                                      │
       └──────────────┬───────────────────────┘
                      │
              ┌───────▼────────┐
              │  Ollama        │
              │  /api/chat     │
              │  think: true   │
              │  stream: true  │
              │                │
              │  gemma4 / any  │
              │  local model   │
              └────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM inference | [Ollama](https://ollama.com) — local, GPU-accelerated |
| Embeddings | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) via Ollama |
| Vector store | [ChromaDB](https://www.trychroma.com) — persisted on disk |
| Orchestration | [LangChain](https://www.langchain.com) (retriever + splitter) |
| Backend API | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn |
| Frontend | [Streamlit](https://streamlit.io) |
| PDF loading | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io) with layout-aware extraction |

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- A CUDA-capable GPU (recommended; CPU works but is slow)

### 1. Clone & install

```bash
git clone https://github.com/evirac/research-assistant-ai.git
cd research-assistant-ai
pip install -r requirements.txt
```

### 2. Pull models

```bash
# The LLM (pick one — see Model Selection below)
ollama pull gemma4:e2b        # 2B  — fast, lighter hardware
ollama pull gemma4:12b        # 12B — better reasoning, needs more VRAM

# The embedding model (required)
ollama pull nomic-embed-text
```

### 3. Add your PDFs

```bash
# Drop your PDF files into the docs/ folder
cp your_papers/*.pdf docs/
```

### 4. Start the backend

```bash
python main.py
# FastAPI runs at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

### 5. Start the frontend

```bash
# In a separate terminal
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

### 6. Ingest your documents

Click **📂 Ingest PDFs** in the sidebar. This runs once per document set — embeddings are persisted in `chroma_db/` so you won't need to re-ingest unless you add new papers.

---

## 🖥️ Usage

### Querying documents

Type any natural language question into the chat input. The assistant will:
1. Retrieve the most relevant chunks from your document library
2. Pull any relevant past exchanges from conversation memory
3. Reason over the combined context (visible in the 🧠 Reasoning Chain expander)
4. Stream a cited answer back to you

### Switching models

The **🤖 Language Model** section at the top of the sidebar shows your active model and a dropdown of every model you have pulled locally. Select a different one — the change takes effect on the very next question.

To add a new model:
```bash
ollama pull llama3:8b   # or any model on https://ollama.com/library
```
Then click **↻ Refresh** in the sidebar.

### Multi-turn conversations

Each conversation in the left panel maintains its own memory. The assistant uses semantic search over past Q&A pairs — not just the last N messages — so it can recall a relevant earlier exchange even if several turns have passed.

### Managing your knowledge base

| Action | How |
|---|---|
| Add new PDFs | Copy to `docs/` → click Ingest PDFs |
| See what's indexed | Knowledge Base section in sidebar |
| Remove a document | `DELETE /documents/{filename}` via the API |
| Ingest a single file | `POST /ingest/{filename}` via the API |

---

## 📁 Project Structure

```
research-assistant-ai/
├── main.py                   # Entry point — starts FastAPI on :8000
├── streamlit_app.py          # Streamlit UI
├── requirements.txt
├── docs/                     # ← Drop your PDFs here
├── chroma_db/                # Vector store (auto-created, gitignored)
└── app/
    ├── api/
    │   └── routes.py         # All FastAPI endpoints
    └── core/
        ├── llm.py            # Ollama client + model management
        ├── rag.py            # HybridRetriever + RAG pipeline
        ├── memory.py         # ConversationMemory + SessionManager
        └── ingestor.py       # PDF loading, chunking, filtering
```

---

## 🔧 Configuration

Key constants you might want to tune, all in `app/core/`:

**`rag.py` — Retrieval tuning**
```python
_CANDIDATE_POOL  = 40    # raw candidates fetched from ChromaDB
_FINAL_K         = 8     # chunks actually passed to the LLM
_MAX_PER_SOURCE  = 2     # max chunks from any single document
_SCORE_THRESHOLD = 1.1   # L2 distance ceiling (lower = stricter)
```

**`llm.py` — Generation options**
```python
_OPTIONS = {
    "temperature": 0.7,
    "num_predict": 2048,
}
```

**`ingestor.py` — Chunking**
```python
# In split_documents()
chunk_size    = 1500
chunk_overlap = 200
```

---

## 🌐 API Reference

The FastAPI backend exposes a full REST API. Interactive docs are available at **http://localhost:8000/docs** when the server is running.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status + active model |
| `POST` | `/ask/stream` | Stream an answer (main endpoint) |
| `POST` | `/ask` | Non-streaming answer |
| `GET` | `/models` | List available Ollama models |
| `PUT` | `/models/current` | Switch active model |
| `POST` | `/conversations/{id}/ask` | Ask within a conversation |
| `GET` | `/conversations/{id}/history` | Full conversation history |
| `DELETE` | `/conversations/{id}` | Clear conversation memory |
| `POST` | `/ingest` | Ingest all PDFs in /docs |
| `GET` | `/documents` | List indexed documents + chunk counts |
| `DELETE` | `/documents/{filename}` | Remove a document from the index |

---

## 🤔 Design Decisions

**Why direct HTTP to Ollama instead of LangChain's OllamaLLM?**
LangChain's `OllamaLLM` uses `/api/generate` which does not expose thinking tokens. The thinking chain only comes through `/api/chat` with `think: true`. The thin HTTP wrapper in `llm.py` was necessary to capture and surface the model's internal reasoning.

**Why a separate ChromaDB collection per conversation?**
Storing memory in the same collection as documents would require careful metadata filtering on every query. Separate collections (`memory_{conversation_id}`) keep document retrieval and memory retrieval completely independent with no risk of cross-contamination.

**Why the per-source diversity cap?**
Without it, a large paper (200+ chunks) will dominate every retrieval result even for questions it barely addresses. Capping each source at `_MAX_PER_SOURCE = 2` chunks forces the retriever to pull from multiple documents, which produces much more useful answers on cross-paper questions.

---

## 📋 Requirements

```
fastapi
uvicorn
streamlit
langchain
langchain-ollama
langchain-chroma
langchain-community
langchain-text-splitters
chromadb
pymupdf
requests
pydantic
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built with 🔬 for researchers who want AI that stays on their machine.
</div>