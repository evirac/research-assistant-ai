# 🔍 Research Assistant AI

A **fully local, privacy-first RAG (Retrieval-Augmented Generation) system** built with LangChain, Gemma 4, Ollama, ChromaDB, and FastAPI. Ask questions against your own documents — no API keys, no cloud, no data leaving your machine.

---

## 🎯 What It Does

Upload any PDF documents and ask questions about them. The system retrieves the most relevant sections from your documents and uses a local LLM to generate accurate, cited answers — grounded strictly in your content.

**Example:**
```
POST /ask
{ "question": "What are the three stages of training large language models?" }

→ "According to the document (Source 1, Page 14), LLMs are trained in three stages:
   pretraining, instruction tuning (SFT), and preference alignment (RLHF)..."
```

---

## 🏗️ Architecture

```
User Request (HTTP)
        │
        ▼
  ┌─────────────┐
  │  FastAPI     │  ← /ask  /ask/stream  /ingest
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────┐
  │         LangChain Orchestration      │
  │                                     │
  │  Query ──► Embedding (nomic-embed)  │
  │               │                     │
  │               ▼                     │
  │         ChromaDB Vector Store       │
  │         (MMR Semantic Search)       │
  │               │                     │
  │               ▼                     │
  │      Top-K Relevant Chunks          │
  │      + Source Citations             │
  │               │                     │
  │               ▼                     │
  │      Prompt Engineering             │
  │      (Grounded RAG Template)        │
  │               │                     │
  │               ▼                     │
  │      Gemma 4 via Ollama             │
  │      (Local GPU Inference)          │
  └─────────────────────────────────────┘
         │
         ▼
  Streaming / Full Response
  with Source Citations
```

---

## ✨ Key Features

### 🔒 Fully Local & Private
- **Zero API costs** — no OpenAI, no Anthropic, no external calls
- **Your data never leaves your machine** — ideal for sensitive documents (legal, medical, financial)
- Runs entirely on consumer hardware with GPU acceleration via CUDA

### 🧠 Intelligent Retrieval
- **MMR (Maximal Marginal Relevance)** search — retrieves diverse, relevant chunks instead of redundant ones
- **Smart chunk filtering** — automatically removes bibliography entries, base64 image data, math notation noise, and other PDF artifacts before indexing
- **Source citations** in every response — know exactly which document and page the answer came from

### ⚡ Streaming Responses
- Token-by-token streaming via `/ask/stream` — responses appear in real-time like ChatGPT
- Non-streaming `/ask` endpoint also available for programmatic use

### 📄 Multi-Document Support
- Ingest multiple PDFs at once via `/ingest`
- All documents indexed in a single persistent ChromaDB vector store
- Re-ingest anytime to add new documents

### 🔧 Production-Ready API
- Auto-generated **Swagger UI** at `http://localhost:8000/docs`
- Pydantic request/response validation
- Clean error handling with HTTP status codes

---

## 🤖 Why Local LLMs?

Most RAG tutorials use the OpenAI API. This project deliberately avoids it. Here's why that matters:

| | Cloud API (OpenAI) | This Project (Local) |
|---|---|---|
| **Cost** | Pay per token | Free forever |
| **Privacy** | Data sent to external servers | Never leaves your machine |
| **Internet** | Required | Fully offline capable |
| **Latency** | Network dependent | Local GPU speed |
| **Customization** | Limited | Full control |
| **Rate Limits** | Yes | None |

Running a local LLM also means you understand the full stack — model quantization, GPU memory management, inference optimization — skills that matter in production AI engineering.

---

## 🧩 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **LLM** | Gemma 4 E2B via Ollama | Local language model inference |
| **Embeddings** | nomic-embed-text via Ollama | Text → vector conversion |
| **Vector Store** | ChromaDB | Semantic similarity search |
| **Orchestration** | LangChain | RAG pipeline & prompt management |
| **API** | FastAPI + Uvicorn | REST endpoints + streaming |
| **PDF Parsing** | PyPDFLoader | Document ingestion |
| **Chunking** | RecursiveCharacterTextSplitter | Intelligent text splitting |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed
- NVIDIA GPU recommended (CPU works but slower)

### 1. Clone the repo

```bash
git clone https://github.com/evirac/research-assistant-ai
cd research-assistant-ai
```

### 2. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install langchain langchain-ollama langchain-community langchain-chroma \
            langchain-text-splitters chromadb fastapi uvicorn pypdf pydantic \
            python-dotenv
```

### 3. Pull required models

```bash
ollama pull gemma4:e2b          # LLM (~3GB)
ollama pull nomic-embed-text    # Embeddings (~274MB)
```

### 4. Add your documents

Drop PDF files into the `docs/` folder.

```bash
mkdir docs
# copy your PDFs here
```

### 5. Ingest documents

```bash
python -m app.core.ingestor
```

Output:
```
Loading: your_document.pdf
Total pages loaded: 30
Total chunks: 221 → Valid chunks: 174
Creating embeddings and storing in ChromaDB...
Done. 174 chunks stored in ChromaDB.
```

### 6. Start the API server

```bash
python main.py
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 📡 API Endpoints

### `POST /ask`
Standard question → answer.

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is pretraining in LLMs?"}'
```

```json
{
  "question": "What is pretraining in LLMs?",
  "answer": "Based on Source 1 (Page 14), pretraining is the first stage of LLM training where the model learns to predict the next word across enormous text corpora using cross-entropy loss..."
}
```

### `POST /ask/stream`
Streaming response — tokens appear in real-time.

```bash
curl -X POST "http://localhost:8000/ask/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain temperature sampling"}' \
  --no-buffer
```

### `POST /ingest`
Re-ingest all PDFs from the `docs/` folder.

```bash
curl -X POST "http://localhost:8000/ingest"
```

### `GET /`
Health check.

```json
{ "status": "running", "model": "gemma4:e2b" }
```

---

## 📁 Project Structure

```
research-assistant-ai/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI endpoints
│   └── core/
│       ├── __init__.py
│       ├── llm.py             # Ollama LLM configuration
│       ├── ingestor.py        # PDF loading, chunking, embedding
│       └── rag.py             # RAG chain, retrieval, prompting
├── docs/                      # Drop your PDFs here
├── chroma_db/                 # Auto-generated vector store (gitignored)
├── main.py                    # Entry point
└── requirements.txt
```

---

## 🔬 Engineering Decisions

**Why Gemma 4 E2B over larger models?**
During development, I benchmarked Gemma 4 26B A4B (MoE) vs E2B. The 26B model produced marginally better answers but took 70-80 seconds per response. E2B runs in 9-10 seconds with quality sufficient for RAG tasks where retrieval does most of the heavy lifting. For a development/demo system, iteration speed matters more than marginal quality gains.

**Why MMR retrieval over simple similarity search?**
Simple cosine similarity often returns 5 chunks that are nearly identical — all from the same paragraph. MMR (Maximal Marginal Relevance) balances relevance with diversity, fetching 20 candidates and selecting the 5 that best cover different aspects of the query. This produces more comprehensive answers.

**Why chunk filtering?**
Raw PDF extraction includes bibliography entries, base64-encoded figure data, mathematical notation fragments, and figure captions. These chunks score high in similarity searches for generic queries but contain no useful answer content. Filtering them out before indexing is a simple improvement that significantly improves retrieval precision.

**Why local embeddings (nomic-embed-text)?**
Most tutorials use OpenAI's embedding API. Using a local embedding model means the entire pipeline — from document ingestion to answer generation — runs without any external network calls. nomic-embed-text is a strong open-source embedding model specifically optimized for retrieval tasks.

---

## 🗺️ Roadmap

- [ ] Conversation memory (multi-turn chat with context)
- [ ] Source citation highlighting in responses
- [ ] Web UI (React frontend)
- [ ] Support for `.txt`, `.docx`, `.md` file formats
- [ ] Docker containerization
- [ ] Evaluation pipeline (retrieval precision metrics)

---

## 📋 Requirements

```
langchain
langchain-ollama
langchain-community
langchain-chroma
langchain-text-splitters
chromadb
fastapi
uvicorn
pypdf
pydantic
python-dotenv
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with LangChain · Ollama · Gemma 4 · ChromaDB · FastAPI*
