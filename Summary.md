# Research Assistant AI - Project Summary

## 📋 Overview

**Research Assistant AI** is a fully local, privacy-first RAG (Retrieval-Augmented Generation) system that allows users to upload PDF documents and ask natural language questions about them. The system retrieves relevant document sections and uses a local LLM to generate accurate, cited answers—all without sending any data to the cloud.

---

## 🎯 Project Objectives

- ✅ Build a production-ready RAG system that runs entirely locally
- ✅ Provide privacy-first document analysis (sensitive data never leaves the machine)
- ✅ Eliminate API costs by using open-source models and local inference
- ✅ Enable multi-document querying with proper source citations
- ✅ Deliver token-by-token streaming responses for real-time feedback
- ✅ Create an intuitive REST API with automatic documentation

---

## 🏗️ Architecture

### High-Level Flow

```
User Question (HTTP) → FastAPI → LangChain Orchestration → ChromaDB Retrieval 
→ Prompt Engineering → Gemma 4 LLM (Ollama) → Streaming/Full Response + Citations
```

### Component Breakdown

1. **FastAPI (Web Server)**
   - RESTful API with automatic Swagger UI documentation
   - Endpoints: `/ask`, `/ask/stream`, `/ingest`, `/` (health check)
   - Pydantic validation for requests/responses

2. **LangChain (Orchestration)**
   - Chains together embedding, retrieval, and LLM components
   - Manages the RAG pipeline
   - Handles prompt templating and output parsing

3. **ChromaDB (Vector Database)**
   - Persistent vector store at `chroma_db/`
   - Stores document embeddings for semantic search
   - Supports MMR (Maximal Marginal Relevance) retrieval

4. **Ollama (Local LLM Inference)**
   - Runs Gemma 4 model locally
   - GPU-accelerated inference via CUDA
   - No external API calls

5. **nomic-embed-text (Embeddings)**
   - Lightweight embedding model
   - Converts text chunks into vectors for similarity search

---

## ✨ Key Features Implemented

### 🔒 Full Privacy & Offline Capability
- **Zero external API calls** — no OpenAI, Anthropic, or cloud dependencies
- **All processing local** — documents and queries never leave your machine
- **Offline-capable** — can operate without internet connection
- **Perfect for sensitive data** — legal, medical, financial, proprietary documents

### 🧠 Intelligent Document Retrieval
- **MMR Search** — retrieves diverse, relevant chunks instead of redundant near-duplicates
- **Smart chunk filtering** — removes PDF artifacts (bibliography, base64 images, math notation noise)
- **Multi-document support** — ingest and query across multiple PDFs simultaneously
- **Source citations** — every answer includes document name and page number

### ⚡ Real-Time Streaming
- **Token-by-token streaming** via `/ask/stream` endpoint
- **Non-streaming option** via `/ask` for programmatic use
- **Live response generation** — answers appear in real-time

### 🔧 Production-Ready API
- **Swagger UI documentation** at `http://localhost:8000/docs`
- **Type validation** with Pydantic models
- **Clean error handling** with appropriate HTTP status codes
- **RESTful design** for easy integration

### 📄 Flexible Document Ingestion
- **Batch upload** via `/ingest` endpoint
- **PDF parsing** with automatic chunking
- **Persistent indexing** — documents remain indexed across sessions
- **Re-ingestion support** — add new documents anytime

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | FastAPI | REST API with auto-documentation |
| **LLM Orchestration** | LangChain | RAG pipeline management |
| **Local LLM** | Gemma 4 + Ollama | Inference engine |
| **Vector DB** | ChromaDB | Document embedding storage |
| **Embeddings** | nomic-embed-text | Text-to-vector conversion |
| **API Validation** | Pydantic | Request/response schemas |
| **Language** | Python 3.x | Core implementation |

---

## 📁 Project Structure

```
research-assistant-ai/
├── main.py                          # Entry point, runs debug + API server
├── streamlit_app.py                 # Streamlit UI application
├── README.md                        # User documentation
├── Summary.md                       # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py               # FastAPI endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ingestor.py             # PDF ingestion logic
│   │   ├── llm.py                  # LLM configuration
│   │   ├── memory.py               # Memory management
│   │   └── rag.py                  # RAG pipeline logic
│   └── data/                        # Data processing utilities
├── chroma_db/                       # Vector database persistence
│   ├── chroma.sqlite3
│   └── [embeddings storage]/
├── docs/                            # Input folder for PDF documents
└── [other generated files]/
```

---

## 🚀 Current State & API

### Running the System
```bash
python main.py
# Server starts at http://localhost:8000
# Access Swagger UI at http://localhost:8000/docs
```

### API Endpoints

#### 1. Health Check
```bash
GET /
# Response: {"status": "running", "model": "gemma4:e2b"}
```

#### 2. Ask Question (Full Response)
```bash
POST /ask
Content-Type: application/json

{ "question": "What are the three stages of training large language models?" }

# Response:
{
  "question": "What are the three stages of training large language models?",
  "answer": "According to the document (Source 1: document.pdf, Page 14), LLMs are trained in three stages: pretraining, instruction tuning (SFT), and preference alignment (RLHF)..."
}
```

#### 3. Ask Question (Streaming Response)
```bash
POST /ask/stream
Content-Type: application/json

{ "question": "What are the three stages of training large language models?" }

# Response: Text stream (token-by-token)
# According to the document...
```

#### 4. Ingest Documents
```bash
POST /ingest

# Indexes all PDFs from docs/ folder
# Response: {"status": "success", "message": "Documents ingested successfully"}
```

---

## 💡 How It Works: Query Execution

1. **User submits question** via `/ask` endpoint
2. **Question embedding** — converted to vector using nomic-embed-text
3. **Semantic search** — ChromaDB performs MMR retrieval to find top relevant chunks
4. **Context formatting** — retrieved chunks formatted with source citations
5. **Prompt construction** — question + context inserted into expert prompt template
6. **LLM generation** — Gemma 4 (via Ollama) generates answer
7. **Response streaming** — answer streamed back to client with citations

### Prompt Template Strategy
The system uses a grounded RAG template that:
- Instructs the LLM to use context as PRIMARY source
- Encourages reasoning within the context bounds
- Requires explicit citations for answers
- Specifies "I don't have enough information" protocol

---

## 🔍 Prompt Engineering Details

The RAG prompt template emphasizes:
1. **Context-grounded reasoning** — answers must be based on provided documents
2. **Citation requirements** — every answer must cite which part of context supports it
3. **Graceful fallback** — only claim insufficient information if context is completely unrelated
4. **Clarity & conciseness** — well-reasoned but brief answers

This design prevents hallucination and ensures all answers are traceable to source documents.

---

## 🎓 Why Local LLMs?

| Factor | Cloud API (OpenAI) | Local LLMs (This Project) |
|--------|------------------|-------------------------|
| **Cost** | Pay per token, recurring expenses | Free forever |
| **Privacy** | Data sent to external servers | Never leaves your machine |
| **Internet** | Required for every query | Fully offline capable |
| **Latency** | Network dependent (100-500ms) | Local GPU speed (varies) |
| **Customization** | Limited to API parameters | Full control over model/prompts |
| **Data Compliance** | External third-party access | Meets strictest regulations |

---

## 🔄 Development Workflow

### Document Ingestion Process
1. Place PDF files in `docs/` folder
2. Call `/ingest` endpoint
3. PDFs are parsed and chunked
4. Chunks are embedded and stored in ChromaDB
5. Persistent index ready for queries

### Query Process
1. User submits question to `/ask` or `/ask/stream`
2. System retrieves top-K relevant chunks via MMR search
3. Prompt is constructed with context + question
4. LLM generates answer in real-time (streaming or buffered)
5. Response includes source metadata for verification

---

## 📊 Performance Characteristics

- **Embedding Speed** — milliseconds per chunk (depends on GPU)
- **Retrieval Speed** — near-instant from ChromaDB (in-memory indexed)
- **LLM Generation Speed** — depends on model size and GPU (Gemma 4 is optimized)
- **Streaming Latency** — token-by-token, no round-trip delay
- **Scalability** — efficiently handles dozens to hundreds of documents

---

## 🔐 Privacy & Security

✅ **Data Never Leaves Local Machine**
- No API calls to external services
- All processing on local hardware
- Suitable for confidential/proprietary documents

✅ **Offline Operation**
- Works without internet connection
- Models and databases stored locally

✅ **No Logging/Telemetry**
- Queries not logged to external services
- Full control over data retention

---

## 🚧 Extensibility & Future Enhancements

The modular architecture allows for:
- **Alternative LLMs** — swap Gemma 4 for other Ollama models
- **Different embeddings** — replace nomic-embed-text as needed
- **Custom RAG strategies** — modify retrieval or prompt logic
- **Database backends** — extend beyond ChromaDB
- **Document formats** — add support for .docx, .txt, HTML, etc.
- **Auth & access control** — add multi-user support
- **Monitoring** — integrate logging and performance metrics

---

## ✅ Completion Status

This project is **feature-complete and production-ready**:
- ✅ Full RAG pipeline implemented
- ✅ Multi-endpoint REST API with streaming
- ✅ Document ingestion and persistent indexing
- ✅ Source citation system
- ✅ Automatic API documentation
- ✅ Error handling and validation
- ✅ Privacy-first architecture
- ✅ Local LLM integration

---

## 🎯 Use Cases

1. **Legal Document Analysis** — ask questions about contracts, NDAs, regulations
2. **Medical Research** — query medical papers and research documents
3. **Financial Analysis** — analyze financial reports, SEC filings, audits
4. **Technical Documentation** — search and summarize technical specs
5. **Knowledge Base Search** — transform company documents into QA system
6. **Research Assistant** — organize and query academic papers
7. **Compliance Review** — audit documents for regulatory requirements

---

## 📞 Getting Started

1. **Install dependencies** — Set up Python venv with required packages
2. **Ensure Ollama running** — Start Ollama with Gemma 4 model available
3. **Place documents** — Add PDFs to `docs/` folder
4. **Ingest** — Call `/ingest` endpoint
5. **Query** — Use `/ask` or `/ask/stream` to ask questions
6. **Review answers** — Check source citations for verification

---

**Project Status:** ✅ Complete and Operational

*Last Updated: May 6, 2026*
