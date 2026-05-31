"""
Enhanced FastAPI Routes with Multi-Turn Conversation Support
Adds conversation memory endpoints while maintaining backward compatibility.
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

from app.core.rag import (
    ask_with_memory,
    ask_without_memory,
    debug_retrieval_hybrid,
    get_hybrid_rag_chain,
    HybridRetriever,
    _extract_structured_sources,
    PROMPT_TEMPLATE_WITH_MEMORY,
)
from app.core.llm import (
    parse_thinking,
    ollama_chat_stream,
    get_model,
    set_model,
    list_available_models,
    get_model_info,
)
from app.core.memory import session_manager
from app.core.ingestor import ingest, list_ingested_documents, delete_document, ingest_single_file

app = FastAPI(
    title="Research Assistant AI - Multi-Turn",
    description="RAG-powered research assistant with conversation memory using Ollama and LangChain",
    version="2.0.0"
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class QuestionRequest(BaseModel):
    """Standard question request."""
    question: str = Field(..., description="The question to answer")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for memory")

class ConversationQuestion(BaseModel):
    """Question for a specific conversation thread."""
    question: str = Field(..., description="The question to answer")

class Citation(BaseModel):
    """A single structured citation from a retrieved document chunk."""
    file: str
    page: int
    preview: str

class AnswerResponse(BaseModel):
    """Response with answer and structured citation metadata."""
    question: str
    answer: str
    conversation_id: Optional[str] = None
    sources: List[Citation] = []
    memory_exchanges_used: int = 0
    documents_used: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)

class ConversationSummary(BaseModel):
    """Summary of a conversation."""
    conversation_id: str
    total_exchanges: int
    memory_size: int

class ConversationHistory(BaseModel):
    """History of a conversation."""
    conversation_id: str
    exchanges: List[dict] = []

class StreamingChunk(BaseModel):
    """Streaming chunk response."""
    chunk: str
    conversation_id: Optional[str] = None

class ModelSwitchRequest(BaseModel):
    """Request to switch the active LLM model."""
    model: str = Field(..., description="Ollama model tag, e.g. 'gemma4:12b'")


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": get_model(),
        "version": "2.0.0",
        "features": ["conversation_memory", "multi_turn_qa", "streaming", "structured_citations", "model_switching"]
    }


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/models", tags=["Models"])
def get_available_models():
    """
    List all locally available Ollama models.
    Queries Ollama's /api/tags endpoint.
    """
    try:
        models = list_available_models()
        current = get_model()
        return {
            "models": models,
            "current_model": current,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch models: {str(e)}")


@app.get("/models/current", tags=["Models"])
def get_current_model():
    """Get the currently active model and its details."""
    try:
        info = get_model_info()
        return {
            "model": get_model(),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not get model info: {str(e)}")


@app.put("/models/current", tags=["Models"])
def switch_model(request: ModelSwitchRequest):
    """
    Switch the active LLM model for all subsequent requests.
    The model must already be pulled locally via `ollama pull <model>`.
    Takes effect immediately — no server restart needed.
    """
    available = list_available_models()
    if request.model not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available: {available}. "
                   f"Pull it first with: ollama pull {request.model}"
        )
    set_model(request.model)
    info = get_model_info()
    return {
        "status": "success",
        "message": f"Switched to model: {request.model}",
        "model": request.model,
        "info": info,
    }


# ============================================================================
# SIMPLE Q&A ENDPOINTS (BACKWARD COMPATIBLE)
# ============================================================================

@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
def ask_question(request: QuestionRequest):
    """
    Ask a single question, optionally with conversation memory.

    If conversation_id is provided, the answer is stored in memory.
    If not, it's a one-off question.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        if request.conversation_id:
            result = ask_with_memory(request.question, request.conversation_id)
            return AnswerResponse(
                question=result["question"],
                answer=result["answer"],
                conversation_id=result["conversation_id"],
                sources=result["sources"],
                memory_exchanges_used=result["memory_exchanges_used"],
                documents_used=result["documents_used"]
            )
        else:
            result = ask_without_memory(request.question)
            return AnswerResponse(
                question=request.question,
                answer=result["answer"],
                sources=result["sources"]
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/ask/stream", tags=["Q&A"])
def ask_stream(request: QuestionRequest):
    """
    Stream the answer token by token.
    Supports conversation memory if conversation_id is provided.
    Uses whichever model is currently active via get_model().
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    def generate():
        try:
            import json
            from app.core.rag import PROMPT_TEMPLATE_WITH_MEMORY

            retriever = HybridRetriever(conversation_id=request.conversation_id)
            docs, memory_exchanges = retriever.retrieve(request.question)
            memory_context, doc_context = retriever.format_hybrid_context(docs, memory_exchanges)

            system_prompt = (
                "You are a research assistant analyzing academic documents. "
                "Answer with precise citations."
            )
            user_message = PROMPT_TEMPLATE_WITH_MEMORY.format(
                memory_context=memory_context,
                document_context=doc_context,
                question=f"Based on the documents, please answer: {request.question}",
            )

            thinking_started = False
            thinking_ended = False
            thinking_chunks = []
            answer_chunks = []

            for tok_type, text in ollama_chat_stream(system_prompt, user_message):
                if tok_type == "thinking":
                    if not thinking_started:
                        yield "<think>\n"
                        thinking_started = True
                    thinking_chunks.append(text)
                    yield text
                else:
                    if thinking_started and not thinking_ended:
                        yield "\n</think>\n\n"
                        thinking_ended = True
                    answer_chunks.append(text)
                    yield text

            if thinking_started and not thinking_ended:
                yield "\n</think>\n\n"

            full_answer = "".join(answer_chunks)
            if request.conversation_id and full_answer:
                memory = session_manager.get_or_create_session(request.conversation_id)
                flat_sources = [doc.metadata.get("source", "unknown") for doc in docs]
                memory.add_exchange(request.question, full_answer, flat_sources)

            structured_sources = _extract_structured_sources(docs)
            citations_payload = json.dumps({
                "sources":               structured_sources,
                "memory_exchanges_used": len(memory_exchanges),
                "documents_used":        len(docs),
                "thinking":              "".join(thinking_chunks),
                "model":                 get_model(),
            })
            yield f"\n||CITATIONS||{citations_payload}"

        except Exception as e:
            yield f"ERROR: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


# ============================================================================
# CONVERSATION-SPECIFIC ENDPOINTS
# ============================================================================

@app.post("/conversations/{conversation_id}/ask", response_model=AnswerResponse, tags=["Conversations"])
def ask_in_conversation(conversation_id: str, request: ConversationQuestion):
    """
    Ask a question within a specific conversation thread.
    Automatically uses conversation memory.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = ask_with_memory(request.question, conversation_id)
        return AnswerResponse(
            question=result["question"],
            answer=result["answer"],
            conversation_id=result["conversation_id"],
            sources=result["sources"],
            memory_exchanges_used=result["memory_exchanges_used"],
            documents_used=result["documents_used"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/conversations/{conversation_id}/summary", response_model=ConversationSummary, tags=["Conversations"])
def get_conversation_summary(conversation_id: str):
    """Get a summary of a conversation (number of exchanges, memory size)."""
    try:
        memory = session_manager.get_or_create_session(conversation_id)
        stats = memory.get_memory_stats()
        return ConversationSummary(
            conversation_id=stats["conversation_id"],
            total_exchanges=stats["total_exchanges"],
            memory_size=stats.get("memory_size", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")


@app.get("/conversations/{conversation_id}/history", tags=["Conversations"])
def get_conversation_history(conversation_id: str):
    """Get the full conversation history as formatted text."""
    try:
        memory = session_manager.get_or_create_session(conversation_id)
        history_text = memory.get_conversation_summary(max_pairs=100)
        return {
            "conversation_id": conversation_id,
            "history": history_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


@app.delete("/conversations/{conversation_id}", tags=["Conversations"])
def delete_conversation(conversation_id: str):
    """Clear all memory for a conversation."""
    try:
        session_manager.delete_session(conversation_id)
        return {
            "status": "success",
            "message": f"Conversation {conversation_id} cleared",
            "conversation_id": conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")


@app.get("/conversations", tags=["Conversations"])
def list_conversations():
    """List all active conversation IDs."""
    try:
        conversations = session_manager.list_sessions()
        return {
            "active_conversations": conversations,
            "count": len(conversations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")


# ============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/ingest", tags=["Documents"])
def ingest_documents():
    """Ingest PDF documents from the /docs folder."""
    try:
        ingest()
        return {"status": "success", "message": "Documents ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/documents", tags=["Documents"])
def list_documents():
    """List all ingested documents with their chunk counts."""
    try:
        documents = list_ingested_documents()
        return {
            "status": "success",
            "documents": documents,
            "total_documents": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.delete("/documents/{filename}", tags=["Documents"])
def remove_document(filename: str):
    """Remove all chunks belonging to a specific document from ChromaDB."""
    try:
        deleted_count = delete_document(filename)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found in vector store")
        return {
            "status": "success",
            "message": f"Document '{filename}' removed",
            "chunks_deleted": deleted_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.post("/ingest/{filename}", tags=["Documents"])
def ingest_single_document(filename: str):
    """Ingest a single PDF file from the /docs folder."""
    try:
        chunks_stored = ingest_single_file(filename)
        return {
            "status": "success",
            "message": f"'{filename}' ingested successfully",
            "chunks_stored": chunks_stored
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in /docs folder")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ============================================================================
# DEBUG ENDPOINTS
# ============================================================================

@app.get("/debug/retrieval", tags=["Debug"])
def debug_retrieval(question: str, conversation_id: Optional[str] = None):
    """Debug endpoint to see what's being retrieved."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        debug_retrieval_hybrid(question, conversation_id)
        return {
            "status": "debug_output_printed",
            "question": question,
            "conversation_id": conversation_id,
            "message": "Check server logs for debug output"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")