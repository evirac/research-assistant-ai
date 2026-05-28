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
from app.core.llm import parse_thinking
from app.core.memory import session_manager
from app.core.ingestor import ingest, list_ingested_documents, delete_document, ingest_single_file

app = FastAPI(
    title="Research Assistant AI - Multi-Turn",
    description="RAG-powered research assistant with conversation memory using Gemma 4 and LangChain",
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

# --- PRIORITY 2: Updated AnswerResponse with structured sources ---
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
    sources: List[Citation] = []          # structured [{file, page, preview}, ...]
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


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model": "gemma4:e2b",
        "version": "2.0.0",
        "features": ["conversation_memory", "multi_turn_qa", "streaming", "structured_citations"]
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
            # With memory — returns structured sources
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
            # Without memory — returns structured sources
            result = ask_without_memory(request.question)
            return AnswerResponse(
                question=request.question,
                answer=result["answer"],
                sources=result["sources"]
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


# --- PRIORITY 3: Fixed streaming endpoint with memory storage ---
@app.post("/ask/stream", tags=["Q&A"])
def ask_stream(request: QuestionRequest):
    """
    Stream the answer token by token.
    Supports conversation memory if conversation_id is provided.

    PRIORITY 3 FIX: Collects all streamed chunks, then stores the
    completed answer in conversation memory after streaming finishes.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    def generate():
        try:
            retriever = HybridRetriever(conversation_id=request.conversation_id)
            docs, memory_exchanges = retriever.retrieve(request.question)
            memory_context, doc_context = retriever.format_hybrid_context(docs, memory_exchanges)

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from app.core.llm import get_llm
            import json

            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WITH_MEMORY)
            llm = get_llm()
            chain = prompt | llm | StrOutputParser()

            enhanced = f"Based on the documents, please answer: {request.question}"
            chunks_collected = []
            in_think_block = False
            think_chunks = []

            for chunk in chain.stream({
                "memory_context": memory_context,
                "document_context": doc_context,
                "question": enhanced
            }):
                chunks_collected.append(chunk)

                # Buffer the full output so far to detect think tags
                so_far = "".join(chunks_collected)

                if "<think>" in so_far and not in_think_block:
                    in_think_block = True

                if in_think_block:
                    # Still inside thinking block — don't stream to user yet
                    think_chunks.append(chunk)
                    if "</think>" in so_far:
                        # Thinking block closed — extract and send sentinel
                        in_think_block = False
                        _, answer_so_far = parse_thinking(so_far)
                        # Send the thinking block as a sentinel so the frontend can display it
                        yield f"||THINKING||{json.dumps(''.join(think_chunks))}"
                        # Stream whatever answer text came after </think>
                        if answer_so_far:
                            yield answer_so_far
                else:
                    yield chunk

            # After streaming, store in memory and send citations as sentinel
            if chunks_collected:
                full_raw = "".join(chunks_collected)
                thinking_text, full_answer = parse_thinking(full_raw)
                if request.conversation_id:
                    memory = session_manager.get_or_create_session(request.conversation_id)
                    flat_sources = [doc.metadata.get("source", "unknown") for doc in docs]
                    # Store clean answer (no think tags) in memory
                    memory.add_exchange(request.question, full_answer, flat_sources)

                structured_sources = _extract_structured_sources(docs)
                citations_payload = json.dumps({
                    "sources": structured_sources,
                    "memory_exchanges_used": len(memory_exchanges),
                    "documents_used": len(docs),
                    "thinking": thinking_text,
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
    """
    Get a summary of a conversation (number of exchanges, memory size).
    """
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
    """
    Get the full conversation history as formatted text.
    """
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
    """
    Clear all memory for a conversation.
    """
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
    """
    List all active conversation IDs.
    """
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
    """
    Ingest PDF documents from the /docs folder.
    Creates embeddings and stores in ChromaDB.
    """
    try:
        ingest()
        return {
            "status": "success",
            "message": "Documents ingested successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# --- PRIORITY 5: Document management endpoints ---

@app.get("/documents", tags=["Documents"])
def list_documents():
    """
    List all ingested documents with their chunk counts.
    Returns a deduplicated list of filenames and how many chunks
    each contributes in the vector store.
    """
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
    """
    Remove all chunks belonging to a specific document from ChromaDB.
    This deletes the document from the vector store.
    """
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
    """
    Ingest a single PDF file from the /docs folder.
    Creates embeddings and stores in ChromaDB.
    """
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
    """
    Debug endpoint to see what's being retrieved.
    Shows both document and memory retrieval results.
    """
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