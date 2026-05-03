from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.core.rag import ask, get_rag_chain
from app.core.ingestor import ingest

app = FastAPI(
    title="Research Assistant AI",
    description="A RAG-powered research assistant using Gemma 4 and LangChain",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def root():
    return {"status": "running", "model": "gemma4:e2b"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = ask(request.question)  # ask() already enhances internally
    return AnswerResponse(question=request.question, answer=answer)

@app.post("/ask/stream")
def ask_stream(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    def generate():
        chain = get_rag_chain()
        enhanced = f"Based on the documents, please answer: {request.question}"
        for chunk in chain.stream(enhanced):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/ingest")
def ingest_documents():
    try:
        ingest()
        return {"status": "success", "message": "Documents ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))