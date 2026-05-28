import uvicorn
from app.api.routes import app
from app.core.rag import debug_retrieval_hybrid as debug_retrieval

if __name__ == "__main__":
    # debug_retrieval("What are LLMs?", conversation_id="test_user")
    uvicorn.run("app.api.routes:app", host="0.0.0.0", port=8000, reload=True)
