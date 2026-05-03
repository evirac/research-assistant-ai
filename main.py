import uvicorn
from app.core.rag import debug_retrieval

if __name__ == "__main__":
    debug_retrieval("What are LLMs?")
    uvicorn.run("app.api.routes:app", host="0.0.0.0", port=8000, reload=True)

