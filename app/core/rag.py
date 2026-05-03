from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.core.llm import get_llm
import os

# CHROMA_DIR = "chroma_db"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "docs") 


PROMPT_TEMPLATE = """
You are an expert research assistant with deep analytical capabilities.
You have been provided with relevant excerpts from research documents.

Your job is to:
1. Use the provided context as your PRIMARY source of information
2. Apply your reasoning to interpret and connect ideas within the context
3. If the context contains related information even if not an exact match, reason through it
4. Only say "I don't have enough information" if the context is completely unrelated to the question
5. Always cite which part of the context supports your answer
6. Answer concisely and clearly.

Context:
{context}

Question:
{question}

Think carefully about what the context is saying and provide a well-reasoned answer:
"""

def get_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def get_rag_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # retrieve top 5 most relevant chunks
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = get_llm()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def ask(question: str) -> str:
    chain = get_rag_chain()
     # Add context to short questions to improve retrieval
    enhanced = f"Based on the documents, please answer: {question}"
    return chain.invoke(enhanced)

def debug_retrieval(question: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    docs = retriever.invoke(question)
    print(f"\n--- Retrieved chunks for: '{question}' ---")
    for i, doc in enumerate(docs):
        print(f"\n[Chunk {i+1}]")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Page: {doc.metadata.get('page', '?')}")
        print(f"Content preview: {doc.page_content[:200]}")
    print("\n--- End of chunks ---")

if __name__ == "__main__":
    question = "What are LLMs?"
    print(f"Question: {question}\n")
    print("Answer:")
    print(ask(question))