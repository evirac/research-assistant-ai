from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

# CHROMA_DIR = "chroma_db"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "docs") 

def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )

def is_valid_chunk(text: str) -> bool:
    stripped = text.strip()
    
    # Too short
    if len(stripped) < 100:
        return False
    
    # Base64 image data
    if "sha1_base64" in stripped or "sha1_base64" in stripped:
        return False
    
    # Bibliography pattern — lines ending with years like (2020), (2024)
    lines = stripped.split("\n")
    year_pattern = sum(1 for line in lines if line.strip().endswith(("2020.", "2021.", "2022.", "2023.", "2024.", "2025.", "2019.", "2018.", "2017.", "2016.")))
    if year_pattern >= 2:
        return False
    
    # Too many comma-separated initials (author lists like "A. B., C. D.,")
    import re
    initial_pattern = len(re.findall(r'[A-Z]\.\s+[A-Z][a-z]+,', stripped))
    if initial_pattern >= 3:
        return False
    
    # Low alphabetic ratio (math/symbols heavy)
    alpha_chars = sum(c.isalpha() for c in stripped)
    if len(stripped) > 0 and (alpha_chars / len(stripped)) < 0.4:
        return False
    
    return True


def load_documents():
    docs = []
    for file in os.listdir(DOCS_DIR):
        if file.endswith(".pdf"):
            print(f"Loading: {file}")
            loader = PyPDFLoader(os.path.join(DOCS_DIR, file))
            docs.extend(loader.load())
    print(f"Total pages loaded: {len(docs)}")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    # Filter bad chunks
    valid_chunks = [c for c in chunks if is_valid_chunk(c.page_content)]
    print(f"Total chunks: {len(chunks)} → Valid chunks: {len(valid_chunks)}")
    return valid_chunks

def ingest():
    print("Starting ingestion...")
    docs = load_documents()
    if not docs:
        print("No PDFs found in /docs folder. Add some PDFs and try again.")
        return
    chunks = split_documents(docs)
    print("Creating embeddings and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR
    )
    print(f"Done. {len(chunks)} chunks stored in ChromaDB.")
    return vectorstore

def list_stored_chunks():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    results = vectorstore.get()
    print(f"\nTotal chunks stored: {len(results['documents'])}")
    print("\n--- First 5 stored chunks ---")
    for i, (doc, meta) in enumerate(zip(results['documents'][:5], results['metadatas'][:5])):
        print(f"\n[{i+1}] Page {meta.get('page','?')}: {doc[:150]}")

if __name__ == "__main__":
    ingest()
    list_stored_chunks()