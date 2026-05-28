from collections import Counter
import os
import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "docs")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


# ---------------------------------------------------------------------------
# FIX 2: Layout-Aware PDF Loader
# ---------------------------------------------------------------------------

def _load_with_fitz(filepath: str) -> List[Document]:
    """
    Load a PDF using PyMuPDF (fitz) with block-level reading-order sort.

    fitz reads each page as a list of text blocks positioned on the page
    canvas. With sort=False it outputs blocks in PDF content-stream order,
    left-to-right *within* each column), which correctly handles two-column
    academic preprint layouts that PyPDFLoader scrambles.

    Metadata mirrors PyPDFLoader: {"source": filepath, "page": 0-indexed int}
    """
    import fitz  # pymupdf

    docs = []
    pdf = fitz.open(filepath)
    filename = os.path.basename(filepath)

    for page_num, page in enumerate(pdf):
        # sort=False: reads blocks in PDF content-stream order (the fix for 2-col layouts).
        # For LaTeX two-column preprints this means column 1 is extracted
        # fully before column 2 — the correct reading order.
        # sort=True would interleave both columns row-by-row (worse for LaTeX PDFs).
        text = page.get_text("text", sort=False)
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": filepath, "page": page_num}
            ))

    pdf.close()
    print(f"  [MuPDF] Loaded {len(docs)} pages from {filename}")
    return docs


def _load_with_pypdf(filepath: str) -> List[Document]:
    """Fallback loader using langchain's PyPDFLoader."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    print(f"  [PyPDF fallback] Loaded {len(pages)} pages from {os.path.basename(filepath)}")
    return pages


def _load_pdf(filepath: str) -> List[Document]:
    """Load a single PDF, preferring fitz for layout accuracy."""
    try:
        import fitz  # noqa: F401
        return _load_with_fitz(filepath)
    except ImportError:
        print("  [Warning] pymupdf not installed — falling back to PyPDFLoader.")
        print("  Install with: pip install pymupdf")
        return _load_with_pypdf(filepath)


# ---------------------------------------------------------------------------
# FIX 1: Math-Aware Chunk Validator
# FIX 4: Smarter Bibliography Filtering
# ---------------------------------------------------------------------------

# Regex patterns compiled once at module level for efficiency
_MATH_MARKERS = re.compile(
    r'[∑∏∫∂∇αβγδεζηθλμνξπρστφχψωΩ]'   # greek / math operators
    r'|\\(frac|sum|prod|int|mathbf|text|operatorname)\b'  # LaTeX macros
    r'|\b(softmax|argmax|argmin|sigmoid|relu|tanh|log|exp)\s*[(\[]'  # ML functions
    r'|[=<>≤≥≈≠±×÷·]\s*\d'             # equations: "= 0.5", "≤ n"
    r'|\d+\s*[×·]\s*\d+'               # dimension notation: "64 × 64"
    r'|\[\s*\d+\s*[,;]'               # matrix notation: "[0, 1;"
    r'|_{[^}]+}\^'                     # subscript-superscript LaTeX: "W_{Q}^T"
    r'|Q\s*K\s*[TV]'                   # transformer notation
)

_PROSE_INDICATORS = re.compile(
    r'\b(propose|demonstrate|show that|we find|results suggest|'
    r'therefore|furthermore|however|in contrast|compared to|'
    r'outperform|achieve|evaluate|previous work|baseline|'
    r'state.of.the.art|sota|fine.tun|pre.train)\b',
    re.IGNORECASE
)

_YEAR_ENDING = re.compile(
    r'\b(201[0-9]|202[0-5])\.\s*$'
)

_AUTHOR_INITIALS = re.compile(r'[A-Z]\.\s+[A-Z][a-z]+,')


def is_valid_chunk(text: str) -> bool:
    """
    Validate a text chunk for quality before indexing.

    Returns False (drop) for:
      - Too short (< 200 chars)
      - Base64 image noise
      - Pure bibliography lists  [FIX 4: now smarter]
      - Dense author-initial lists
      - Low alpha ratio AND no math content  [FIX 1: math exemption]
      - Pure figure/caption stubs
      - Header-only lines
    """
    stripped = text.strip()

    # ── Too short ────────────────────────────────────────────────────────────
    if len(stripped) < 200:
        return False

    # ── Base64 image noise ───────────────────────────────────────────────────
    if "sha1_base64" in stripped:
        return False

    lines = stripped.split("\n")

    # ── FIX 4: Smarter bibliography filter ───────────────────────────────────
    year_line_count = sum(
        1 for line in lines if _YEAR_ENDING.search(line.strip())
    )
    if year_line_count >= 2:
        # Only drop if it ALSO lacks substantive prose indicators.
        # A "Related Work" paragraph will have long lines AND connective words.
        has_long_line = any(len(l) > 80 for l in lines)
        has_prose = bool(_PROSE_INDICATORS.search(stripped))

        if not has_long_line and not has_prose:
            return False   # pure reference list → drop
        # else: analytical text citing papers inline → keep

    # ── Author initial lists ─────────────────────────────────────────────────
    initial_count = len(_AUTHOR_INITIALS.findall(stripped))
    if initial_count >= 3:
        return False

    # ── FIX 1: Alpha ratio with math-content exemption ───────────────────────
    alpha_chars = sum(c.isalpha() for c in stripped)
    alpha_ratio = alpha_chars / len(stripped) if stripped else 1.0

    if alpha_ratio < 0.25:
        # Hard drop — even math shouldn't produce <25% alpha
        return False

    if alpha_ratio < 0.45:
        # Grey zone (was hard-dropped at 0.40) — keep only if math markers present
        if not _MATH_MARKERS.search(stripped):
            return False
        # Math content confirmed → keep regardless of ratio

    # ── Figure/caption stubs ─────────────────────────────────────────────────
    if re.match(r'^(Figure|Fig\.|Table|Algorithm)\s+\d+', stripped) and len(stripped) < 400:
        return False

    # ── Header-only lines ────────────────────────────────────────────────────
    sentence_count = len(re.findall(r'[.!?]', stripped))
    if sentence_count < 2 and len(stripped) < 500:
        return False

    return True


# ---------------------------------------------------------------------------
# Document Loading (public API — used by ingest() and ingest_single_file())
# ---------------------------------------------------------------------------

def load_documents() -> List[Document]:
    """Load all PDFs from DOCS_DIR using layout-aware fitz loader."""
    docs = []
    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDFs found in /docs folder.")
        return docs

    for filename in pdf_files:
        print(f"Loading: {filename}")
        pages = _load_pdf(os.path.join(DOCS_DIR, filename))
        docs.extend(pages)

    print(f"Total pages loaded: {len(docs)}")
    return docs


def load_single_document(filepath: str) -> List[Document]:
    """Load a single PDF file using layout-aware fitz loader."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    pages = _load_pdf(filepath)
    print(f"Loaded {len(pages)} pages from {os.path.basename(filepath)}")
    return pages


# ---------------------------------------------------------------------------
# Splitting & Validation
# ---------------------------------------------------------------------------

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    valid_chunks = [c for c in chunks if is_valid_chunk(c.page_content)]
    print(f"Total chunks: {len(chunks)} → Valid chunks: {len(valid_chunks)} "
          f"({len(chunks) - len(valid_chunks)} dropped)")
    return valid_chunks


# ---------------------------------------------------------------------------
# Vector Store helpers
# ---------------------------------------------------------------------------

def get_document_vectorstore():
    """Get or create the document vector store (default ChromaDB collection)."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings()
    )


# ---------------------------------------------------------------------------
# Ingestion entry points
# ---------------------------------------------------------------------------

def ingest():
    """Ingest all PDFs from /docs into ChromaDB."""
    print("Starting ingestion...")
    docs = load_documents()
    if not docs:
        return

    chunks = split_documents(docs)
    if not chunks:
        print("No valid chunks generated.")
        return

    print("Creating embeddings and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR
    )
    print(f"Done. {len(chunks)} chunks stored in ChromaDB.")
    return vectorstore


def ingest_single_file(filename: str) -> int:
    """
    Ingest a single PDF file from the /docs folder into ChromaDB.

    Returns number of chunks stored.
    Raises FileNotFoundError if the file doesn't exist.
    """
    filepath = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"'{filename}' not found in {DOCS_DIR}")

    print(f"\n--- Ingesting single file: {filename} ---")
    pages = load_single_document(filepath)
    chunks = split_documents(pages)

    if not chunks:
        print("No valid chunks generated.")
        return 0

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings()
    )
    vectorstore.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks for '{filename}'")
    return len(chunks)


# ---------------------------------------------------------------------------
# Document Management (unchanged API surface)
# ---------------------------------------------------------------------------

def list_ingested_documents() -> list:
    """List all ingested documents with their chunk counts."""
    vectorstore = get_document_vectorstore()
    results = vectorstore.get()

    if not results or not results.get("metadatas"):
        return []

    source_counts = Counter()
    for meta in results["metadatas"]:
        source = meta.get("source", "unknown")
        filename = os.path.basename(source) if source else "unknown"
        source_counts[filename] += 1

    return [
        {"filename": fname, "chunks": count}
        for fname, count in sorted(source_counts.items())
    ]


def delete_document(filename: str) -> int:
    """Remove all chunks for a specific document from ChromaDB."""
    vectorstore = get_document_vectorstore()
    results = vectorstore.get()

    if not results or not results.get("ids"):
        return 0

    ids_to_delete = [
        doc_id for doc_id, meta in zip(results["ids"], results["metadatas"])
        if os.path.basename(meta.get("source", "")) == filename
    ]

    if not ids_to_delete:
        return 0

    vectorstore.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} chunks for '{filename}'")
    return len(ids_to_delete)


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def list_stored_chunks():
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings()
    )
    results = vectorstore.get()
    print(f"\nTotal chunks stored: {len(results['documents'])}")
    print("\n--- First 5 stored chunks ---")
    for i, (doc, meta) in enumerate(zip(results["documents"][:5], results["metadatas"][:5])):
        print(f"\n[{i+1}] Page {meta.get('page', '?')}: {doc[:150]}")


def audit_chunk_filter(filepath: str):
    """
    Dry-run the full ingestion pipeline on a single PDF and report
    how many chunks each filter rule drops. Useful for tuning.
    """
    print(f"\n=== CHUNK FILTER AUDIT: {os.path.basename(filepath)} ===")
    pages = _load_pdf(filepath)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(pages)
    print(f"Raw chunks: {len(chunks)}")

    drop_reasons = Counter()
    kept = 0
    for chunk in chunks:
        t = chunk.page_content.strip()
        if len(t) < 200:
            drop_reasons["too_short"] += 1; continue
        if "sha1_base64" in t:
            drop_reasons["base64"] += 1; continue
        lines = t.split("\n")
        year_count = sum(1 for l in lines if _YEAR_ENDING.search(l.strip()))
        if year_count >= 2:
            has_long = any(len(l) > 80 for l in lines)
            has_prose = bool(_PROSE_INDICATORS.search(t))
            if not has_long and not has_prose:
                drop_reasons["bibliography"] += 1; continue
        if len(_AUTHOR_INITIALS.findall(t)) >= 3:
            drop_reasons["author_list"] += 1; continue
        alpha = sum(c.isalpha() for c in t) / len(t)
        if alpha < 0.25:
            drop_reasons["hard_low_alpha"] += 1; continue
        if alpha < 0.45 and not _MATH_MARKERS.search(t):
            drop_reasons["low_alpha_no_math"] += 1; continue
        if re.match(r'^(Figure|Fig\.|Table|Algorithm)\s+\d+', t) and len(t) < 400:
            drop_reasons["figure_caption"] += 1; continue
        if len(re.findall(r'[.!?]', t)) < 2 and len(t) < 500:
            drop_reasons["header_only"] += 1; continue
        kept += 1

    print(f"Kept: {kept}")
    print("Dropped by rule:")
    for rule, count in drop_reasons.most_common():
        print(f"  {rule:25s}: {count}")
    print("=" * 50)


if __name__ == "__main__":
    ingest()
    list_stored_chunks()