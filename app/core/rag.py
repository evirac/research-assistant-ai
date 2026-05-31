import os
from typing import List, Dict, Tuple
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.core.llm import ollama_chat, parse_thinking
from app.core.memory import ConversationMemory, session_manager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

PROMPT_TEMPLATE_WITH_MEMORY = """
You are a research assistant analyzing academic documents. Answer with precise citations.

**CRITICAL INSTRUCTION – COMPARATIVE QUESTIONS:**
When asked to compare/contrast two concepts (e.g., "How does X differ from Y?"):
1. First extract what the documents say about X.
2. Then extract what they say about Y.
3. Synthesize the differences based on their described properties, even if no document explicitly states "X differs from Y by...".
4. If a difference is clearly implied (e.g., X has component A, Y has component B instead), state it confidently but mark it as "inferred from the descriptions".
5. Only say "not enough information" if the documents contain NO description of one of the concepts.

**Example of good synthesis:**
- Documents describe Transformer as having both encoder and decoder stacks.
- Documents describe BERT as using only the encoder.
- Answer: "According to the papers: The original Transformer (Attention is All You Need, p.2) uses both an encoder and decoder. BERT (BERT paper, p.2) uses only the encoder stack. This is the key architectural difference – BERT removes the decoder for pre-training tasks. (Note: The comparison is inferred from separate descriptions.)"

**GROUNDING RULES:**
- For directly stated facts: cite with filename and page.
- For inferred comparisons: cite the sources used for each side and explicitly note "inferred".
- Never invent properties not mentioned anywhere.

Conversation Context:
{memory_context}

Document Context:
{document_context}

Question:
{question}

Provide a well-reasoned answer with inline citations in the format (filename.pdf, p.X). Use inferred statements where appropriate but clearly mark them.
"""

# Retrieval tuning constants — adjust here, not scattered through code
_CANDIDATE_POOL    = 40    # how many raw candidates to fetch from ChromaDB
_FINAL_K           = 8     # how many chunks to pass to the LLM
_MAX_PER_SOURCE    = 2     # max chunks allowed from any single document
_SCORE_THRESHOLD   = 1.1   # L2 distance ceiling (lower = more relevant, 0 = perfect match)                           
                           # bypassing the per-source cap entirely. 1.1 lets the cap
                           # do the diversity work instead of the threshold.


class HybridRetriever:
    """
    Retrieves and re-ranks documents from both conversation memory and document store.

    Key behaviour vs the original:
    - Fetches _CANDIDATE_POOL=40 candidates instead of k=6
    - Filters out weak matches above _SCORE_THRESHOLD
    - Caps each source document at _MAX_PER_SOURCE chunks
    - Selects the best _FINAL_K after those constraints
    This prevents any single large document (e.g. 208-chunk LLaMA 2 paper)
    from flooding all retrieval slots.
    """

    def __init__(self, conversation_id: str = None):
        self.conversation_id = conversation_id
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.doc_vectorstore = self._get_document_store()
        self.memory = None

        if conversation_id:
            self.memory = session_manager.get_or_create_session(conversation_id)

    def _get_document_store(self):
        """Get the document vector store (original ChromaDB collection)."""
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
        )

    def retrieve(self, question: str, k_docs: int = _FINAL_K, k_memory: int = 3):
        """
        Retrieve document chunks and memory exchanges for a question.

        Document retrieval pipeline:
          1. Fetch _CANDIDATE_POOL candidates with similarity scores
          2. Drop candidates above _SCORE_THRESHOLD (too dissimilar)
          3. Cap each source at _MAX_PER_SOURCE chunks (prevents large-doc dominance)
          4. Return the top k_docs by score

        Returns:
            (ranked_docs, memory_exchanges)
        """
        # --- Step 1: Fetch large candidate pool with scores ---
        results_with_scores = self.doc_vectorstore.similarity_search_with_score(
            question,
            k=_CANDIDATE_POOL
        )

        # --- Step 2: Filter by relevance threshold ---
        # ChromaDB L2 distance: 0 = identical, higher = more different
        filtered = [
            (doc, score)
            for doc, score in results_with_scores
            if score <= _SCORE_THRESHOLD
        ]

        # If threshold is too strict and we got nothing, fall back to top-10 by score
        if not filtered:
            filtered = sorted(results_with_scores, key=lambda x: x[1])[:10]

        # --- Step 3: Per-source cap ---
        # Track how many chunks we've accepted per source filename
        source_counts: Dict[str, int] = {}
        capped: List[Tuple[Document, float]] = []

        for doc, score in sorted(filtered, key=lambda x: x[1]):  # best score first
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            count = source_counts.get(source, 0)
            if count < _MAX_PER_SOURCE:
                capped.append((doc, score))
                source_counts[source] = count + 1
            if len(capped) >= k_docs:
                break

        # Extract just the documents (scores already used for ranking)
        document_chunks = [doc for doc, _ in capped]

        # --- Memory retrieval (unchanged) ---
        memory_exchanges = []
        if self.memory:
            memory_exchanges = self.memory.retrieve_relevant_context(question, k=k_memory)

        return document_chunks, memory_exchanges

    def format_hybrid_context(self, documents: List[Document], memory_exchanges: List[Dict]) -> Tuple[str, str]:
        """
        Format documents and memory exchanges for the prompt.

        Returns:
            Tuple of (memory_context, document_context)
        """
        doc_context = self._format_documents(documents)
        memory_context = self._format_memory_exchanges(memory_exchanges)
        return memory_context, doc_context

    def _format_documents(self, docs: List[Document]) -> str:
        if not docs:
            return "No relevant documents found."

        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            filename = os.path.basename(source) if source else "unknown"
            page = doc.metadata.get("page", "?")
            content = doc.page_content[:800]
            formatted.append(f"[{filename}, p.{page}]\n{content}")

        return "\n\n".join(formatted)

    def _format_memory_exchanges(self, exchanges: List[Dict]) -> str:
        """Format past conversation exchanges for the prompt."""
        if not exchanges:
            return "No prior conversation context."

        formatted = []
        for i, exchange in enumerate(exchanges):
            similarity = exchange.get("similarity_score", 0)
            formatted.append(
                f"[Previous Exchange {i+1}] (Relevance: {similarity:.2f})\n"
                f"Q: {exchange.get('question', '')}\n"
                f"A: {exchange.get('answer', '')}"
            )

        return "\n\n".join(formatted)


def _extract_structured_sources(docs: List[Document]) -> List[dict]:
    """
    Build structured, deduplicated citations from retrieved documents.

    Returns a list of dicts with:
        file:    filename only (e.g. "paper.pdf")
        page:    page number as int
        preview: first 120 chars of the chunk
    """
    seen = set()
    structured = []

    for doc in docs:
        full_source = doc.metadata.get("source", "unknown")
        filename = os.path.basename(full_source) if full_source else "unknown"
        page = doc.metadata.get("page", "?")

        key = (filename, str(page))
        if key in seen:
            continue
        seen.add(key)

        preview = doc.page_content[:120].strip().replace("\n", " ")
        if len(doc.page_content) > 120:
            preview += "..."

        structured.append({
            "file": filename,
            "page": page,
            "preview": preview
        })

    return structured


def get_hybrid_rag_chain(conversation_id: str = None):
    """
    Kept for backward compatibility.
    Returns (None, retriever) — callers should use ask_with_memory directly.
    """
    retriever = HybridRetriever(conversation_id=conversation_id)
    return None, retriever


def ask_with_memory(question: str, conversation_id: str) -> Dict:
    """
    Ask a question with conversation memory context.
    Uses ollama_chat (direct /api/chat) so thinking tokens are captured.
    """
    retriever = HybridRetriever(conversation_id=conversation_id)
    docs, memory_exchanges = retriever.retrieve(question)
    memory_context, doc_context = retriever.format_hybrid_context(docs, memory_exchanges)

    # Build the filled prompt string directly (no LangChain chain needed)
    system_prompt = "You are a research assistant analyzing academic documents. Answer with precise citations."
    user_message = PROMPT_TEMPLATE_WITH_MEMORY.format(
        memory_context=memory_context,
        document_context=doc_context,
        question=f"Based on the documents, please answer: {question}",
    )

    thinking, answer = ollama_chat(system_prompt, user_message)

    structured_sources = _extract_structured_sources(docs)

    memory_instance = session_manager.get_or_create_session(conversation_id)
    flat_sources = [doc.metadata.get("source", "unknown") for doc in docs]
    memory_instance.add_exchange(question, answer, flat_sources)

    return {
        "question":              question,
        "answer":                answer,
        "thinking":              thinking,
        "sources":               structured_sources,
        "conversation_id":       conversation_id,
        "memory_exchanges_used": len(memory_exchanges),
        "documents_used":        len(docs),
    }


def ask_without_memory(question: str) -> Dict:
    """
    Ask a question WITHOUT memory (one-off questions).
    Uses ollama_chat so thinking tokens are captured.
    """
    retriever = HybridRetriever(conversation_id=None)
    docs, _ = retriever.retrieve(question)
    memory_context, doc_context = retriever.format_hybrid_context(docs, [])

    system_prompt = "You are a research assistant analyzing academic documents. Answer with precise citations."
    user_message = PROMPT_TEMPLATE_WITH_MEMORY.format(
        memory_context=memory_context,
        document_context=doc_context,
        question=f"Based on the documents, please answer: {question}",
    )

    thinking, answer = ollama_chat(system_prompt, user_message)
    structured_sources = _extract_structured_sources(docs)

    return {
        "question": question,
        "answer":   answer,
        "thinking": thinking,
        "sources":  structured_sources,
    }


def debug_retrieval_hybrid(question: str, conversation_id: str = None):
    """
    Debug tool to see what's being retrieved from both sources.
    Now also prints per-source chunk counts and scores.
    """
    retriever = HybridRetriever(conversation_id=conversation_id)
    docs, memory_exchanges = retriever.retrieve(question)

    print(f"\n{'='*60}")
    print(f"HYBRID RETRIEVAL DEBUG: {question}")
    print(f"{'='*60}")

    # Count chunks per source in result
    from collections import Counter
    source_counts = Counter(
        os.path.basename(doc.metadata.get("source", "unknown")) for doc in docs
    )
    print(f"\n📊 SOURCE DISTRIBUTION ({len(docs)} chunks total):")
    for src, count in source_counts.most_common():
        print(f"   {src}: {count} chunk(s)")

    print(f"\n📄 DOCUMENTS RETRIEVED:")
    print("-" * 60)
    for i, doc in enumerate(docs):
        print(f"\n[Doc {i+1}] {doc.metadata.get('source', 'unknown')} (Page {doc.metadata.get('page', '?')})")
        print(f"Content: {doc.page_content[:200]}...")

    if conversation_id and memory_exchanges:
        print(f"\n💬 CONVERSATION MEMORY ({len(memory_exchanges)} exchanges):")
        print("-" * 60)
        for i, exchange in enumerate(memory_exchanges):
            print(f"\n[Memory {i+1}] (Similarity: {exchange.get('similarity_score', 0):.2f})")
            print(f"Q: {exchange.get('question', '')}")
            print(f"A: {exchange.get('answer', '')[:200]}...")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    print("TEST 1: Question with conversation memory")
    print("-" * 60)
    result = ask_with_memory("What are LLMs?", conversation_id="test_user_1")
    print(f"Answer: {result['answer']}\n")
    print(f"Structured sources: {result['sources']}\n")

    print("\nTEST 2: Follow-up question (should use memory)")
    print("-" * 60)
    result2 = ask_with_memory("Can you expand on transformers?", conversation_id="test_user_1")
    print(f"Answer: {result2['answer']}\n")
    print(f"Memory exchanges used: {result2['memory_exchanges_used']}")

    print("\nDEBUG: Hybrid retrieval")
    debug_retrieval_hybrid("How does BERT differ from the original Transformer?")