"""
Conversation Memory Manager
Stores and retrieves conversation history using ChromaDB with a separate collection.
Provides summarization and relevance filtering to avoid context bloat.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from app.core.llm import ollama_chat

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


class ConversationMemory:
    """
    Manages multi-turn conversation history.
    Stores Q&A pairs in a separate ChromaDB collection from document embeddings.
    Retrieves relevant past exchanges to provide context for current questions.
    """

    def __init__(self, conversation_id: str):
        """
        Initialize memory for a specific conversation.

        Args:
            conversation_id: Unique ID for this conversation thread
        """
        self.conversation_id = conversation_id
        self.vectorstore = self._get_or_create_memory_store()

    def _get_or_create_memory_store(self):
        """Get or create ChromaDB collection for conversation memory."""
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return Chroma(
            persist_directory=CHROMA_DIR,
            collection_name=f"memory_{self.conversation_id}",
            embedding_function=embeddings
        )

    def add_exchange(self, question: str, answer: str, sources: List[str] = None):
        """
        Store a question-answer exchange in memory.

        Args:
            question: User's question
            answer: AI's answer
            sources: List of source documents used
        """
        # Create a searchable text combining Q&A
        exchange_text = f"Q: {question}\nA: {answer}"

        metadata = {
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": ",".join(sources) if sources else "",
            "type": "qa_pair"
        }

        # Add to vector store
        self.vectorstore.add_texts(
            texts=[exchange_text],
            metadatas=[metadata],
            ids=[f"{self.conversation_id}_{datetime.now().timestamp()}"]
        )

    def retrieve_relevant_context(self, question: str, k: int = 3) -> List[Dict]:
        """
        Retrieve k most relevant past Q&A pairs for the current question.

        Args:
            question: Current question
            k: Number of past exchanges to retrieve

        Returns:
            List of past exchanges with metadata
        """
        try:
            results = self.vectorstore.similarity_search_with_score(
                question,
                k=k,
                filter={"conversation_id": self.conversation_id}
            )

            relevant_exchanges = []
            for doc, score in results:
                if score < 1.5:  # Only include reasonably similar exchanges
                    relevant_exchanges.append({
                        "exchange": doc.page_content,
                        "question": doc.metadata.get("question", ""),
                        "answer": doc.metadata.get("answer", ""),
                        "similarity_score": score,
                        "timestamp": doc.metadata.get("timestamp", "")
                    })

            return relevant_exchanges
        except Exception as e:
            print(f"Error retrieving memory context: {e}")
            return []

    def get_conversation_summary(self, max_pairs: int = 10) -> str:
        """
        Get a summary of the conversation so far.
        Useful for context-aware responses.

        Args:
            max_pairs: Maximum Q&A pairs to summarize

        Returns:
            String summary of conversation history
        """
        try:
            results = self.vectorstore.get(limit=max_pairs)

            if not results or not results.get("documents"):
                return "No prior conversation history."

            # Build conversation string
            conversation_text = "Recent conversation:\n"
            for doc, meta in zip(results["documents"], results["metadatas"]):
                conversation_text += f"\nQ: {meta.get('question', 'N/A')}\nA: {meta.get('answer', 'N/A')}\n"

            # Optional: Summarize if too long (for very long conversations)
            if len(conversation_text) > 2000:
                conversation_text = self._summarize_conversation(conversation_text)

            return conversation_text
        except Exception as e:
            print(f"Error getting conversation summary: {e}")
            return "No prior conversation history."

    def _summarize_conversation(self, conversation_text: str) -> str:
        """
        Summarize a long conversation to fit in context.
        Calls ollama_chat directly — no LangChain chain needed.
        """
        try:
            _, summary = ollama_chat(
                system_prompt="You are a helpful assistant that summarizes conversations concisely.",
                user_message=(
                    "Summarize this conversation in 3-4 bullet points, "
                    "keeping key topics and facts:\n\n"
                    f"{conversation_text}\n\nSummary:"
                ),
            )
            return f"Conversation summary:\n{summary}"
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return conversation_text[:1500] + "..."

    def clear_conversation(self):
        """Clear all history for this conversation."""
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = self._get_or_create_memory_store()
            print(f"Cleared conversation {self.conversation_id}")
        except Exception as e:
            print(f"Error clearing conversation: {e}")

    def get_memory_stats(self) -> Dict:
        """Get stats about the conversation memory."""
        try:
            results = self.vectorstore.get()
            return {
                "conversation_id": self.conversation_id,
                "total_exchanges": len(results.get("documents", [])),
                "memory_size": len(str(results.get("documents", []))),
            }
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {"conversation_id": self.conversation_id, "total_exchanges": 0}


class SessionManager:
    """
    Manages multiple conversation sessions.
    Maps conversation IDs to ConversationMemory instances.
    """

    def __init__(self):
        self.sessions: Dict[str, ConversationMemory] = {}

    def get_or_create_session(self, conversation_id: str) -> ConversationMemory:
        """Get existing session or create new one."""
        if conversation_id not in self.sessions:
            self.sessions[conversation_id] = ConversationMemory(conversation_id)
        return self.sessions[conversation_id]

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())

    def delete_session(self, conversation_id: str):
        """Delete a session and clear its memory."""
        if conversation_id in self.sessions:
            self.sessions[conversation_id].clear_conversation()
            del self.sessions[conversation_id]


# Global session manager instance
session_manager = SessionManager()