"""
Streamlit Chat Frontend for the Research Assistant AI
Connects to the FastAPI backend at http://localhost:8000

Features:
- Chat interface with message history
- Sidebar for conversation management and document management
- Per-message structured citations (file, page, preview)
- Streaming support via requests with stream=True
"""

import uuid
import json
import requests
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

API_BASE = "http://localhost:8000"

# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = "default_session"
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Each: {"role": "user"|"assistant", "content": str, "meta": dict}
    if "streaming" not in st.session_state:
        st.session_state.streaming = True
    if "api_base" not in st.session_state:
        st.session_state.api_base = API_BASE


init_session_state()


# =============================================================================
# API Helper Functions
# =============================================================================

def api_get(endpoint: str, params: dict = None) -> dict:
    """Send a GET request to the API."""
    try:
        resp = requests.get(f"{st.session_state.api_base}{endpoint}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def api_post(endpoint: str, json_data: dict = None) -> dict:
    """Send a POST request to the API."""
    try:
        resp = requests.post(
            f"{st.session_state.api_base}{endpoint}",
            json=json_data,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def api_delete(endpoint: str) -> dict:
    """Send a DELETE request to the API."""
    try:
        resp = requests.delete(f"{st.session_state.api_base}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def stream_answer(question: str, conversation_id: str):
    """
    Stream the answer from /ask/stream endpoint.
    Yields (text_chunk, is_error, citations_meta) tuples.
    citations_meta is only non-None on the final sentinel chunk.
    """
    try:
        resp = requests.post(
            f"{st.session_state.api_base}/ask/stream",
            json={"question": question, "conversation_id": conversation_id},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        buffer = ""
        SENTINEL_LEN = max(len("||CITATIONS||"), len("||THINKING||"))
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                buffer += chunk

                # Handle THINKING sentinel (comes before answer text)
                if "||THINKING||" in buffer:
                    parts = buffer.split("||THINKING||", 1)
                    pre = parts[0]
                    if pre:
                        yield pre, False, None, None
                    rest = parts[1]
                    # The thinking JSON ends at the next newline or ||CITATIONS||
                    # It is json.dumps of a string so it ends with a closing quote
                    # Safe to try parsing once we have a full JSON string
                    try:
                        thinking_text = json.loads(rest)
                        yield "", False, None, thinking_text
                        buffer = ""
                    except Exception:
                        buffer = "||THINKING||" + rest
                    continue

                # Handle CITATIONS sentinel (final)
                if "||CITATIONS||" in buffer:
                    parts = buffer.split("||CITATIONS||", 1)
                    text_part = parts[0]
                    if text_part:
                        yield text_part, False, None, None
                    try:
                        meta = json.loads(parts[1])
                    except Exception:
                        meta = {}
                    yield "", False, meta, None
                    buffer = ""
                else:
                    safe = buffer[:-SENTINEL_LEN] if len(buffer) > SENTINEL_LEN else ""
                    if safe:
                        yield safe, False, None, None
                        buffer = buffer[-SENTINEL_LEN:]
        # Flush remaining
        if buffer:
            if buffer.startswith("ERROR:"):
                yield buffer, True, None, None
            else:
                yield buffer, False, None, None
    except Exception as e:
        yield f"Streaming error: {e}", True, None


def ask_sync(question: str, conversation_id: str) -> dict:
    """
    Non-streaming question via /conversations/{id}/ask endpoint.
    Returns the full response dict with structured sources.
    """
    return api_post(f"/conversations/{conversation_id}/ask", {"question": question})


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with conversation management and document management."""
    with st.sidebar:
        st.title("🔬 Research Assistant")
        st.caption("RAG-powered local LLM research chat")

        # --- API Settings ---
        st.divider()
        st.subheader("⚙️ Connection")
        api_input = st.text_input("API Base URL", value=st.session_state.api_base)
        if api_input != st.session_state.api_base:
            st.session_state.api_base = api_input

        # Health check
        health = api_get("/")
        if health and health.get("status") == "running":
            st.success(f"✅ Connected — model: {health.get('model', 'unknown')}")
        else:
            st.error("❌ Backend unreachable")
            st.info("Start the backend: `python main.py`", icon="💡")

        # --- Conversation Management ---
        st.divider()
        st.subheader("💬 Conversation")

        # Conversation ID input
        conv_id = st.text_input(
            "Conversation ID",
            value=st.session_state.conversation_id,
            key="conv_id_input",
        )
        st.session_state.conversation_id = conv_id

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New", use_container_width=True):
                st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        # List active conversations
        st.subheader("📋 Active Conversations")
        convs = api_get("/conversations")
        active = convs.get("active_conversations", [])

        if active:
            for cid in active:
                is_current = cid == st.session_state.conversation_id
                label = f"{'➤ ' if is_current else '  '}{cid}"
                c1, c2 = st.columns([0.8, 0.2])
                with c1:
                    if st.button(label, key=f"switch_{cid}", use_container_width=True):
                        st.session_state.conversation_id = cid
                        st.session_state.messages = []
                        st.rerun()
                with c2:
                    if st.button("🗑", key=f"del_{cid}", help="Delete conversation memory"):
                        api_delete(f"/conversations/{cid}")
                        if cid == st.session_state.conversation_id:
                            st.session_state.messages = []
                        st.rerun()
        else:
            st.caption("No active conversations yet.")

        # Streaming toggle
        st.divider()
        st.session_state.streaming = st.toggle("⚡ Streaming", value=st.session_state.streaming)

        # --- Document Management ---
        st.divider()
        st.subheader("📚 Documents")

        # List ingested documents
        docs_data = api_get("/documents")
        documents = docs_data.get("documents", [])

        if documents:
            for doc in documents:
                fname = doc["filename"]
                chunks = doc["chunks"]
                c1, c2 = st.columns([0.75, 0.25])
                with c1:
                    st.caption(f"📄 {fname}")
                with c2:
                    st.caption(f"{chunks} chunks")
        else:
            st.caption("No documents ingested yet.")

        st.divider()
        # Ingest all
        if st.button("📂 Ingest All PDFs", use_container_width=True, type="primary"):
            with st.spinner("Ingesting documents..."):
                result = api_post("/ingest")
                if result and result.get("status") == "success":
                    st.success("Ingestion complete!")
                    st.rerun()
                else:
                    st.error("Ingestion failed.")


# =============================================================================
# Main Chat Area
# =============================================================================

def render_chat():
    """Render the main chat interface."""
    # Header showing current conversation
    st.header(f"Conversation: `{st.session_state.conversation_id}`")
    st.caption("Ask questions about your uploaded research papers.")

    # Display existing messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

                # --- Per-message metadata (sources, memory, docs used) ---
                meta = msg.get("meta", {})
                sources = meta.get("sources", [])
                mem_used = meta.get("memory_exchanges_used", 0)
                docs_used = meta.get("documents_used", 0)
                thinking = meta.get("thinking", "")

                # Show thinking trace if available
                if thinking:
                    with st.expander("🧠 Reasoning trace", expanded=False):
                        st.markdown(
                            f"<div style='font-size:0.85em; color:#888; white-space:pre-wrap;'>{thinking}</div>",
                            unsafe_allow_html=True,
                        )

                with st.expander(f"📎 Citations  ({len(sources)} sources · {mem_used} memory · {docs_used} docs)", expanded=False):
                    if sources:
                        for i, src in enumerate(sources, 1):
                            file = src.get("file", "unknown")
                            page = src.get("page", "?")
                            preview = src.get("preview", "")
                            st.markdown(
                                f"**{i}.** `{file}` — page {page}  \n"
                                f"<span style='color:#666'>_{preview}_</span>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No document sources for this answer.")

                    if mem_used > 0:
                        st.caption(f"🔁 {mem_used} memory exchanges used")

    # --- Chat input ---
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant"):
            if st.session_state.streaming:
                response_container = st.empty()
                full_text = ""
                is_error = False
                citations_meta = {}

                thinking_text = ""
                for chunk, error, meta, thinking in stream_answer(user_input, st.session_state.conversation_id):
                    if error:
                        full_text = chunk
                        is_error = True
                        break
                    if meta is not None:
                        citations_meta = meta
                        # thinking also arrives in final meta
                        if meta.get("thinking"):
                            thinking_text = meta["thinking"]
                    elif thinking is not None:
                        thinking_text = thinking
                    else:
                        full_text += chunk
                        response_container.markdown(full_text + "▌")

                response_container.markdown(full_text)

                if not is_error:
                    meta = {
                        "sources": citations_meta.get("sources", []),
                        "memory_exchanges_used": citations_meta.get("memory_exchanges_used", 0),
                        "documents_used": citations_meta.get("documents_used", 0),
                        "thinking": thinking_text or citations_meta.get("thinking", ""),
                    }
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_text,
                        "meta": meta,
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant", "content": full_text, "meta": {},
                    })
                st.rerun()
            else:
                # --- Non-streaming mode ---
                with st.spinner("Thinking..."):
                    result = ask_sync(user_input, st.session_state.conversation_id)

                answer = result.get("answer", "No answer received.")
                sources = result.get("sources", [])
                mem_used = result.get("memory_exchanges_used", 0)
                docs_used = result.get("documents_used", 0)

                st.markdown(answer)

                meta = {
                    "sources": sources,
                    "memory_exchanges_used": mem_used,
                    "documents_used": docs_used,
                }
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "meta": meta,
                })

                # Inline citations
                with st.expander(f"📎 Citations ({len(sources)} sources · {mem_used} memory · {docs_used} docs)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        file = src.get("file", "unknown")
                        page = src.get("page", "?")
                        preview = src.get("preview", "")
                        st.markdown(
                            f"**{i}.** `{file}` — page {page}  \n"
                            f"<span style='color:#666'>_{preview}_</span>",
                            unsafe_allow_html=True,
                        )
                st.rerun()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()