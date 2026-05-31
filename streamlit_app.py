import uuid
import json
import os
import time
import requests
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"
CHAT_STORE_FILE = "chat_history.json"


def load_chat_store():
    if os.path.exists(CHAT_STORE_FILE):
        with open(CHAT_STORE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_chat_store(store_data):
    with open(CHAT_STORE_FILE, "w") as f:
        json.dump(store_data, f, indent=2)


def init_session_state():
    if "api_base" not in st.session_state:
        st.session_state.api_base = API_BASE
    if "chat_store" not in st.session_state:
        st.session_state.chat_store = load_chat_store()
    if "conversation_id" not in st.session_state:
        if st.session_state.chat_store:
            st.session_state.conversation_id = list(st.session_state.chat_store.keys())[-1]
        else:
            create_new_chat()
    if "streaming" not in st.session_state:
        st.session_state.streaming = True


def create_new_chat():
    new_id = f"session_{uuid.uuid4().hex[:8]}"
    chat_num = len(st.session_state.chat_store) + 1
    st.session_state.chat_store[new_id] = {
        "title": f"Research Chat {chat_num}",
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }
    st.session_state.conversation_id = new_id
    save_chat_store(st.session_state.chat_store)


init_session_state()


def api_get(endpoint: str, params: dict = None) -> dict:
    try:
        resp = requests.get(
            f"{st.session_state.api_base}{endpoint}", params=params, timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def api_post(endpoint: str, json_data: dict = None) -> dict:
    try:
        resp = requests.post(
            f"{st.session_state.api_base}{endpoint}",
            json=json_data,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def api_delete(endpoint: str) -> dict:
    try:
        resp = requests.delete(f"{st.session_state.api_base}{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def stream_answer(question: str, conversation_id: str):
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
                if "||THINKING||" in buffer:
                    parts = buffer.split("||THINKING||", 1)
                    pre = parts[0]
                    if pre:
                        yield pre, False, None, None
                    rest = parts[1]
                    try:
                        thinking_text = json.loads(rest)
                        yield "", False, None, thinking_text
                        buffer = ""
                    except Exception:
                        buffer = "||THINKING||" + rest
                    continue
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
                    safe = (
                        buffer[:-SENTINEL_LEN] if len(buffer) > SENTINEL_LEN else ""
                    )
                    if safe:
                        yield safe, False, None, None
                        buffer = buffer[-SENTINEL_LEN:]
        if buffer:
            if buffer.startswith("ERROR:"):
                yield buffer, True, None, None
            else:
                yield buffer, False, None, None
    except Exception as e:
        yield f"Streaming error: {e}", True, None, None


def ask_sync(question: str, conversation_id: str) -> dict:
    return api_post(f"/conversations/{conversation_id}/ask", {"question": question})


def inject_custom_css():
    st.markdown(
        """
    <style>
    /* ═══════════════════════════════════════════════════
       HIDE DEFAULT CHROME
       ═══════════════════════════════════════════════════ */
    #MainMenu { visibility: hidden; height: 0; }
    footer { visibility: hidden; height: 0; }
    header { background-color: transparent !important; }
    [data-testid="stToolbar"] > div:nth-child(2),
    [data-testid="stToolbar"] > div:nth-child(3) { visibility: hidden; }

    /* ═══════════════════════════════════════════════════
       GLOBAL BASE (Neon Dark Theme)
       ═══════════════════════════════════════════════════ */
    .stApp {
        background-color: #05050A;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ═══════════════════════════════════════════════════
       SCROLLBARS (Sleek & Colorful)
       ═══════════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #05050A; }
    ::-webkit-scrollbar-thumb { background: #2A2A40; border-radius: 4px; border: 2px solid #05050A; }
    ::-webkit-scrollbar-thumb:hover { background: #FF2A6D; }

    /* ═══════════════════════════════════════════════════
       SIDEBAR
       ═══════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background-color: #0B0B14;
        border-right: 1px solid #1A1A2E;
    }
    [data-testid="stSidebar"]::before {
        content: '';
        display: block;
        height: 4px;
        background: linear-gradient(90deg, #FF2A6D, #7B2CBF, #05D9E8);
        margin-bottom: 15px;
        box-shadow: 0 2px 10px rgba(255, 42, 109, 0.4);
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    
    /* Neon New Chat Button */
    .sidebar-new-chat > button {
        background: linear-gradient(135deg, #FF2A6D 0%, #7B2CBF 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        border: none !important;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(255, 42, 109, 0.25);
        transition: all 0.2s ease-in-out;
    }
    .sidebar-new-chat > button:hover {
        box-shadow: 0 6px 20px rgba(255, 42, 109, 0.4);
        transform: translateY(-2px);
    }

    /* Ingest Button */
    .sidebar-ingest > button {
        background: linear-gradient(135deg, #05D9E8 0%, #01A2B5 100%);
        color: #05050A !important;
        font-weight: 700;
        border: none !important;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(5, 217, 232, 0.2);
        transition: all 0.2s ease-in-out;
    }
    .sidebar-ingest > button:hover {
        box-shadow: 0 6px 20px rgba(5, 217, 232, 0.4);
        transform: translateY(-2px);
    }

    /* ═══════════════════════════════════════════════════
       UNIFIED CHAT BUTTONS (SPLIT BUTTON STYLE)
       ═══════════════════════════════════════════════════ */
    /* The row wrapper acts as the single unified button body */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        background-color: #0E0E1A;
        border: 1px solid #1F1F33;
        border-radius: 8px;
        margin-bottom: 8px;
        gap: 0 !important; /* Forces the elements flush together */
        align-items: stretch !important;
        overflow: hidden;
        transition: all 0.2s ease-in-out;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover {
        border-color: #05D9E8;
        box-shadow: 0 0 10px rgba(5, 217, 232, 0.15);
    }

    /* Make the actual Streamlit buttons totally transparent */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        height: 100% !important;
        min-height: 42px !important;
        margin: 0 !important;
        border-radius: 0 !important;
    }

    /* Left Side (Chat Select) */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) button {
        justify-content: flex-start;
        padding-left: 12px !important;
        color: #94A3B8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) button:hover {
        background-color: rgba(255, 255, 255, 0.03) !important;
        color: #FFFFFF !important;
    }

    /* Right Side (Delete 🗑️) */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) {
        border-left: 1px solid #1F1F33;
        background-color: rgba(0, 0, 0, 0.2);
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) button {
        padding: 0 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem !important;
        opacity: 0.6;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) button:hover {
        background-color: rgba(255, 42, 109, 0.2) !important;
        opacity: 1;
    }

    /* Status Dot */
    .dot {
        display: inline-block; width: 8px; height: 8px;
        border-radius: 50%; margin-right: 8px; vertical-align: middle;
    }
    .dot-on { background: #00FF9D; box-shadow: 0 0 10px rgba(0,255,157,0.6); }
    .dot-off { background: #FF2A6D; box-shadow: 0 0 10px rgba(255,42,109,0.6); }
    .dot-text { font-size: 0.8rem; font-weight: 500; color: #94A3B8; vertical-align: middle; }

    /* Labels & Dividers */
    .sb-sep { border: none; height: 1px; background: #1A1A2E; margin: 20px 0; }
    .sb-label {
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.15em; color: #64748B; margin-bottom: 12px;
    }

    /* Doc entries */
    .doc-row {
        font-size: 0.8rem; color: #94A3B8; padding: 10px;
        background-color: #10101C; border: 1px solid #1A1A2E;
        border-radius: 8px; margin-bottom: 8px;
    }

    /* ═══════════════════════════════════════════════════
       MAIN CHAT AREA
       ═══════════════════════════════════════════════════ */
    .main-hdr {
        color: #FFFFFF; font-size: 1.8rem; font-weight: 800;
        letter-spacing: -0.03em; margin-bottom: 6px;
    }
    .main-hdr-line {
        height: 3px; background: linear-gradient(90deg, #FF2A6D, transparent);
        border-radius: 2px; opacity: 0.8; margin-bottom: 30px; width: 150px;
    }

    /* Empty State */
    .empty-wrap { text-align: center; padding: 80px 20px; background: #0A0A14; border-radius: 16px; border: 1px dashed #1A1A2E; }
    .empty-icon { font-size: 3rem; margin-bottom: 15px; filter: drop-shadow(0 0 10px rgba(5,217,232,0.5)); }
    .empty-h { color: #E2E8F0; font-size: 1.4rem; font-weight: 700; margin-bottom: 10px; }
    .empty-p { color: #94A3B8; font-size: 0.95rem; max-width: 400px; margin: 0 auto; line-height: 1.6; }

    /* ═══════════════════════════════════════════════════
       CHAT BUBBLES & CARDS
       ═══════════════════════════════════════════════════ */
    [data-testid="stChatMessage"] {
        background-color: #0E0E1A;
        border: 1px solid #1F1F33;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stChatMessage"] div[data-testid="stAvatar"] {
        background: linear-gradient(135deg, #7B2CBF, #FF2A6D);
        border: 2px solid #05050A;
        box-shadow: 0 0 10px rgba(255,42,109,0.3);
    }
    /* Make code blocks stand out */
    [data-testid="stChatMessage"] pre {
        background-color: #05050A; border: 1px solid #1F1F33; border-radius: 8px;
    }
    [data-testid="stChatMessage"] code { color: #05D9E8; background-color: rgba(5, 217, 232, 0.1); padding: 2px 6px; border-radius: 4px; }
    [data-testid="stChatMessage"] a { color: #FF2A6D; text-decoration: none; font-weight: 600; }
    [data-testid="stChatMessage"] a:hover { color: #05D9E8; }

    /* ═══════════════════════════════════════════════════
       EXPANDERS (Thinking & Sources Wrapper)
       ═══════════════════════════════════════════════════ */
    [data-testid="stExpander"] details {
        background-color: #0B0B14;
        border: 1px solid #1F1F33;
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        background-color: #121220;
        padding: 10px 15px;
        color: #E2E8F0;
        font-weight: 600;
        border-bottom: 1px solid #1F1F33;
    }
    [data-testid="stExpander"] summary:hover { background-color: #1A1A2E; }

    /* ═══════════════════════════════════════════════════
       LIVE THINKING BLOCK & EXPANDER CONTENT
       ═══════════════════════════════════════════════════ */
    .live-think-block, .exp-think-content {
        background: linear-gradient(90deg, rgba(255,42,109,0.05), transparent);
        border-left: 4px solid #FF2A6D;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .live-think-body, .exp-think-content p, .exp-think-content li {
        color: #A0AEC0 !important;
        font-size: 0.85rem;
        line-height: 1.6;
        font-style: italic;
    }
    .live-think-body strong { color: #FF2A6D !important; font-style: normal; }

    /* ═══════════════════════════════════════════════════
       SOURCE CARDS
       ═══════════════════════════════════════════════════ */
    .src-card {
        background: linear-gradient(90deg, rgba(5,217,232,0.08), #0B0B14);
        border: 1px solid #1F1F33;
        border-left: 4px solid #05D9E8;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
        transition: all 0.2s;
    }
    .src-card:hover { border-color: #05D9E8; background: linear-gradient(90deg, rgba(5,217,232,0.12), #121220); }
    .src-name { font-size: 0.85rem; font-weight: 700; color: #FFFFFF; }
    .src-pg { 
        font-size: 0.7rem; color: #05050A; font-weight: 700; 
        background: #05D9E8; padding: 2px 6px; border-radius: 4px; margin-left: 8px; 
    }
    .src-prev { font-size: 0.8rem; color: #94A3B8; display: block; margin-top: 6px; border-top: 1px solid #1F1F33; padding-top: 6px; }

    /* ═══════════════════════════════════════════════════
       CHAT INPUT FIELD
       ═══════════════════════════════════════════════════ */
    [data-testid="stChatInput"] {
        background-color: #0A0A14;
        border: 2px solid #1F1F33;
        border-radius: 12px;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.5);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #FF2A6D;
        box-shadow: 0 0 15px rgba(255,42,109,0.3);
    }
    [data-testid="stChatInput"] textarea { color: #FFFFFF; font-size: 1rem; }

    /* ═══════════════════════════════════════════════════
       TYPING CURSOR & SPINNERS
       ═══════════════════════════════════════════════════ */
    .cur {
        color: #05D9E8;
        text-shadow: 0 0 8px rgba(5,217,232,0.8);
        animation: curblink 0.8s steps(1) infinite;
    }
    @keyframes curblink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }

    [data-testid="stSpinner"] > div { border-top-color: #05D9E8; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def check_api_connection() -> bool:
    try:
        resp = requests.get(f"{st.session_state.api_base}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="main-hdr" style="font-size:1.2rem; margin-bottom:2px;">'
            "🔬 Research Assistant</div>",
            unsafe_allow_html=True,
        )
        st.caption("RAG-powered local LLM research")

        online = check_api_connection()
        dot_cls = "dot-on" if online else "dot-off"
        dot_label = "Backend Connected" if online else "Backend Offline"
        st.markdown(
            f'<div style="margin-top: 10px;"><span class="dot {dot_cls}"></span>'
            f'<span class="dot-text">{dot_label}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="sb-sep">', unsafe_allow_html=True)
        st.markdown('<p class="sb-label">Conversations</p>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-new-chat">', unsafe_allow_html=True)
        if st.button("➕ New Chat", use_container_width=True, type="secondary"):
            create_new_chat()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        for cid, chat_data in reversed(list(st.session_state.chat_store.items())):
            is_active = cid == st.session_state.conversation_id
            icon = "💬" if is_active else "🗨️"

            # The CSS will now visually merge these columns into a single card
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                if st.button(
                    f"{icon} {chat_data['title']}",
                    key=f"sel_{cid}",
                    use_container_width=True,
                ):
                    st.session_state.conversation_id = cid
                    st.rerun()
            with col2:
                # Replaced the "x" with the 🗑️ emoji and removed the div wrapper
                if st.button("🗑️", key=f"del_{cid}", help="Delete chat"):
                    del st.session_state.chat_store[cid]
                    save_chat_store(st.session_state.chat_store)
                    api_delete(f"/conversations/{cid}")
                    if cid == st.session_state.conversation_id:
                        if st.session_state.chat_store:
                            st.session_state.conversation_id = list(
                                st.session_state.chat_store.keys()
                            )[-1]
                        else:
                            create_new_chat()
                    st.rerun()

        st.markdown('<hr class="sb-sep">', unsafe_allow_html=True)
        st.markdown('<p class="sb-label">Knowledge Base</p>', unsafe_allow_html=True)

        docs_data = api_get("/documents")
        documents = docs_data.get("documents", [])

        if documents:
            for doc in documents:
                st.markdown(
                    f'<div class="doc-row">'
                    f'📄 <strong style="color:#E2E8F0;">{doc["filename"]}</strong><br>'
                    f'<span style="font-size:0.75rem;">{doc["chunks"]} chunks</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<span class="dot-text">No documents ingested yet.</span>',
                unsafe_allow_html=True,
            )

        st.write("")

        st.markdown('<div class="sidebar-ingest">', unsafe_allow_html=True)
        if st.button("📂 Ingest PDFs", use_container_width=True, type="secondary"):
            with st.spinner("Ingesting documents..."):
                api_post("/ingest")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_chat():
    current_chat = st.session_state.chat_store[st.session_state.conversation_id]
    messages = current_chat["messages"]

    st.markdown(
        f'<div class="main-hdr">{current_chat["title"]}</div>'
        f'<div class="main-hdr-line"></div>',
        unsafe_allow_html=True,
    )

    if not messages:
        st.markdown(
            """
        <div class="empty-wrap">
            <div class="empty-icon">🧬</div>
            <p class="empty-h">Ready to analyze</p>
            <p class="empty-p">
                Query your local documents. The assistant will retrieve evidence 
                and construct answers securely on your hardware.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    for msg in messages:
        with st.chat_message(msg["role"]):
            
            # 1. RENDER THINKING FIRST (If it exists)
            if msg["role"] == "assistant" and msg.get("meta") and msg["meta"].get("thinking"):
                think_time = msg["meta"].get("think_time", 0)
                think_label = f"🧠 Reasoning Chain ({think_time:.1f}s)" if think_time else "🧠 Reasoning Chain"
                
                with st.expander(think_label, expanded=False): # Keep closed in history for clean UI
                    st.markdown(f'<div class="exp-think-content">', unsafe_allow_html=True)
                    st.markdown(msg["meta"].get("thinking"))
                    st.markdown("</div>", unsafe_allow_html=True)

            # 2. RENDER THE ACTUAL ANSWER
            # Strip out raw <think> tags from history if they snuck through
            clean_content = msg["content"]
            if "</think>" in clean_content:
                clean_content = clean_content.split("</think>")[-1].strip()
            
            st.markdown(clean_content)

            # 3. RENDER SOURCES LAST (If they exist)
            if msg["role"] == "assistant" and msg.get("meta") and msg["meta"].get("sources"):
                sources = msg["meta"].get("sources", [])
                with st.expander(f"🔍 Retrieved Citations ({len(sources)})", expanded=False):
                    for src in sources:
                        st.markdown(
                            f'<div class="src-card">'
                            f'<span class="src-name">{src.get("file", "Unknown")}</span>'
                            f'<span class="src-pg">PG {src.get("page", "?")}</span>'
                            f'<span class="src-prev">"{src.get("preview", "")}"</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    user_input = st.chat_input("Query your knowledge base...")

    if user_input:
        current_chat["messages"].append({"role": "user", "content": user_input})
        save_chat_store(st.session_state.chat_store)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if st.session_state.streaming:
                thinking_display = st.empty()
                response_container = st.empty()
                full_text = ""
                is_error = False
                citations_meta = {}
                thinking_text = ""
                think_duration = 0.0
                stream_start_time = time.time()

                for chunk, error, meta, thinking in stream_answer(
                    user_input, st.session_state.conversation_id
                ):
                    if error:
                        full_text = chunk
                        is_error = True
                        break
                    if meta is not None:
                        citations_meta = meta
                        if meta.get("thinking"):
                            thinking_text = meta["thinking"]
                    elif thinking is not None:
                        think_duration = time.time() - stream_start_time
                        clean_think = (
                            thinking
                            .replace("<think>", "")
                            .replace("</think>", "")
                            .strip()
                        )
                        thinking_text = clean_think
                        thinking_display.markdown(
                            f'<div class="live-think-block">🧠 <strong>Thinking ({think_duration:.1f}s)</strong><br><br>{clean_think}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        full_text += chunk
                        response_container.markdown(
                            full_text + '<span class="cur">▌</span>',
                            unsafe_allow_html=True,
                        )

                response_container.markdown(full_text)

                if not is_error:
                    final_thinking = thinking_text or citations_meta.get("thinking", "")
                    if citations_meta.get("thinking"):
                        final_thinking = citations_meta["thinking"]
                    meta_data = {
                        "sources": citations_meta.get("sources", []),
                        "thinking": final_thinking,
                        "think_time": think_duration,
                    }
                    current_chat["messages"].append(
                        {
                            "role": "assistant",
                            "content": full_text,
                            "meta": meta_data,
                        }
                    )
                    save_chat_store(st.session_state.chat_store)
                st.rerun()


def main():
    inject_custom_css()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()