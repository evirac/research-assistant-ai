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
    header {
        background-color: transparent !important;
    }
    [data-testid="stToolbar"] > div:nth-child(2),
    [data-testid="stToolbar"] > div:nth-child(3) {
        visibility: hidden;
    }

    /* ═══════════════════════════════════════════════════
       BASE
       ═══════════════════════════════════════════════════ */
    .stApp {
        background-color: #0c0c0c;
        color: #e0ddd8;
    }

    /* ═══════════════════════════════════════════════════
       SCROLLBAR
       ═══════════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #444; }

    /* ═══════════════════════════════════════════════════
       SIDEBAR
       ═══════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #1e1e1e;
    }
    [data-testid="stSidebar"]::before {
        content: '';
        display: block;
        height: 2px;
        background: linear-gradient(90deg, #d4943a, #3ab89a, transparent);
        margin-bottom: 12px;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e0ddd8 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #a09a92;
    }

    /* New Chat button */
    .sidebar-new-chat > button {
        background-color: rgba(212,148,58,0.10);
        color: #d4943a;
        border: 1px solid rgba(212,148,58,0.18);
        border-radius: 6px;
    }
    .sidebar-new-chat > button:hover {
        background-color: rgba(212,148,58,0.20);
        border-color: rgba(212,148,58,0.35);
    }

    /* Chat list buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent;
        color: #8a8480;
        border: 1px solid transparent;
        border-radius: 6px;
        text-align: left;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #1a1a1a;
        color: #e0ddd8;
    }

    /* ── DELETE BUTTON ── */
    .sidebar-del,
    .sidebar-del > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
    }
    /* target the actual button regardless of Streamlit wrapper depth */
    .sidebar-del button {
        background-color: transparent !important;
        color: #3a3a3a !important;
        border: none !important;
        border-radius: 4px !important;
        width: 22px !important;
        height: 22px !important;
        min-width: 22px !important;
        max-width: 22px !important;
        padding: 0 !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.8rem !important;
        line-height: 1 !important;
        transition: all 0.15s ease;
    }
    .sidebar-del button:hover {
        color: #e05050 !important;
        background-color: rgba(220,60,60,0.12) !important;
    }

    /* Ingest button */
    .sidebar-ingest > button {
        background-color: rgba(58,184,154,0.08);
        color: #3ab89a;
        border: 1px solid rgba(58,184,154,0.25);
        border-radius: 6px;
    }
    .sidebar-ingest > button:hover {
        background-color: rgba(58,184,154,0.16);
    }

    .sb-sep {
        border: none;
        height: 1px;
        background: #1e1e1e;
        margin: 14px 0;
    }
    .sb-label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #555;
        margin-bottom: 6px;
    }

    /* Status dot */
    .dot {
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .dot-on {
        background: #3ad46a;
        box-shadow: 0 0 6px rgba(58,212,106,0.5);
        animation: dotpulse 2.5s ease-in-out infinite;
    }
    .dot-off {
        background: #d43a3a;
        box-shadow: 0 0 6px rgba(212,58,58,0.4);
    }
    @keyframes dotpulse {
        0%,100% { box-shadow: 0 0 4px rgba(58,212,106,0.35); }
        50%     { box-shadow: 0 0 12px rgba(58,212,106,0.7); }
    }
    .dot-text {
        font-size: 0.72rem;
        color: #555;
        vertical-align: middle;
    }

    /* Doc entries */
    .doc-row {
        font-size: 0.78rem;
        color: #777;
        padding: 4px 0;
        border-bottom: 1px solid #181818;
    }
    .doc-row:last-child { border-bottom: none; }

    /* ═══════════════════════════════════════════════════
       MAIN AREA
       ═══════════════════════════════════════════════════ */
    .main-hdr {
        color: #e0ddd8;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .main-hdr-line {
        height: 2px;
        background: linear-gradient(90deg, #d4943a 0%, transparent 60%);
        border-radius: 1px;
        opacity: 0.5;
        margin-bottom: 20px;
    }

    .empty-wrap { text-align: center; padding: 60px 20px; }
    .empty-icon { font-size: 2.2rem; opacity: 0.4; }
    .empty-h {
        color: #888; font-size: 1rem; font-weight: 600;
        margin: 10px 0 6px 0;
    }
    .empty-p {
        color: #555; font-size: 0.85rem; max-width: 360px;
        margin: 0 auto; line-height: 1.55;
    }

    /* ═══════════════════════════════════════════════════
       CHAT MESSAGES
       ═══════════════════════════════════════════════════ */
    [data-testid="stChatMessage"] { background-color: transparent; }
    [data-testid="stChatMessage"] div[data-testid="stAvatar"] {
        background-color: #1a1a1a;
    }
    [data-testid="stChatMessage"] p { color: #d8d5d0; }
    [data-testid="stChatMessage"] code {
        background-color: #181818;
        border: 1px solid #252525;
        border-radius: 3px;
        padding: 1px 5px;
        color: #d4943a;
    }
    [data-testid="stChatMessage"] pre {
        background-color: #111111;
        border: 1px solid #222;
        border-radius: 6px;
    }
    [data-testid="stChatMessage"] pre code {
        background-color: transparent;
        border: none;
        padding: 0;
        color: #d8d5d0;
    }
    [data-testid="stChatMessage"] strong { color: #f0ede8; }
    [data-testid="stChatMessage"] a { color: #d4943a; }
    [data-testid="stChatMessage"] li { color: #c8c4be; }

    /* ═══════════════════════════════════════════════════
       EXPANDERS
       ═══════════════════════════════════════════════════ */
    .streamlit-expanderHeader {
        background-color: #161616;
        border: 1px solid #222;
        border-radius: 6px;
        color: #999;
        font-size: 0.8rem;
    }
    .streamlit-expanderHeader:hover {
        border-color: #333;
        color: #ccc;
    }
    [data-testid="stExpander"] details[open] .streamlit-expanderHeader {
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
        border-bottom-color: transparent;
    }
    .streamlit-expanderContent {
        background-color: #161616;
        border: 1px solid #222;
        border-top: none;
        border-bottom-left-radius: 6px;
        border-bottom-right-radius: 6px;
    }
    .exp-think .streamlit-expanderHeader {
        border-left: 3px solid #d4943a;
    }
    .exp-src .streamlit-expanderHeader {
        border-left: 3px solid #3ab89a;
    }
    /* ═══════════════════════════════════════════════════
       THINKING BLOCK (live + expander inner content)
       ═══════════════════════════════════════════════════ */
    /* Live thinking block shown while streaming */
    .live-think-block {
        background-color: #0f0f0f;
        border: 1px solid #252525;
        border-left: 3px solid #d4943a;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
    .live-think-header {
        font-size: 0.78rem;
        font-weight: 600;
        color: #d4943a;
        margin-bottom: 8px;
        letter-spacing: 0.02em;
    }
    .live-think-body {
        color: #777;
        font-size: 0.80rem;
        line-height: 1.6;
        font-style: italic;
    }
    .live-think-body p { color: #777 !important; margin-bottom: 6px; }
    .live-think-body strong { color: #999 !important; font-style: normal; }
    .live-think-body ol, .live-think-body ul { padding-left: 18px; }
    .live-think-body li { color: #777 !important; margin-bottom: 4px; }

    /* Expander inner content for thinking (historical messages) */
    .exp-think .streamlit-expanderContent p { color: #777; font-size: 0.82rem; }
    .exp-think .streamlit-expanderContent strong { color: #999; }
    .exp-think .streamlit-expanderContent li { color: #777; font-size: 0.82rem; }
    .exp-think .streamlit-expanderContent ol,
    .exp-think .streamlit-expanderContent ul { padding-left: 18px; }

    .think-text {
        color: #666;
        font-style: italic;
        font-size: 0.82rem;
    }

    .src-card {
        background-color: #0f0f0f;
        border: 1px solid #1e1e1e;
        border-left: 3px solid #3ab89a;
        border-radius: 5px;
        padding: 8px 10px;
        margin-bottom: 8px;
    }
    .src-card:last-child { margin-bottom: 0; }
    .src-card:hover { border-left-color: #3ab89a; }
    .src-name { font-size: 0.8rem; font-weight: 600; color: #ccc; }
    .src-pg { font-size: 0.7rem; color: #3ab89a; font-weight: 500; margin-left: 6px; }
    .src-prev {
        font-size: 0.75rem; color: #555; font-style: italic;
        display: block; margin-top: 3px;
    }

    /* ═══════════════════════════════════════════════════
       CHAT INPUT
       ═══════════════════════════════════════════════════ */
    [data-testid="stChatInput"] {
        background-color: #111111;
        border: 1px solid #222;
        border-radius: 10px;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #d4943a;
        box-shadow: 0 0 0 3px rgba(212,148,58,0.12);
    }
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea { color: #e0ddd8; }
    [data-testid="stChatInput"] input::placeholder { color: #444; }

    /* ═══════════════════════════════════════════════════
       STREAMING CURSOR
       ═══════════════════════════════════════════════════ */
    .cur {
        color: #d4943a;
        animation: curblink 0.7s steps(1) infinite;
    }
    @keyframes curblink {
        0%,50% { opacity: 1; }
        51%,100% { opacity: 0; }
    }

    /* ═══════════════════════════════════════════════════
       SPINNER & ERROR
       ═══════════════════════════════════════════════════ */
    [data-testid="stSpinner"] > div { border-top-color: #d4943a; }
    .stException {
        background-color: rgba(212,58,58,0.08);
        border: 1px solid rgba(212,58,58,0.25);
        border-radius: 6px;
        color: #f08080;
    }
    .stCaption { color: #555; }
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
            '<div class="main-hdr" style="font-size:1.05rem; margin-bottom:2px;">'
            "🔬 Research Assistant</div>",
            unsafe_allow_html=True,
        )
        st.caption("RAG-powered local LLM research")

        online = check_api_connection()
        dot_cls = "dot-on" if online else "dot-off"
        dot_label = "Backend connected" if online else "Backend offline"
        st.markdown(
            f'<span class="dot {dot_cls}"></span>'
            f'<span class="dot-text">{dot_label}</span>',
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

            col1, col2 = st.columns([0.87, 0.13])
            with col1:
                if st.button(
                    f"{icon} {chat_data['title']}",
                    key=f"sel_{cid}",
                    use_container_width=True,
                ):
                    st.session_state.conversation_id = cid
                    st.rerun()
            with col2:
                st.markdown('<div class="sidebar-del">', unsafe_allow_html=True)
                # Cleaner × character (U+00D7) instead of ✖ (U+2716)
                if st.button("×", key=f"del_{cid}", help="Delete"):
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
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<hr class="sb-sep">', unsafe_allow_html=True)
        st.markdown('<p class="sb-label">Knowledge Base</p>', unsafe_allow_html=True)

        docs_data = api_get("/documents")
        documents = docs_data.get("documents", [])

        if documents:
            for doc in documents:
                st.markdown(
                    f'<div class="doc-row">'
                    f'📄 <strong style="color:#aaa;">{doc["filename"]}</strong>'
                    f' — {doc["chunks"]} chunks</div>',
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
            <div class="empty-icon">🔬</div>
            <p class="empty-h">Start a new inquiry</p>
            <p class="empty-p">
                Ask a question about your ingested documents and the
                assistant will retrieve relevant evidence to construct
                an answer.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                thinking = meta.get("thinking", "")
                sources = meta.get("sources", [])

                if thinking:
                    think_time = msg.get("meta", {}).get("think_time", 0)
                    think_label = (
                        f"🧠 Thought for {think_time:.1f}s"
                        if think_time
                        else "🧠 Reasoning Trace"
                    )
                    with st.expander(think_label, expanded=True):
                        st.markdown(
                            f'<div class="live-think-body">', unsafe_allow_html=True
                        )
                        st.markdown(thinking)
                        st.markdown("</div>", unsafe_allow_html=True)

                if sources:
                    with st.expander(
                        f"🔍 Retrieved Evidence ({len(sources)})", expanded=False
                    ):
                        for src in sources:
                            st.markdown(
                                f'<div class="src-card">'
                                f'<span class="src-name">{src.get("file", "Unknown")}</span>'
                                f'<span class="src-pg">p. {src.get("page", "?")}</span>'
                                f'<span class="src-prev">"{src.get("preview", "")}"</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    user_input = st.chat_input("Ask a question about your documents...")

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
                        # Thinking block just arrived — record elapsed time & render it live
                        think_duration = time.time() - stream_start_time
                        # Strip raw <think></think> wrapper tags if present
                        clean_think = (
                            thinking
                            .replace("<think>", "")
                            .replace("</think>", "")
                            .strip()
                        )
                        thinking_text = clean_think
                        # Render thinking as proper markdown (bullets, bold etc. all work)
                        thinking_display.markdown(
                            f"🧠 **Thought for {think_duration:.1f}s**\n\n"
                            + clean_think
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
                    # Prefer the clean version from citations payload (tags already stripped)
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