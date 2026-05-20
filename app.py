"""
IAS RAG Chatbot — Streamlit Interface
Run: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IAS RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #666; font-size: 0.95rem; margin-top: 2px; margin-bottom: 1.5rem; }
    .source-badge {
        display: inline-block; background: #e8f0fe; color: #1a73e8;
        border-radius: 12px; padding: 2px 10px; font-size: 0.78rem;
        margin: 2px; font-weight: 500;
    }
    .chunk-box {
        background: #f8f9fa; border-left: 3px solid #1a73e8;
        padding: 10px 14px; border-radius: 4px; font-size: 0.85rem;
        margin-bottom: 8px; color: #333;
    }
    .stat-card {
        background: #f0f4ff; border-radius: 10px;
        padding: 12px 16px; text-align: center;
    }
    .stat-num { font-size: 1.6rem; font-weight: 700; color: #1a73e8; }
    .stat-lbl { font-size: 0.78rem; color: #666; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role", "content", "sources"}
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "groq"


# ── Load RAG engine (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def get_rag():
    from rag_engine import RAGEngine
    return RAGEngine()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # LLM provider
    st.markdown("### 🤖 LLM Provider")
    provider = st.radio(
        "Choose provider",
        ["Groq (free cloud)", "Ollama (local)"],
        index=0,
        help="Groq is faster on low-spec laptops. Ollama runs fully offline.",
    )
    st.session_state.llm_provider = "groq" if "Groq" in provider else "ollama"

    if st.session_state.llm_provider == "groq":
        api_key = st.text_input(
            "Groq API key",
            type="password",
            placeholder="gsk_...",
            help="Free at console.groq.com",
        )
        groq_model = st.selectbox(
            "Model",
            ["llama3-8b-8192", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0,
        )
    else:
        api_key = None
        ollama_model = st.selectbox(
            "Ollama model",
            ["gemma:2b", "phi3:mini", "llama3.2:1b", "tinyllama"],
            index=0,
            help="Run 'ollama pull gemma:2b' first",
        )

    st.divider()

    # PDF upload
    st.markdown("### 📚 Upload IAS Books (PDF)")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("➕ Index uploaded books", use_container_width=True, type="primary"):
            rag = get_rag()
            total_new = 0
            prog = st.progress(0)
            for i, f in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                new = rag.ingest_pdf(tmp_path)
                total_new += new
                os.unlink(tmp_path)
                prog.progress((i + 1) / len(uploaded_files))
            prog.empty()
            if total_new:
                st.success(f"✅ Indexed {total_new} new chunks!")
            else:
                st.info("ℹ️ Books already indexed.")

    st.divider()

    # Stats
    st.markdown("### 📊 Knowledge Base")
    try:
        rag = get_rag()
        sources = rag.list_sources()
        chunks  = rag.total_chunks()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{len(sources)}</div><div class="stat-lbl">Books</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-num">{chunks}</div><div class="stat-lbl">Chunks</div></div>', unsafe_allow_html=True)

        if sources:
            st.markdown("**Indexed books:**")
            for s in sources:
                st.markdown(f'<span class="source-badge">📄 {s}</span>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading engine: {e}")

    st.divider()

    # Retrieval settings
    st.markdown("### 🔍 Retrieval")
    top_k = st.slider("Chunks to retrieve (k)", 2, 8, 4)
    show_sources = st.toggle("Show source chunks", value=True)

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📚 IAS RAG Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions from your IAS books — powered by local embeddings + free LLMs</p>', unsafe_allow_html=True)

# Quick-start tips if no books indexed
try:
    rag = get_rag()
    if rag.total_chunks() == 0:
        st.info(
            "👋 **Getting started:**  \n"
            "1. Upload your IAS PDFs (NCERT, Laxmikanth, etc.) in the sidebar  \n"
            "2. Click **Index uploaded books**  \n"
            "3. Set your Groq API key (free at [console.groq.com](https://console.groq.com))  \n"
            "4. Start asking questions below!"
        )
except Exception:
    st.warning("⚠️ Could not load RAG engine. Make sure all dependencies are installed.")

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "📚"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("📎 Retrieved chunks", expanded=False):
                for chunk_text, source, score in msg["sources"]:
                    relevance = round((1 - score) * 100, 1)
                    st.markdown(
                        f'<div class="chunk-box">'
                        f'<b>{source}</b> — relevance: {relevance}%<br><br>'
                        f'{chunk_text[:400]}{"..." if len(chunk_text) > 400 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Example questions
if not st.session_state.chat_history:
    st.markdown("**💡 Try asking:**")
    example_qs = [
        "What is Article 370 of the Indian Constitution?",
        "Explain the Panchayati Raj system in India",
        "What are the Fundamental Rights?",
        "Describe the functions of the Election Commission",
    ]
    cols = st.columns(2)
    for i, q in enumerate(example_qs):
        if cols[i % 2].button(q, key=f"eq_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q, "sources": []})
            st.rerun()


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything from your IAS books..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt, "sources": []})

    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Searching books and generating answer..."):
            try:
                rag    = get_rag()
                chunks = rag.retrieve(prompt, k=top_k)

                if not chunks:
                    answer = "⚠️ No books indexed yet. Please upload and index PDFs first."
                elif st.session_state.llm_provider == "groq":
                    if not api_key:
                        answer = "⚠️ Please enter your Groq API key in the sidebar. Get one free at console.groq.com"
                    else:
                        from llm_connector import ask_groq
                        answer = ask_groq(prompt, chunks, api_key, groq_model)
                else:
                    from llm_connector import ask_ollama
                    answer = ask_ollama(prompt, chunks, ollama_model)

            except Exception as e:
                answer  = f"❌ Error: {e}"
                chunks  = []

        st.markdown(answer)

        if show_sources and chunks:
            with st.expander("📎 Retrieved chunks", expanded=False):
                for chunk_text, source, score in chunks:
                    relevance = round((1 - score) * 100, 1)
                    st.markdown(
                        f'<div class="chunk-box">'
                        f'<b>{source}</b> — relevance: {relevance}%<br><br>'
                        f'{chunk_text[:400]}{"..." if len(chunk_text) > 400 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks if chunks else [],
    })
