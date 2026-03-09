"""
app.py
──────
Streamlit frontend for the Agentic RAG Pipeline.
Run with:  streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from document_loader import load_documents
from graph import run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG Pipeline",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTextArea textarea { font-size: 15px; }
    .source-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 8px;
    }
    .source-hybrid  { background:#0f2818; color:#4ade80; border:1px solid #22c55e; }
    .source-llm     { background:#1a0a2e; color:#c084fc; border:1px solid #a855f7; }
    .source-tavily  { background:#001a1a; color:#22d3ee; border:1px solid #06b6d4; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 Agentic RAG Pipeline")
st.caption("Upload a document (optional) and ask any question. The pipeline auto-routes your query.")

st.divider()

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, source}
if "documents" not in st.session_state:
    st.session_state.documents = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

# ── Sidebar: document upload ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📎 Document Upload")
    st.caption("Upload a file to enable document-based Q&A. Leave empty to use LLM / web search.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt"],
        help="Supported: PDF, DOCX, TXT"
    )

    if uploaded_file:
        # Only re-process if a new file is uploaded
        if uploaded_file.name != st.session_state.uploaded_filename:
            with st.spinner(f"Processing **{uploaded_file.name}**…"):
                # Save to a temp file so document_loader can read it
                suffix = os.path.splitext(uploaded_file.name)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    st.session_state.documents = load_documents([tmp_path])
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.success(f"✅ Loaded **{uploaded_file.name}**  "
                               f"({len(st.session_state.documents)} chunks)")
                except Exception as e:
                    st.error(f"❌ Failed to load file: {e}")
                finally:
                    os.unlink(tmp_path)
        else:
            st.success(f"✅ **{uploaded_file.name}** ready  "
                       f"({len(st.session_state.documents)} chunks)")
    else:
        # User removed the file
        if st.session_state.uploaded_filename:
            st.session_state.documents = []
            st.session_state.uploaded_filename = None
        st.info("No document uploaded — questions will be answered by LLM or Tavily web search.")

    st.divider()

    # Routing legend
    st.markdown("**Routing Legend**")
    st.markdown("🟢 &nbsp;**Hybrid Search** — answer from your document")
    st.markdown("🟣 &nbsp;**LLM** — answer from model knowledge")
    st.markdown("🔵 &nbsp;**Tavily** — answer from live web search")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
source_style = {
    "hybrid_search": ("🟢 Hybrid Search", "source-hybrid"),
    "llm_knowledge": ("🟣 LLM Knowledge", "source-llm"),
    "tavily_search":  ("🔵 Tavily Web Search", "source-tavily"),
}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("source"):
            label, css = source_style.get(msg["source"], ("", ""))
            if label:
                st.markdown(
                    f'<span class="source-badge {css}">{label}</span>',
                    unsafe_allow_html=True,
                )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question…"):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = run_pipeline(
                    user_query=prompt,
                    documents=st.session_state.documents,
                    verbose=False,
                )
                answer = result.get("final_answer", "I could not generate an answer.")
                source = result.get("source", "llm_knowledge")
            except Exception as e:
                answer = f"⚠️ Pipeline error: {e}"
                source = None

        st.markdown(answer)

        if source:
            label, css = source_style.get(source, ("", ""))
            if label:
                st.markdown(
                    f'<span class="source-badge {css}">{label}</span>',
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "source": source,
    })