import streamlit as st
import os
import time
import fitz
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# API Keys
# -------------------------

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# -------------------------
# Page Config
# -------------------------

st.set_page_config(page_title="Hybrid RAG", page_icon="🔍", layout="wide")
st.title("🔍 Hybrid RAG — Dense + Sparse + Reranking")

# -------------------------
# Cached Resources
# -------------------------

@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-base")

llm = load_llm()
embeddings = load_embeddings()
reranker = load_reranker()

web_search = TavilySearchResults(max_results=5) if tavily_api_key else None

# -------------------------
# Query Rewriting
# -------------------------

def rewrite_query(question: str) -> str:
    try:
        prompt = f"""
Rewrite the following question to improve document retrieval.
Do NOT answer the question.

Question: {question}
"""
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        return rewritten if rewritten else question
    except:
        return question

# -------------------------
# Prompt
# -------------------------

prompt = ChatPromptTemplate.from_template(
"""
Use the context below to answer the question.

<context>
{context}
</context>

Conversation history:
{history}

Question: {question}

Answer:
"""
)

# -------------------------
# Session State
# -------------------------

for key, default in [
    ("faiss_store", None),
    ("bm25_retriever", None),
    ("all_chunks", []),
    ("chat_history", []),
    ("last_question", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------
# Document Parsing
# -------------------------

def parse_pdf(path, source):
    docs = []
    pdf = fitz.open(path)
    for page_num, page in enumerate(pdf):
        text = page.get_text().strip()
        if text:
            docs.append(Document(page_content=text,
            metadata={"source": source, "page": page_num+1}))
    pdf.close()
    return docs

def parse_docx(path, source):
    loader = Docx2txtLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = source
    return docs

def parse_txt(file, source):
    text = file.read().decode("utf-8", errors="ignore").strip()
    if text:
        return [Document(page_content=text, metadata={"source":source})]
    return []

# -------------------------
# Build Hybrid Index
# -------------------------

def build_stores(uploaded_files):

    all_docs=[]

    for f in uploaded_files:
        name=f.name.lower()

        if name.endswith(".pdf"):
            with NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:
                tmp.write(f.read())
                path=tmp.name
            all_docs.extend(parse_pdf(path,f.name))
            os.remove(path)

        elif name.endswith(".docx"):
            with NamedTemporaryFile(delete=False,suffix=".docx") as tmp:
                tmp.write(f.read())
                path=tmp.name
            all_docs.extend(parse_docx(path,f.name))
            os.remove(path)

        elif name.endswith(".txt"):
            all_docs.extend(parse_txt(f,f.name))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_docs)

    with ThreadPoolExecutor(max_workers=2) as executor:
        faiss_future = executor.submit(FAISS.from_documents,chunks,embeddings)
        bm25_future = executor.submit(BM25Retriever.from_documents,chunks)

        faiss_store = faiss_future.result()
        bm25 = bm25_future.result()

    bm25.k=8

    st.session_state.faiss_store = faiss_store
    st.session_state.bm25_retriever = bm25
    st.session_state.all_chunks = chunks

    return True

# -------------------------
# Hybrid Retrieval
# -------------------------

def hybrid_retrieve(query,top_k=5):

    faiss_retriever = st.session_state.faiss_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":8,"fetch_k":20}
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(faiss_retriever.invoke,query)
        sparse_future = executor.submit(st.session_state.bm25_retriever.invoke,query)

        dense_docs = dense_future.result()
        sparse_docs = sparse_future.result()

    merged=[]
    seen=set()

    for d in dense_docs + sparse_docs:
        key=d.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(d)

    if not merged:
        return []

    pairs=[(query,d.page_content) for d in merged]
    scores=reranker.predict(pairs)

    reranked=[
        doc for _,doc in sorted(zip(scores,merged),key=lambda x:x[0],reverse=True)
    ]

    return reranked[:top_k]

# -------------------------
# LLM needs web check
# -------------------------

def llm_needs_web(text):
    keywords=[
        "knowledge cutoff",
        "no real-time",
        "cannot access the internet",
        "as of my last update"
    ]
    return any(k in text.lower() for k in keywords)

# -------------------------
# Tavily Search
# -------------------------

def tavily_answer(question):

    if web_search is None:
        return "", "no_tavily"

    results=web_search.invoke(question)

    context="\n\n".join(
        r.get("content","") for r in results if isinstance(r,dict)
    )

    if not context:
        return "", "tavily_empty"

    response=llm.invoke(
        f"Use these search results to answer:\n\n{context}\n\nQuestion:{question}"
    )

    return response.content,"🌐 Tavily Web Search"

# -------------------------
# Main Answer Logic
# -------------------------

def answer_question(question,docs):

    # 1️⃣ RAG
    if docs:
        context="\n\n".join(d.page_content for d in docs)

        chain = prompt | llm

        response = chain.invoke({
            "context":context,
            "history":format_history(),
            "question":question
        })

        rag_answer=response.content.strip()

        if rag_answer:
            return rag_answer,f"📄 Hybrid RAG ({len(docs)} chunks)"

    # 2️⃣ LLM
    llm_response=llm.invoke(
        f"Answer the question:\n\n{question}"
    )

    llm_text=llm_response.content.strip()

    if not llm_needs_web(llm_text):
        return llm_text,"🧠 LLM Knowledge"

    # 3️⃣ Tavily
    tavily_text,source=tavily_answer(question)

    if tavily_text:
        return tavily_text,source

    return llm_text,"🧠 LLM fallback"

# -------------------------
# History Formatting
# -------------------------

def format_history(max_turns=3):

    recent=st.session_state.chat_history[-(2*max_turns):]

    lines=[]

    for msg in recent:
        role="User" if msg["role"]=="user" else "Assistant"
        lines.append(f"{role}:{msg['content']}")

    return "\n".join(lines)

# -------------------------
# UI
# -------------------------

col1,col2=st.columns([1,2])

with col1:

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "PDF DOCX TXT",
        type=["pdf","docx","txt"],
        accept_multiple_files=True
    )

    if st.button("Build Index"):

        if uploaded_files:

            with st.spinner("Indexing..."):

                build_stores(uploaded_files)

            st.success("Index built")

with col2:

    st.subheader("Chat")

    # Display chat history
    for msg in st.session_state.chat_history:

        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])

        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                st.caption(f"{msg.get('source','')} • {msg.get('elapsed','?')}s")

    # User input
    question = st.chat_input("Ask question")

    if question and question != st.session_state.last_question:

        st.session_state.last_question = question

        start = time.perf_counter()

        docs = []

        if st.session_state.faiss_store:
            rewritten = rewrite_query(question)
            docs = hybrid_retrieve(rewritten)

        answer, source = answer_question(question, docs)

        elapsed = round(time.perf_counter() - start, 2)

        st.session_state.chat_history.append({"role":"user","content":question})

        st.session_state.chat_history.append({
            "role":"assistant",
            "content":answer,
            "source":source,
            "elapsed":elapsed,
            "docs":docs
        })

        st.rerun()