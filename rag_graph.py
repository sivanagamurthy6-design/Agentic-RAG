import streamlit as st
import os
import fitz
from tempfile import NamedTemporaryFile
from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools.tavily_search import TavilySearchResults

from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# =========================
# API KEYS
# =========================

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# =========================
# MODELS
# =========================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

reranker = CrossEncoder("BAAI/bge-reranker-base")

web_search = TavilySearchResults(max_results=5)

# =========================
# STREAMLIT STATE
# =========================

if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None


# =========================
# DOCUMENT LOADING
# =========================

def parse_pdf(path):
    docs = []
    pdf = fitz.open(path)

    for i, page in enumerate(pdf):
        text = page.get_text().strip()

        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": i+1}
                )
            )

    return docs


def build_vector_store(files):

    docs = []

    for f in files:

        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            path = tmp.name

        docs.extend(parse_pdf(path))
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    faiss = FAISS.from_documents(chunks, embeddings)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 8

    return faiss, bm25


# =========================
# HYBRID RETRIEVAL
# =========================

def hybrid_retrieve(query):

    if st.session_state.faiss_store is None:
        return []

    dense_docs = st.session_state.faiss_store.similarity_search(query, k=8)
    sparse_docs = st.session_state.bm25.invoke(query)

    merged = dense_docs + sparse_docs

    pairs = [(query, d.page_content) for d in merged]

    scores = reranker.predict(pairs)

    ranked = [
        doc for _, doc in sorted(
            zip(scores, merged),
            key=lambda x: x[0],
            reverse=True
        )
    ]

    return ranked[:5]


# =========================
# GRAPH STATE
# =========================

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    docs: List[Document]
    docs_uploaded: bool




def check_docs(state: GraphState):

    docs_exist = st.session_state.faiss_store is not None

    return {
        "docs_uploaded": docs_exist
    }

def retrieve_context_node(state: GraphState):

    query = state["question"]

    docs = hybrid_retrieve(query)

    context = "\n\n".join(d.page_content for d in docs)

    return {
        "docs": docs,
        "context": context
    }


# =========================
# GRAPH NODES
# =========================

def rewrite_query(state: GraphState):

    q = state["question"]

    response = llm.invoke(
        f"Rewrite for retrieval:\n{q}"
    )

    return {"rewritten": response.content}


def retrieve_docs(state: GraphState):

    query = state["rewritten"]

    docs = hybrid_retrieve(query)

    return {"docs": docs}


def rag_generate(state: GraphState):

    context = state["context"]
    question = state["question"]

    prompt = f"""
Use the following document context to answer the question.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def llm_direct(state: GraphState):

    question = state["question"]

    response = llm.invoke(question)

    return {
        "answer": response.content
    }


def tavily_answer(state: GraphState):

    text = state["answer"].lower()

    if "knowledge cutoff" not in text:
        return {}

    results = web_search.invoke(state["question"])

    context = "\n\n".join(
        r["content"] for r in results
    )

    response = llm.invoke(
        f"Use this web info:\n{context}\n\nQuestion:{state['question']}"
    )

    return {"answer": response.content}


# =========================
# BUILD GRAPH
# =========================

graph = StateGraph(GraphState)

graph.add_node("rewrite", rewrite_query)
graph.add_node("retrieve", retrieve_docs)
graph.add_node("rag_generate", rag_generate)
graph.add_node("rag", rag_answer)
graph.add_node("llm", llm_answer)
graph.add_node("tavily", tavily_answer)

graph.set_entry_point("rewrite")

graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "rag")
graph.add_edge("rag", "llm")
graph.add_edge("llm", "tavily")
graph.add_edge("tavily", END)

graph = graph.compile()


# =========================
# STREAMLIT UI
# =========================

st.title("Hybrid RAG System")

files = st.file_uploader(
    "Upload Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Build Index"):

    faiss, bm25 = build_vector_store(files)

    st.session_state.faiss_store = faiss
    st.session_state.bm25 = bm25

    st.success("Vector store created")


question = st.chat_input("Ask a question")

if question:

    result = graph.invoke({
        "question": question
    })

    st.write(result["answer"])