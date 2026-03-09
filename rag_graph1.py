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
from langchain_core.prompts import ChatPromptTemplate
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

#state schema

class GraphState(TypedDict):
    user_query: str
    rewritten_query: Optional[str]
    documents_attached: bool
    retrieved_docs: Optional[List[str]]
    web_results: Optional[str]
    llm_response: Optional[str]
    final_answer: Optional[str]


# Node 1-agent - router to choose between hybrid search and llm generate

def agent_router(state: GraphState):

    query=state["user_query"]
    prompt = ChatPromptTemplate.from_template(
        """
Rewrite the user query to improve retrieval quality.

User Query:
{query}

Rewritten Query:
"""
    )

    chain = prompt | llm

    rewritten_query = chain.invoke({"query": query}).content

    return {
        "rewritten_query": rewritten_query
    }

    if state["documents_attached"]:
        return "hybrid_search"
    else:
        return "llm_generate"

#Node 2- Hybrid search

def hybrid_search(state: GraphState):

    query = state["user_query"]

    # Example placeholder
    retrieved_docs = ["doc chunk 1", "doc chunk 2"]

    return {
        "retrieved_docs": retrieved_docs
    }

# summarise the agent if any context found in the vector db
def summarize_agent(state: GraphState):

    docs = state["retrieved_docs"]
    query = state["user_query"]

    summary = f"Summarized answer based on docs: {docs}"

    return {
        "final_answer": summary
    }
# if no doc attached go to llm 
def llm_generate(state: GraphState):

    query = state["user_query"]

    response = f"LLM response for: {query}"

    return {
        "llm_response": response,
        "final_answer": response
    }
# tavily node

tavily = TavilySearchResults()

def tavily_search(state: GraphState):

    query = state["user_query"]

    results = tavily.invoke(query)

    return {
        "web_results": str(results)
    }
# web summarisation agent 
def web_summarization(state: GraphState):

    web_data = state["web_results"]

    summary = f"Summary of web results: {web_data}"

    return {
        "final_answer": summary
    }
# conditional logic 

def route_documents(state: GraphState):

    if state["documents_attached"]:
        return "hybrid_search"
    else:
        return "llm_generate"

def route_web_search(state: GraphState):

    query = state["user_query"]

    if "latest" in query or "news" in query:
        return "tavily_search"

    return "llm_generate"
#graph
builder = StateGraph(GraphState)

builder.add_node("agent_router", agent_router)
builder.add_node("hybrid_search", hybrid_search)
builder.add_node("summarize_agent", summarize_agent)
builder.add_node("llm_generate", llm_generate)
builder.add_node("tavily_search", tavily_search)
builder.add_node("web_summarization", web_summarization)

# edges
builder.set_entry_point("agent_router")
builder.add_conditional_edges(
    "agent_router",
    route_documents,
    {
        "hybrid_search": "hybrid_search",
        "llm_generate": "llm_generate"
    }
)
builder.add_edge("hybrid_search", "summarize_agent")
builder.add_edge("summarize_agent", END)
builder.add_edge("llm_generate", END)
builder.add_edge("tavily_search", "web_summarization")
builder.add_edge("web_summarization", END)

graph = builder.compile()

result = graph.invoke({
    "user_query": "Explain hybrid search",
    "documents_attached": True
})

print(result["final_answer"])