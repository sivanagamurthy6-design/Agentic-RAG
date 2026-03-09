"""
nodes.py
────────
Every node in the LangGraph pipeline lives here.
Each node is a plain Python function that accepts GraphState and returns
a dict with the keys it wants to update in the state.

Node map (mirrors the visual flow):
  agent_node          →  rewrites the user query
  doc_check_node      →  checks if documents are present
  hybrid_search_node  →  vector + BM25 retrieval over uploaded docs
  validator_node      →  scores whether retrieved context is relevant
  relevance_check_node   → conditional edge helper (not a node, see router fns)
  max_retry_check_node   → conditional edge helper
  llm_node            →  answers from LLM parametric knowledge
  llm_check_node      →  decides if query needs live/recent web data
  tavily_node         →  real-time web search via Tavily
  end_node            →  assembles the final answer
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
#from langchain.retrievers.ensemble import EnsembleRetriever
#from langchain_community.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults


from config import (
    LLM_MODEL, EMBEDDING_MODEL,
    TAVILY_API_KEY, TAVILY_MAX_RESULTS,
    TOP_K, BM25_WEIGHT, VECTOR_WEIGHT,
    RELEVANCE_THRESHOLD, MAX_REWRITE_RETRIES,
)
from state import GraphState

logger = logging.getLogger(__name__)

# ── Shared LLM instance ───────────────────────────────────────────────────────
_llm = ChatGroq (model="llama-3.1-8b-instant")


# ═════════════════════════════════════════════════════════════════════════════
# 1. AGENT NODE  —  query rewriter
# ═════════════════════════════════════════════════════════════════════════════

REWRITE_SYSTEM = """You are an expert query optimisation agent for a RAG pipeline.
Your job is to rewrite the user's query so it retrieves the most relevant documents
from a vector store.

Rules:
- Expand abbreviations and acronyms.
- Add relevant synonyms in parentheses.
- Break compound questions into the most important sub-question.
- Keep the rewritten query under 60 words.
- Return ONLY the rewritten query string, nothing else.
"""

def agent_node(state: GraphState) -> dict:
    """
    Rewrites / optimises the user query for better retrieval.
    On retries (feedback loop) it receives the original query again and
    tries a different expansion strategy.
    """
    query   = state.get("user_query", "")
    retries = state.get("rewrite_count", 0)

    retry_hint = (
        f"\n\nNote: This is rewrite attempt #{retries + 1}. "
        "The previous rewrite did NOT retrieve relevant context. "
        "Try a significantly different angle, vocabulary, or specificity level."
        if retries > 0 else ""
    )

    messages = [
        SystemMessage(content=REWRITE_SYSTEM),
        HumanMessage(content=f"Original query: {query}{retry_hint}"),
    ]

    response = _llm.invoke(messages)
    rewritten = response.content.strip()

    logger.info("[agent_node] rewrite #%d: '%s' → '%s'", retries, query, rewritten)

    return {
        "rewritten_query": rewritten,
        "rewrite_count": retries + 1,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. DOCUMENT CHECK NODE  —  router helper
# ═════════════════════════════════════════════════════════════════════════════

def doc_check_node(state: GraphState) -> dict:
    """
    Checks whether the user supplied any documents in this session.
    Sets doc_present = True / False so the conditional edge can route correctly.
    """
    docs = state.get("documents", [])
    present = bool(docs)
    logger.info("[doc_check_node] doc_present=%s  (%d docs)", present, len(docs))
    return {"doc_present": present}

def _merge_results(
    bm25_docs: list[Document],
    vector_docs: list[Document],
    bm25_weight: float,
    vector_weight: float,
    top_k: int,
) -> list[Document]:
    """
    Manually merges BM25 and vector results using weighted scoring.
    Each doc gets a rank-based score; combined scores are summed, then sorted.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(bm25_docs):
        key = safe_content(doc)[:200]          # use content snippet as unique key
        scores[key]  = scores.get(key, 0.0) + bm25_weight * (1.0 / (rank + 1))
        doc_map[key] = doc

    for rank, doc in enumerate(vector_docs):
        key = safe_content(doc)[:200]
        scores[key]  = scores.get(key, 0.0) + vector_weight * (1.0 / (rank + 1))
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys[:top_k]]


def _ensure_documents(results: list) -> list[Document]:
    """
    Normalise retriever output — some retrievers return (Document, score)
    tuples instead of plain Document objects. This unwraps them safely.
    """
    clean = []
    for item in results:
        if isinstance(item, tuple):
            # (Document, score) tuple — take the Document
            doc = item[0]
        else:
            doc = item
        if hasattr(doc, "page_content"):
            clean.append(doc)
        else:
            logger.warning("[hybrid_search] Skipping unexpected item type: %s", type(item))
    return clean
    
def safe_content(item) -> str:
    """
    Safely extract page_content from a Document or a (Document, score) tuple.
    Never crashes — always returns a string.
    """
    if isinstance(item, tuple):
        item = item[0]
    if hasattr(item, "page_content"):
        return item.page_content
    return str(item)

## after document upload this node should call
# def build_retrievers_node(state: GraphState):

#     docs = state.get("documents", [])

#     if not docs:
#         return {}

#     embeddings = HuggingFaceEmbeddings(
#         model="BAAI/bge-base-en-v1.5"
#     )
#     print("build retriver node is called 1")
#     vector_store = FAISS.from_documents(docs, embeddings)
#     print("build retriver node is called 2 ")
#     vector_retriever = vector_store.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": TOP_K},
#     )
#     print("build retriver node is called 3 ")
#     bm25_retriever = BM25Retriever.from_documents(docs)
#     bm25_retriever.k = TOP_K
#     print("build retriver node is called 4")
#     return {
#         "vector_store": vector_store,
#         "vector_retriever": vector_retriever,
#         "bm25_retriever": bm25_retriever,
#     }
# ═════════════════════════════════════════════════════════════════════════════
# 3. HYBRID SEARCH NODE  —  vector + BM25 ensemble retrieval
# ═════════════════════════════════════════════════════════════════════════════

def hybrid_search_node(state: GraphState) -> dict:
    """
    Performs hybrid (dense vector + sparse BM25) retrieval over the
    user-uploaded documents.

    Dense  : FAISS in-memory vector store with OpenAI embeddings.
    Sparse : BM25Retriever from LangChain community.
    Both   : combined via EnsembleRetriever with configurable weights.
    """
    query = state.get("rewritten_query") or state.get("user_query", "")
    docs  = state.get("documents", [])

    if not docs:
        logger.warning("[hybrid_search_node] No documents available; returning empty.")
        return {"retrieved_docs": []}

    # ── Build vector store ────────────────────────────────────────────────────
    
    embeddings   = HuggingFaceEmbeddings(model="BAAI/bge-base-en-v1.5")
    vector_store     = FAISS.from_documents(docs, embeddings)
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    vector_results = _ensure_documents(vector_retriever.invoke(query))  
    # ── Build BM25 retriever ──────────────────────────────────────────────────
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = TOP_K
    # print("build retriver node is called 5 ")
    # vector_retrievers = state["vector_retriever"]
    # print("build retriver node is called 6 ")
    # bm25_retrievers = state["bm25_retriever"]

    vector_results = _ensure_documents(vector_retriever.invoke(query))
    bm25_results = _ensure_documents(bm25_retriever.invoke(query))
    

   
    # ── Merge ─────────────────────────────────────────────────────────────────
    
    merged = _merge_results(
        bm25_results, vector_results,
        BM25_WEIGHT, VECTOR_WEIGHT, TOP_K,
    )
    logger.info("[hybrid_search_node] retrieved %d merged docs", len(merged))
    return {"retrieved_docs": merged}


# ═════════════════════════════════════════════════════════════════════════════
# 4. RELEVANCE VALIDATOR NODE
# ═════════════════════════════════════════════════════════════════════════════

VALIDATOR_SYSTEM = """You are a strict relevance-grading agent for a RAG pipeline.

Given:
  - A user query
  - A set of retrieved document chunks

Your job:
  1. Assess whether the retrieved context ACTUALLY contains information that
     can answer the query.
  2. Return a JSON object with exactly two keys:
       "relevant": true | false
       "score":    a float between 0.0 (totally irrelevant) and 1.0 (perfectly relevant)

Rules:
  - Be strict. Tangential mentions do NOT count as relevant.
  - If any chunk directly addresses the query, relevant = true.
  - Return ONLY the JSON object, no other text.

Example output:
{"relevant": true, "score": 0.87}
"""

def validator_node(state: GraphState) -> dict:
    """
    Uses an LLM to score whether the retrieved context answers the query.
    Writes context_relevant (bool) and relevance_score (float) to state.
    """
    query    = state.get("rewritten_query") or state.get("user_query", "")
    ret_docs = state.get("retrieved_docs", [])

    if not ret_docs:
        logger.info("[validator_node] No retrieved docs → marking as not relevant.")
        return {"context_relevant": False, "relevance_score": 0.0}

    context_text = "\n\n---\n\n".join(
        f"Chunk {i+1}:\n{safe_content(d)}" for i, d in enumerate(ret_docs)
    )

    user_msg = (
        f"User Query:\n{query}\n\n"
        f"Retrieved Context:\n{context_text}"
    )

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM),
        HumanMessage(content=user_msg),
    ]

    raw = _llm.invoke(messages).content.strip()

    # Parse JSON response
    try:
        parsed        = json.loads(raw)
        is_relevant   = bool(parsed.get("relevant", False))
        score         = float(parsed.get("score", 0.0))
    except (json.JSONDecodeError, ValueError):
        logger.warning("[validator_node] Failed to parse JSON: %s", raw)
        # Fallback: treat any "true" in the response as relevant
        is_relevant = "true" in raw.lower()
        score       = 0.7 if is_relevant else 0.2

    # Apply threshold
    if score >= RELEVANCE_THRESHOLD:
        is_relevant = True
    elif score >= RELEVANCE_THRESHOLD:
        logger.info("[validator_node] Medium relevance detected — accepting context")
        is_relevant = True

    else:
        is_relevant = False
    logger.info(
        "[validator_node] relevant=%s  score=%.2f  (threshold=%.2f)",
        is_relevant, score, RELEVANCE_THRESHOLD,
    )

    return {
        "context_relevant": is_relevant,
        "relevance_score":  score,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. LLM NODE  —  parametric knowledge answer
# ═════════════════════════════════════════════════════════════════════════════

LLM_ANSWER_SYSTEM = """You are a helpful AI assistant.
Answer the user's question clearly and concisely using your training knowledge.
If you are unsure or if the question requires very recent information (after your
knowledge cut-off), say so explicitly instead of guessing.
"""

def llm_node(state: GraphState) -> dict:
    """
    Generates an answer directly from the LLM's parametric knowledge.
    Used in two scenarios:
      • No document was attached (Case 3).
      • Document was attached but context was irrelevant after max retries (Case 2).
    """
    query = state.get("rewritten_query") or state.get("user_query", "")

    messages = [
        SystemMessage(content=LLM_ANSWER_SYSTEM),
        HumanMessage(content=query),
    ]

    answer = _llm.invoke(messages).content.strip()
    logger.info("[llm_node] generated answer (%d chars)", len(answer))

    return {"llm_answer": answer}


# ═════════════════════════════════════════════════════════════════════════════
# 6. LLM CONFIDENCE / RECENCY CHECK NODE
# ═════════════════════════════════════════════════════════════════════════════

RECENCY_SYSTEM = """You are a routing agent for a RAG system.

Determine whether the user's question requires RECENT or LIVE information
that a language model's static training data would NOT reliably contain.

Examples of recent/live queries:
  - "Who won yesterday's match?"
  - "What is the current price of Bitcoin?"
  - "Latest news about X"
  - "What happened in the 2025 election?"

Examples of non-recent queries:
  - "Explain the theory of relativity."
  - "What is Python's GIL?"
  - "Summarise World War II."

Return ONLY a JSON object:
{"is_recent": true | false, "reason": "short explanation"}
"""

def llm_check_node(state: GraphState) -> dict:
    """
    Classifies whether the query needs live/recent web data.
    If is_recent=True the graph will route to Tavily Search.
    Also assembles final_answer from llm_answer if not recent.
    """
    query      = state.get("rewritten_query") or state.get("user_query", "")
    llm_answer = state.get("llm_answer", "")

    query_lower = query.lower()
    
    recent_keywords = [
        "latest", "today", "yesterday", "current", "now",
        "news", "recent", "update", "price", "score","who won",
        "match", "stock", "weather", "2024", "2025", "2026"]

    keyword_recent = any(k in query_lower for k in recent_keywords)

    messages = [
        SystemMessage(content=RECENCY_SYSTEM),
        HumanMessage(content=f"Query: {query}\nLLM Answer: {llm_answer}"),
    ]

    raw = _llm.invoke(messages).content.strip()

    try:
        parsed    = json.loads(raw)
        is_recent = bool(parsed.get("is_recent", False))
    except (json.JSONDecodeError, ValueError):
        is_recent = any(kw in query.lower() for kw in [
            "today", "yesterday", "latest", "current", "now",
            "recent", "live", "price", "news", "2024", "2025",
        ])

    logger.info("[llm_check_node] is_recent=%s for query: '%s'", is_recent, query)

    return {"is_recent_query": is_recent or keyword_recent}


# ═════════════════════════════════════════════════════════════════════════════
# 7. TAVILY SEARCH NODE  —  real-time web retrieval
# ═════════════════════════════════════════════════════════════════════════════

def tavily_node(state: GraphState) -> dict:
    """
    Calls the Tavily Search API to fetch real-time web results.
    Synthesises the results into a coherent answer using the LLM.
    """
    query = state.get("rewritten_query") or state.get("user_query", "")

    # ── Tavily API call ───────────────────────────────────────────────────────
    tool    = TavilySearchResults(
        max_results=TAVILY_MAX_RESULTS,
        tavily_api_key=TAVILY_API_KEY,
    )
    
    results = tool.invoke({"query": query})   # returns list[dict]
    logger.info("[tavily_node] got %d results from Tavily", len(results))

    # ── Synthesise with LLM ───────────────────────────────────────────────────
    context = "\n\n".join(
        f"[{i+1}] {r.get('url','')}\n{r.get('content','')}"
        for i, r in enumerate(results)
    )

    synth_prompt = (
        f"Using ONLY the following web search results, answer this query:\n\n"
        f"Query: {query}\n\n"
        f"Search Results:\n{context}\n\n"
        "Provide a concise, well-structured answer. "
        "Cite sources by their [number] where relevant."
    )

    answer = _llm.invoke([HumanMessage(content=synth_prompt)]).content.strip()
    logger.info("[tavily_node] synthesised answer (%d chars)", len(answer))

    return {
        "tavily_results": results,
        "final_answer":   answer,
        "source":         "tavily_search",
    }


# ═════════════════════════════════════════════════════════════════════════════
# 8. END NODE  —  final answer assembly
# ═════════════════════════════════════════════════════════════════════════════

def end_node(state: GraphState) -> dict:
    """
    Assembles the final answer.

    Priority:
      1. If Tavily already wrote final_answer → use it.
      2. If context was relevant → synthesise from retrieved_docs + rewritten_query.
      3. Otherwise → use the llm_answer.
    """
    # If Tavily already handled it, don't overwrite
    if state.get("source") == "tavily_search" and state.get("final_answer"):
        logger.info("[end_node] Using Tavily answer.")
        return {}

    query      = state.get("rewritten_query") or state.get("user_query", "")
    ret_docs   = state.get("retrieved_docs",  [])
    llm_answer = state.get("llm_answer",      "")

    if state.get("context_relevant") and ret_docs:
        # ── Build answer from retrieved context ───────────────────────────────
        context_text = "\n\n---\n\n".join(
            f"Passage {i+1}:\n{safe_content(d)}" for i, d in enumerate(ret_docs)
        )
        prompt = (
            f"Using ONLY the following passages from the document, "
            f"answer the question.\n\n"
            f"Question: {query}\n\n"
            f"Passages:\n{context_text}\n\n"
            "Give a clear, well-structured answer."
        )
        answer = _llm.invoke([HumanMessage(content=prompt)]).content.strip()
        source = "hybrid_search"
        logger.info("[end_node] Answer from hybrid search context.")
    else:
        # ── Fall back to LLM parametric answer ───────────────────────────────
        answer = llm_answer or "I could not find a relevant answer."
        source = "llm_knowledge"
        logger.info("[end_node] Answer from LLM parametric knowledge.")

    return {
        "final_answer": answer,
        "source":       source,
    }