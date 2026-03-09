"""
state.py
────────
Shared TypedDict state that flows through every node in the graph.
LangGraph passes this dict between nodes; each node reads what it needs
and writes back its outputs.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    user_query: str                  # original question from the user
    documents: list[Any]             # list of LangChain Document objects (uploaded by user)

    # ── Agent (query rewriter) ─────────────────────────────────────────────────
    rewritten_query: str             # query after the agent has optimised it
    rewrite_count: int               # how many times we have rewritten (for retry limit)

    # ── Document check ─────────────────────────────────────────────────────────
    doc_present: bool                # True if the user uploaded at least one document

    # ── Hybrid search ──────────────────────────────────────────────────────────
    retrieved_docs: list[Any]        # top-k docs returned by hybrid search

    # ── Relevance validation ───────────────────────────────────────────────────
    context_relevant: bool           # True if validator says context answers the query
    relevance_score: float           # 0.0 – 1.0 confidence score from validator

    # ── LLM answer ────────────────────────────────────────────────────────────
    llm_answer: str                  # raw answer produced by the LLM node
    is_recent_query: bool            # True if the question needs live / recent web data

    # ── Tavily search ──────────────────────────────────────────────────────────
    tavily_results: list[dict]       # raw results from Tavily API

    # ── Final ─────────────────────────────────────────────────────────────────
    final_answer: str                # the answer that will be returned to the user
    source: str                      # which node produced the final answer
    vector_store: Any
    bm25_retriever: Any