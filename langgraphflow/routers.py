"""
routers.py
──────────
Conditional-edge functions for the LangGraph pipeline.
Each function receives the current GraphState and returns a string
that LangGraph uses to decide which node to visit next.

Router map:
  route_doc_check     →  "build_retrievers"  |  "llm"
  route_relevance     →  "end"            |  "max_retry_check"
  route_max_retry     →  "agent"          |  "llm"
  route_llm_check     →  "end"            |  "tavily"
"""

from __future__ import annotations

import logging
from state import GraphState
from config import MAX_REWRITE_RETRIES

logger = logging.getLogger(__name__)


# ── Router 1: Document Check ──────────────────────────────────────────────────
def route_doc_check(state: GraphState) -> str:
    """
    After doc_check_node:
      • doc present  →  hybrid_search (Case 1 & 2)
      • no doc       → llm             (Case 3)
    """
    if state.get("doc_present"):
        logger.info("[router] doc_check → hybrid_search")
        return "hybrid_search"
    logger.info("[router] doc_check → llm")
    return "llm"


# ── Router 2: Relevance / Context Check ──────────────────────────────────────
def route_relevance_check(state: GraphState) -> str:
    """
    After validator_node:
      • context relevant → end               (Case 1 happy path)
      • not relevant     → max_retry_check   (check if we should rewrite or escalate)
    """
    if state.get("context_relevant"):
        logger.info("[router] relevance_check → end  (context is relevant)")
        return "end"
    logger.info("[router] relevance_check → max_retry_check  (context NOT relevant)")
    return "max_retry_check"


# ── Router 3: Max Retry Gate ──────────────────────────────────────────────────
def route_max_retry(state: GraphState) -> str:
    """
    After relevance check fails:
      • retries < MAX_REWRITE_RETRIES → agent   (rewrite the query and try again)
      • retries >= MAX_REWRITE_RETRIES → llm    (give up on doc, escalate to LLM)

    This implements the feedback loop back to the agent node.
    """
    retries = state.get("rewrite_count", 0)
    if retries < MAX_REWRITE_RETRIES:
        logger.info(
            "[router] max_retry → agent  (retry %d/%d)",
            retries, MAX_REWRITE_RETRIES,
        )
        return "agent"
    logger.info(
        "[router] max_retry → llm  (max retries %d reached, escalating)",
        MAX_REWRITE_RETRIES,
    )
    return "llm"


# ── Router 4: LLM Recency Check ───────────────────────────────────────────────
def route_llm_check(state: GraphState) -> str:
    """
    After llm_check_node:
      • is_recent_query = False  → end     (LLM answer is sufficient)
      • is_recent_query = True   → tavily  (need live web data)
    """
    if state.get("is_recent_query"):
        logger.info("[router] llm_check → tavily  (recent/live query detected)")
        return "tavily"
    logger.info("[router] llm_check → end  (LLM answer is sufficient)")
    return "end"