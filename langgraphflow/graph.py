"""
graph.py
────────
Builds and compiles the full LangGraph StateGraph that mirrors the visual
flow diagram exactly.

Graph topology
══════════════

  START
    │
    ▼
  agent_node          ← also receives feedback from max_retry_check
    │
    ▼
  doc_check_node
    │
    ├─[doc present]──────────────────────────────┐
    │                                             ▼
    │                                       hybrid_search_node
    │                                             │
    │                                             ▼
    │                                       validator_node
    │                                             │
    │                                             ▼
    │                                       relevance_check  ──[relevant]──► END
    │                                             │
    │                                        [not relevant]
    │                                             │
    │                                             ▼
    │                                       max_retry_check
    │                                        │         │
    │                              [retries left]   [max retries]
    │                                   │                │
    │                            ◄──────┘                │
    │                         (back to agent)             │
    │                                                     ▼
    └─[no doc]──────────────────────────────────► llm_node
                                                         │
                                                         ▼
                                                   llm_check_node
                                                    │         │
                                               [llm ok]   [recent query]
                                                   │           │
                                                   ▼           ▼
                                                  END      tavily_node
                                                               │
                                                               ▼
                                                              END
"""

from __future__ import annotations
import logging
from langgraph.graph import StateGraph, END

from state import GraphState
from nodes import (
    agent_node,
    doc_check_node,
    #build_retrievers_node,
    hybrid_search_node,
    validator_node,
    llm_node,
    llm_check_node,
    tavily_node,
    end_node
)
from routers import (
    route_doc_check,
    route_relevance_check,
    route_max_retry,
    route_llm_check,
)

logger = logging.getLogger(__name__)
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv
# load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(
#     groq_api_key=groq_api_key,
#     model_name="llama-3.1-8b-instant",
#     temperature=0)

def build_graph() -> StateGraph:
    """
    Constructs, wires, and compiles the LangGraph pipeline.
    Returns a compiled graph ready to invoke.
    """

    # ── 1. Initialise the graph with our shared state schema ──────────────────
    workflow = StateGraph(GraphState)



    # ── 2. Register every node ────────────────────────────────────────────────
    workflow.add_node("agent",        agent_node)
    workflow.add_node("doc_check",    doc_check_node)
    # workflow.add_node("build_retrievers", build_retrievers_node)
    workflow.add_node("hybrid_search",hybrid_search_node)
    workflow.add_node("validator",    validator_node)
    workflow.add_node("max_retry_gate", lambda s: {}) 
    workflow.add_node("llm",          llm_node)
    workflow.add_node("llm_check",    llm_check_node)
    workflow.add_node("tavily",       tavily_node)
    workflow.add_node("end",          end_node)

    # ── 3. Entry point ────────────────────────────────────────────────────────
    workflow.set_entry_point("agent")

    # ── 4. Fixed (unconditional) edges ────────────────────────────────────────
    workflow.add_edge("agent",         "doc_check")
    # workflow.add_edge("build_retrievers", "hybrid_search")
    workflow.add_edge("hybrid_search", "validator")   # → see conditional below
    workflow.add_edge("llm",           "llm_check")
    workflow.add_edge("tavily",        "end")
    workflow.add_edge("end",           END)

    # ── 5. Conditional edges (routers) ────────────────────────────────────────

    # Router A: doc_check → hybrid_search  OR  llm
    workflow.add_conditional_edges(
        "doc_check",
        route_doc_check,
        {
            "hybrid_search": "hybrid_search",
            "llm":           "llm",
        },
    )

    # We need a lightweight pass-through node for the relevance gate
    # (LangGraph conditional edges must originate from a real node)
    # validator_node already sets context_relevant, so we hang the router
    # directly on validator output:
    workflow.add_conditional_edges(
        "validator",
        route_relevance_check,
        {
            "end":             "end",
            "max_retry_check": "max_retry_gate",   # → see node below
        },
    )

    # Router B: max_retry_gate → agent (rewrite)  OR  llm (escalate)
    # We use a tiny pass-through node so the router has a proper origin node.
    workflow.add_conditional_edges(
        "max_retry_gate",
        route_max_retry,
        {
            "agent": "agent",
            "llm":   "llm",
        },
    )

    # Router C: llm_check → end  OR  tavily
    workflow.add_conditional_edges(
        "llm_check",
        route_llm_check,
        {
            "end":    "end",
            "tavily": "tavily",
        },
    )

    # ── 6. Compile ────────────────────────────────────────────────────────────
    return workflow.compile()


# ── Convenience run function ──────────────────────────────────────────────────

def run_pipeline(
    user_query: str,
    documents: list | None = None,
    verbose: bool = True,
) -> dict:
    """
    High-level entry point.  Call this from your application.

    Parameters
    ----------
    user_query : str
        The question from the user.
    documents  : list[Document] | None
        LangChain Document objects (e.g. loaded from a PDF / DOCX / TXT).
        Pass None or [] if no document is attached.
    verbose    : bool
        Print intermediate state after each node (useful for debugging).

    Returns
    -------
    dict  –  the final GraphState including final_answer and source.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    graph = build_graph()
   

    

    initial_state: GraphState = {
        "user_query":    user_query,
        "documents":     documents or [],
        "rewrite_count": 0,
    }

    logger.info("\n%s\nStarting pipeline for query: %r\n%s",
                "═" * 60, user_query, "═" * 60)

    final_state = graph.invoke(initial_state)

    if verbose:
        print("\n" + "═" * 60)
        print("✅  FINAL ANSWER")
        print("─" * 60)
        print(final_state.get("final_answer", "No answer generated."))
        print(f"\n📌  Source: {final_state.get('source', 'unknown')}")
        print("═" * 60 + "\n")

    return final_state


