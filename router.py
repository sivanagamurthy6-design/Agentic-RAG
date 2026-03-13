from state import GraphState
from langgraph.graph import END
def route_doc_check(state: GraphState) -> str:
        """
        After build_embeddings:
        • if docs found → hybrid_search
        • else        → llm_invoke
        """
        doc_present = state.get("doc_present")
        print(f"Document check - doc_present: {doc_present}")
        if state.get("doc_present"):
            print("Documents found. Routing to hybrid search.")
            return "hybrid_search"
        print("No documents found. Routing to LLM.")
        return "llm_invoke"
    # this was the original doc check node logic before we moved it to router.py
    # With this we were checking for the presence of documents after the embedding node and routing accordingly. However, we want to move this logic to the router to keep the nodes focused on their specific tasks and make the graph more modular.
    # without build_embeddings node
    # """
    # After doc_check_node:
    #   • doc present  →  hybrid_search (Case 1 & 2)
    #   • no doc       → llm             (Case 3)
    # """
    # if state.get("doc_present"):
    #     #rint("Documents found. Routing to hybrid search.")
    #     return "hybrid_search"
    # #rint("No documents found. Routing to LLM.")
    # return "llm_invoke"
#define tavily check node to check if the user query is related to tavily or not
def route_tavily_check(state: GraphState) -> str:
    """
    After llm_invoke:
      • if query is related to tavily → tavily_node
      • else                         → END
    """
    """if LLM knowledge cut-off is before 2024, simply use tavily_node for queries mentioning 'tavily' since the LLM won't have info on it."""
    user_query = state.get("user_query", "").lower()
    answer = state.get("final_answer", "").lower()
    failure_signals = [
        "insufficient_context",
        "unfotunate",
        "future",
        "i do not have information",
        "don't know",
        "not enough information",
        "cannot answer",
        "no relevant information",
        "cutoff",
        "knowledge gap",
        "not sure",
        "unfortunately", "unable","access the real","time","current","date and time",
          ]
    #print(f"Routing check - User query: '{user_query}', LLM answer: '{answer}'")
    for words in answer.split():
        if words in failure_signals:
            #print(f"LLM answer contains failure signal '{words}'. Routing to Tavily node.")
            return "tavily_node"
        #print(f"Checked word '{words}' in LLM answer; no failure signal detected.")
    if any(signal in answer for signal in failure_signals):
        #print("LLM indicates insufficient context. Routing to Tavily node.")
        return "tavily_node"
    if "tavily" in user_query:
        #print("User query mentions Tavily. Routing to Tavily node.")
        return "tavily_node"
    #print("No need for Tavily. Routing to END.")
    return "end"