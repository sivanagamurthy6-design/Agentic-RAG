from langgraph.graph import StateGraph, START, END
from state import GraphState
from nodes import hybrid_search_node, llm_invoke, tavily_node
from router import route_tavily_check, route_doc_check


def build_query_graph():

    graph = StateGraph(GraphState)

    graph.add_node("hybrid_search", hybrid_search_node)
    graph.add_node("llm_invoke", llm_invoke)
    graph.add_node("tavily_node", tavily_node)
    #graph.add_node("route_tavily_check", route_tavily_check)

    #graph.add_edge(START, "hybrid_search")

    #graph.add_edge("hybrid_search", "llm_invoke")
    graph.add_conditional_edges(
        START,
        route_doc_check,
        {
            "hybrid_search": "hybrid_search",
            "llm_invoke": "llm_invoke",
        },
    )
    graph.add_edge("hybrid_search", "llm_invoke")
    graph.add_conditional_edges(
        "llm_invoke",
        route_tavily_check,
        {
            "tavily_node": "tavily_node",
            "end": END,
        },
    )

    graph.add_edge("tavily_node", END)

    return graph.compile()

if __name__ == "__main__":
    inputs = {
    "user_query": "Who won the Mens's T20 World cup in 2026?"
        }
    query_graph = build_query_graph()
    final_state = query_graph.invoke(inputs)
    #print("Final state:", final_state)