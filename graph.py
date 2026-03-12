from nodes import *
from router import *
from langgraph.graph import StateGraph,START, END
from state import GraphState

# we need to build a RAG graph that can handle the following steps:
# 1. Load documents from various file types (PDF, DOCX, TXT, CSV, MD)
# 2. Split documents into chunks
# 3. Build embeddings for the chunks
# 4. Store embeddings in a vector database (Pinecone)
def build_rag_graph():

    graph = StateGraph(GraphState)

    graph.add_node("load_and_split", load_documents)
    graph.add_node("build_embeddings", build_embeddings)
    graph.add_node("hybrid_search", hybrid_search_node)
    graph.add_node("llm_invoke", llm_invoke)
    graph.add_node("tavily_node", tavily_node)
    graph.add_node("route_tavily_check", route_tavily_check)

    graph.add_edge(START, "load_and_split")
    graph.add_edge("load_and_split", "build_embeddings")

    graph.add_conditional_edges(
        "build_embeddings",
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
    }
        )
    #raph.add_edge("llm_invoke", "route_tavily_check")
    graph.add_edge("tavily_node", END)

    return graph.compile()
#need to compile grapg if required print the graph
#we need to check how the graph is connected
# Example usage:
inputs = {
    "user_query": "what is the time now in india ?",
    "file_paths": ["C:\\Users\\sivan_7\\OneDrive\\Desktop\\rag\\Agentic-RAG\\langgraphflow\\MAHESH - CV.pdf"]
    #"file_paths": [""]
}

if __name__ == "__main__":
    rag_graph = build_rag_graph()
    final_state = rag_graph.invoke(inputs)
    #print("Final state:", final_state)
    #here we need to print llm output
    print("Final answer:", final_state.get("final_answer"))
    print("Source of answer:", final_state.get("source"))
    

