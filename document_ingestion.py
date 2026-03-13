from langgraph.graph import StateGraph, START, END
from state import GraphState
from nodes import load_documents, build_embeddings


def build_ingestion_graph():

    graph = StateGraph(GraphState)

    graph.add_node("load_and_split", load_documents)
    graph.add_node("build_embeddings", build_embeddings)

    graph.add_edge(START, "load_and_split")
    graph.add_edge("load_and_split", "build_embeddings")
    graph.add_edge("build_embeddings", END)

    return graph.compile()
# Example usage:

if __name__ == "__main__":
    inputs = {"file_paths": ["C:\\Users\\sivan_7\\OneDrive\\Desktop\\rag\\Agentic-RAG\\langgraphflow\\MAHESH - CV.pdf"]}
    ingestion_graph = build_ingestion_graph()
    final_state = ingestion_graph.invoke(inputs)
    #print("Final state:", final_state)