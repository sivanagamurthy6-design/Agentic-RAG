import os
from dotenv.main import logger
from langchain_core import messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage
#rom langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch

from state import GraphState
load_dotenv()
# Load GROQ API key from environment variable
TOP_K = 5
#print("Initializing GROQ LLM with model 'llama-3.1-8b-instant'...")
def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return key
groq_api_key = get_groq_api_key()
#LLM node with state we need to invoke the LLM with the prompt and get the response and then return the response to the user.

def llm_invoke(state: GraphState) -> str:
    user_query = state.get("user_query", "What is AI?")
    file_paths = state.get("file_paths")
    print(f"file psth in llm invoke : {file_paths}------------------")  # Debug: print file paths received in LLM node
    if file_paths:  
        # set doc_flag to true if documents are present in the state
        state["doc_present"] = True
    else:
        state["doc_present"] = False
    doc_present_flag = state.get("doc_present")
    print(f"Documents present flag in LLM node: {doc_present_flag} for user query: '{user_query}'")
    retrieved_docs = state.get("retrieved_docs", [])
    print(f"LLM Invoke Node - User query: '{user_query}', Document present: {doc_present_flag}, retrieved doc true or false: {len(retrieved_docs)}")
    #rint(f"Invoking LLM with query: {user_query}")
    _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_api_key)
    print(f"LLM initialized. Document present flag: {doc_present_flag}. Retrieved {len(retrieved_docs)} documents.")
    if doc_present_flag and len(retrieved_docs) > 0:
        print(f"Documents are present. Invoking LLM with retrieved documents for query: '{user_query}'")
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print(f"Context for LLM:\n{context[:500]}...")  # Print the first 500 characters of the context for debugging
        messages = [
            SystemMessage(content="You are a helpful assistant. Use the following retrieved documents to answer the question."),
            HumanMessage(content=f"Question: {user_query}\n\nContext:\n{context}\n\nAnswer:")
        ]

        response = _llm.invoke(messages)
        #print(f"LLM response: {response}")
        return {
        "final_answer": response.content,
        "source": "hybrid_search"
            }
    else:
       #print("No documents present. Invoking LLM without retrieved documents.")
        messages = [
            SystemMessage(content="You are a helpful assistant. Answer the following question to the best of your ability."),
            HumanMessage(content=f"Question: {user_query}\n\nAnswer:")
        ]
        response = _llm.invoke(messages)
        #rint(f"LLM response: {response}")
        
        return {
        "final_answer": response.content,
        "source": "llm"
            }
#------------------------------------------------------------------
#load the documents
def _unwrap(docs: list) -> list[Document]:
    """
    Some loaders return (Document, score) tuples instead of plain Documents.
    This unwraps them safely.
    """
    clean = []
    for item in docs:
        if isinstance(item, tuple):
            item = item[0]          # unwrap (Document, anything) → Document
        if isinstance(item, Document):
            clean.append(item)
    return clean
#-----------------------------------------------------------------------------
#node -1 - load documents and split them into chunks
def load_documents(state: GraphState):
    #file_paths = state.get("file_paths", ["C:\\Users\\sivan_7\\OneDrive\\Desktop\\rag\\Agentic-RAG\\langgraphflow\\MAHESH - CV.pdf"])
    print("STATE RECEIVED:", state)
    file_paths = state.get("file_paths")
    print(f"Loading documents from paths:-0000000000000--------------------------------- {file_paths}")
    if not file_paths:
        print("No file paths provided. Skipping document loading.")
        return {
        "documents": [],
        "doc_present": False
            }
    all_docs: list[Document] = []
    #rint(f"*******************Loading documents from: {file_paths}")
    for path_str in file_paths:
        path = Path(path_str)
        #print(f"Processing file: {path}")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path_str}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            raw = PyPDFLoader(str(path)).load()
            #print(f"-------------------{raw}--------------------")  # Debug: print raw loaded content
        elif ext == ".docx":
            raw = Docx2txtLoader(str(path)).load()
        elif ext in (".txt", ".text", ".md", ".markdown"):
            raw = TextLoader(str(path), encoding="utf-8").load()
        elif ext == ".csv":
            raw = CSVLoader(str(path)).load()
        else:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                "Supported: .pdf, .docx, .txt, .csv, .md"
            )
    #print(f"Loaded {len(raw)} raw documents from {path.name}-----{raw[0]}...")
#load_documents(["Agentic-RAG/requirements.txt"])

        raw = _unwrap(raw)   # ← unwrap tuples right after loading
        #print(f"Loaded {len(raw)} raw documents from {path.name}")
        for doc in raw:
            doc.metadata.setdefault("source", str(path))

        all_docs.extend(raw)
        #print(f"Total documents loaded so far: {len(all_docs)}")
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 300
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
     )
    chunked = _unwrap(splitter.split_documents(all_docs))  # ← unwrap after splitting too
    #print(f"Split into {len(chunked)} chunks (chunk size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    present = bool(chunked)
    return {
    "documents": chunked,
    "doc_present": bool(chunked)
            }
#here we need to build embeddings for the documents and then store them in the vector database. 
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
            #logger.warning("[hybrid_search] Skipping unexpected item type: %s", type(item))
            #print(f"Skipping unexpected item type: {type(item)}")
            pass
    return clean

#docs=load_documents(["C:\\Users\\sivan_7\\OneDrive\\Desktop\\rag\\Agentic-RAG\\langgraphflow\\MAHESH - CV.pdf"])
VECTOR_DB_PATH = "vector_DB"
def build_embeddings(state: GraphState) -> dict:
    docs = state["documents"]
    if not docs:
       #print("No documents to build embeddings for.")
        return {
        "vector_retriever": None,
        "bm25_retriever": None
            }
    embeddings   = HuggingFaceEmbeddings(model="BAAI/bge-base-en-v1.5")
    vector_store     = FAISS.from_documents(docs, embeddings)
    
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_DB_PATH)
    print(f"FAISS index saved to: {VECTOR_DB_PATH}")
    vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )
    # bm25_retriever = BM25Retriever.from_documents(docs)
    # bm25_retriever.k = TOP_K
    return {
        "vector_retriever": vector_retriever
        #"bm25_retriever": bm25_retriever
            }
#vector_retriever, bm25_retriever = build_embeddings(docs)
#---------------------------------
# here we need to do semantic search and BM25 search and then merge the results and return the final answer to the user.
#user_query = "What is AI"

BM25_WEIGHT = 0.25
VECTOR_WEIGHT = 0.75
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
# Node -2 - hybrid search node that combines BM25 and vector search results
# This node will take the question as input, perform both BM25 and vector search, and then merge the results based on the specified weights.

#def hybrid_search_node(state: GraphState, bm25_weight=BM25_WEIGHT, vector_weight=VECTOR_WEIGHT, QUESTION=user_query, top_k=TOP_K):
def hybrid_search_node(state: GraphState) -> list[Document]:
    embeddings   = HuggingFaceEmbeddings(model="BAAI/bge-base-en-v1.5")
    user_query = state.get("user_query")


    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    vector_results = vector_retriever.invoke(user_query)

    bm25_retriever = state.get("bm25_retriever")
    bm25_results = bm25_retriever.invoke(user_query) if bm25_retriever else []

    docs = vector_results + bm25_results
    combined_results = _ensure_documents(docs)
    print(f"Vector search returned {len(vector_results)} results, BM25 search returned {len(bm25_results)} results, combined unique results: {len(combined_results)}")


    #------------------------MERGE THE RESULTS------------------------------------------
    # vector_retriver = state.get("vector_retriever", None)
    # print(f"Performing hybrid search for query: {user_query}11111{vector_retriver}------------------")
    # bm25_retriver = state.get("bm25_retriever", None)
    # print(f"Performing hybrid search for query: {user_query}22222{bm25_retriver}------------------")
    # top_k = state.get("top_k", 5)
    # # This is a placeholder for your merging logic.
    # # You could, for example, combine the results based on relevance scores,
    # # or simply concatenate them and remove duplicates.
    # # vector_retriver=vector_retriver.invoke(user_query)
    # vector_results = vector_retriver.invoke(user_query) if vector_retriver else []
    # bm25_results = bm25_retriver.invoke(user_query) if bm25_retriver else []
    # print(f"Vector search returned {len(vector_results)} results, BM25 search returned {len(bm25_results)} results.")

    #  #print(f"Vector search returned {len(vector_results)} results.")
    # # vector_results = _ensure_documents(vector_retriver)
    # # bm25_retriver=bm25_retriver.invoke(user_query)
    # # bm25_results = _ensure_documents(bm25_retriver)
    # #COMBINE THE RESULTS with conditions
    # combined_results = vector_results + bm25_results
    # # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for result in combined_results:
        key = safe_content(result)
        if key not in seen:
            seen.add(key)
            unique_results.append(result)
    #rint(f"usere_query: {user_query}in hybrid_searech_node ")
    return {
    "retrieved_docs": unique_results[:TOP_K]
        }

#now we need to build tavily node
def tavily_node(state: GraphState) -> dict:
    """
    Calls the Tavily Search API to fetch real-time web results.
    Synthesises the results into a coherent answer using the LLM.
    """
    query = state.get("user_query", "")
    #load the tavily API key from environment variable
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # ── Tavily API call ───────────────────────────────────────────────────────
    tool    = TavilySearch(
        tavily_api_key=TAVILY_API_KEY,
    )
    
    tavily_result = tool.invoke({"query": query})   # returns list[dict]

    # ── Synthesise with LLM ───────────────────────────────────────────────────
    # context = "\n\n".join(
    #     f"[{i+1}] {r.get('url','')}\n{r.get('content','')}"
    #     for i, r in enumerate(results)
    # )
    #get the user query from the state and then create a prompt for the LLM to synthesise the results and then return the final answer to the user.
    #print(f"context--------{tavily_result['results'][0]['content']}---------aadd-") 
    context = tavily_result['results'][0].get("content", "") if tavily_result['results'] else ""
    
    # query = state.get("user_query", "")

    # messages = [
    #         SystemMessage(content="You are a helpful assistant. Use the following retrieved documents to answer the question."),
    #         HumanMessage(content=f"Question: {query}\n\nContext:\n{context}\n\nAnswer:")
    #     ]



    # synth_prompt = (
    #     f"Using ONLY the following web search results, answer this query:\n\n"
    #     f"Query: {query}\n\n"
    #     f"Search Results:\n{context}\n\n"
    #     "Provide a concise, well-structured answer. "
    #     "Cite sources by their [number] where relevant."
    # )
    # _llm = ChatGroq(
    #         model="llama-3.1-8b-instant",
    #         groq_api_key=groq_api_key)
    # #rint(f"Invoking LLM to synthesise Tavily results for query: {query}")
    
    # response = _llm.invoke(messages)
    # #print(f"Tavily search results for query '{query}': {context}")
    return {
        #"tavily_results": results,
        #"final_answer":   response.content,
            "final_answer": context,
            "source":         "tavily_search",
    }
