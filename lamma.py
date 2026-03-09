import streamlit as st
import os
import time
import fitz   #pymupdf
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor  #added

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import BM25Retriever  # added
from dotenv import load_dotenv

load_dotenv()

## load the GROQ, HF and Tavily API KEYs
groq_api_key = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
tavily_api_key = os.getenv("TAVILY_API_KEY")

st.title("Chatgroq With Llama3 Demo")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
)

# Tavily web search tool (used only when needed)
web_search: TavilySearchResults | None = None
if tavily_api_key:
    web_search = TavilySearchResults(max_results=5)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

You are given some context from uploaded documents. 
Use this context when it is relevant, but if the context
does not contain the necessary information, answer the
question from your own general knowledge instead.

<context>
{context}
<context>

Question: {input}
"""
)

# File uploader for user PDFs, Word, or text files
uploaded_files = st.file_uploader(
    "Upload one or more documents (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def format_history(max_turns: int = 3) -> str:
    """
    Format the last few user/assistant messages as plain text.
    Stored as a flat list of dicts: [{"role": "user"/"assistant", "content": "..."}].
    """
    # Each "turn" is typically user + assistant, so keep up to 2 * max_turns messages
    recent_messages = st.session_state.chat_history[-2 * max_turns :]
    lines = []
    for msg in recent_messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


with st.expander("Conversation (last few turns)"):
    hist_preview = format_history()
    if hist_preview:
        st.text(hist_preview)
    else:
        st.caption("No conversation history yet.")


def vector_embedding(files):
    if "vectors" in st.session_state:
        return

    if not files:
        st.error("Please upload at least one PDF before building the vector store.")
        return

    all_docs = []
    for uploaded_file in files:
        filename = uploaded_file.name.lower()

        if filename.endswith(".pdf"):
            # Save uploaded PDF to a temporary file so PyPDFLoader can read it
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            all_docs.extend(loader.load())

            os.remove(tmp_path)

        elif filename.endswith(".docx"):
            # Save uploaded Word document to a temporary file for Docx2txtLoader
            with NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = Docx2txtLoader(tmp_path)
            all_docs.extend(loader.load())

            os.remove(tmp_path)

        elif filename.endswith(".txt"):
            # Read raw text from the uploaded .txt file
            raw_bytes = uploaded_file.read()
            text = raw_bytes.decode("utf-8", errors="ignore")
            if text.strip():
                all_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name},
                    )
                )

    if not all_docs:
        st.error("No text could be extracted from the uploaded PDFs. Please check your documents.")
        return

    st.session_state.embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_docs)

    if not chunks:
        st.error("No text chunks were created from the PDFs. Please check your documents.")
        return

    st.session_state.vectors = FAISS.from_documents(
        chunks, st.session_state.embeddings
    )


prompt1 = st.text_input("Enter Your Question From Documents", key="user_question")


def answer_with_web_search(question: str):
    """
    Use Tavily web search plus the LLM to answer questions
    when neither the PDFs nor general knowledge are sufficient.
    """
    if web_search is None:
        # Tavily not configured – fall back to plain LLM
        return llm.invoke(
            f"Answer the following question from your general knowledge: {question}"
        )

    # Run web search
    results = web_search.invoke(question)

    # TavilySearchResults usually returns a list of dicts; make a readable context string
    if isinstance(results, str):
        web_context = results
    else:
        try:
            web_context = "\n\n".join(
                r.get("content", "") for r in results if isinstance(r, dict)
            )
        except Exception:
            web_context = str(results)

    web_prompt = f"""
You are a helpful AI assistant with access to fresh web search results.

Web search results:
{web_context}

User question: {question}

Using the web results above, provide an accurate and up‑to‑date answer.
If the results are inconclusive, say that clearly.
"""

    return llm.invoke(web_prompt)

if st.button("Documents Embedding"):
    vector_embedding(uploaded_files)
    if "vectors" in st.session_state:
        st.success("Vector Store DB Is Ready")


if prompt1:
    answered = False

    # Build a question that includes short conversation history for the LLM
    history_text = format_history()
    if history_text:
        question_with_history = f"{history_text}\nUser: {prompt1}"
    else:
        question_with_history = prompt1

    if "vectors" not in st.session_state:
        if not uploaded_files:
            # No documents uploaded: first try LLM, then fall back to web search if it
            # clearly says it lacks up-to-date information.
            start = time.process_time()
            primary_response = llm.invoke(
                f"Answer the following question from your general knowledge: {question_with_history}"
            )

            text = getattr(primary_response, "content", "") or str(primary_response)
            lowered = text.lower()

            needs_web = any(
                phrase in lowered
                for phrase in [
                    "knowledge cutoff",
                    "knowledge cut-off",
                    "don't have information about events after",
                    "do not have information about events after",
                    "i don't have real-time",
                    "i do not have real-time",
                    "cannot access the internet",
                    "can't access the internet",
                    "do not have access to current information",
                ]
            )

            if needs_web:
                response = answer_with_web_search(question_with_history)
            else:
                response = primary_response

            print("Response time :", time.process_time() - start)
            st.write(response.content)
            answered = True
        else:
            # Documents uploaded but embeddings not built yet
            st.warning(
                "Please click 'Documents Embedding' first to build the vector store "
                "for your uploaded documents."
            )
    else:
        retriever = st.session_state.vectors.as_retriever()

        start = time.process_time()
        # LangChain v1 retrievers use .invoke() instead of .get_relevant_documents()
        docs = retriever.invoke(prompt1)

        if not docs:
            # Fallback: no relevant context found, answer using web search + LLM
            st.info(
                "No relevant information found in the uploaded documents. "
                "Fetching information from the web instead."
            )
            response = answer_with_web_search(question_with_history)
            answered = True
        else:
            context_text = "\n\n".join(doc.page_content for doc in docs)

            rag_chain = prompt | llm
            rag_response = rag_chain.invoke(
                {"context": context_text, "input": question_with_history}
            )

            # Check if the RAG answer itself complains about knowledge cutoff / no realtime info.
            rag_text = getattr(rag_response, "content", "") or str(rag_response)
            rag_lowered = rag_text.lower()

            needs_web_from_rag = any(
                phrase in rag_lowered
                for phrase in [
                    "knowledge cutoff",
                    "knowledge cut-off",
                    "don't have information about events after",
                    "do not have information about events after",
                    "i don't have real-time",
                    "i do not have real-time",
                    "cannot access the internet",
                    "can't access the internet",
                    "do not have access to current information",
                ]
            )

            if needs_web_from_rag:
                st.info(
                    "The documents don't contain this up‑to‑date information. "
                    "Fetching information from the web instead."
                )
                response = answer_with_web_search(question_with_history)
            else:
                response = rag_response

            answered = True

        print("Response time :", time.process_time() - start)

        # `response` is a ChatMessage
        st.write(response.content)

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(docs):
                st.write(doc.page_content)
                st.write("--------------------------------")

    # After answering, append this turn to the in-memory conversation history
    if answered:
        # Store just the current user question (without history prefix)
        st.session_state.chat_history.append({"role": "user", "content": prompt1})
        answer_text = getattr(response, "content", "") or str(response)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer_text}
        )

