import streamlit as st
import tempfile
import os

from document_ingestion import build_ingestion_graph
from retrive_answer import build_query_graph


st.set_page_config(page_title="Agentic RAG", layout="wide")

st.title("Chat Assistant ")

# Build graphs
ingestion_graph = build_ingestion_graph()
query_graph = build_query_graph()


# -----------------------
# Session State
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_present" not in st.session_state:
    st.session_state.doc_present = False

if "file_paths" not in st.session_state:
    st.session_state.file_paths = None


# -----------------------
# Document Upload
# -----------------------

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf","txt","docx"]
)


if uploaded_file and not st.session_state.doc_present:

    temp_dir = tempfile.mkdtemp()
    file_paths = os.path.join(temp_dir, uploaded_file.name)

    with open(file_paths,"wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.file_paths = file_paths

    st.success("Document uploaded")

    with st.spinner("Processing document..."):

        ingestion_graph.invoke({
            "file_paths":[file_paths]
        })

    st.session_state.doc_present = True

    st.success("Document indexed successfully")


# -----------------------
# Display Chat History
# -----------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -----------------------
# Chat Input
# -----------------------

user_prompt = st.chat_input("Ask a question")

if user_prompt:

    # Show user message
    st.session_state.messages.append({
        "role":"user",
        "content":user_prompt
    })

    with st.chat_message("user"):
        st.write(user_prompt)


    # Run graph
    with st.spinner("Thinking..."):

        result = query_graph.invoke({
            "user_query":user_prompt,
            "doc_present":st.session_state.doc_present,
            "file_paths": st.session_state.file_paths
        })

        answer = result.get("final_answer","No answer found")
        source = result.get("source","unknown")


    # Clean Tavily Markdown
    answer = answer.replace("#","")

    assistant_reply = f"{answer}\n\n(Source: {source})"


    # Store response
    st.session_state.messages.append({
        "role":"assistant",
        "content":assistant_reply
    })


    # Display assistant message
    with st.chat_message("assistant"):
        st.write(assistant_reply)