"""
document_loader.py
──────────────────
Loads files into LangChain Document objects.
Supports: PDF, DOCX, TXT, CSV, Markdown.
"""

from __future__ import annotations
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


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


def load_documents(file_paths: list[str]) -> list[Document]:
    all_docs: list[Document] = []

    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path_str}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            raw = PyPDFLoader(str(path)).load()
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

        raw = _unwrap(raw)   # ← unwrap tuples right after loading

        for doc in raw:
            doc.metadata.setdefault("source", str(path))

        all_docs.extend(raw)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked = _unwrap(splitter.split_documents(all_docs))  # ← unwrap after splitting too
    return chunked