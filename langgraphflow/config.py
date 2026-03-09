"""
config.py
─────────
Central place for every tunable constant and environment variable.
Edit this file (or use a .env file) to configure the pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env if present

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY","")
LLM_MODEL=os.getenv("llama-3.1-8b-instant")
# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# ── Tavily ────────────────────────────────────────────────────────────────────
TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K            = int(os.getenv("TOP_K", "5"))          # number of docs to retrieve
BM25_WEIGHT      = float(os.getenv("BM25_WEIGHT", "0.4"))  # weight for BM25 vs vector
VECTOR_WEIGHT    = float(os.getenv("VECTOR_WEIGHT", "0.6"))

# ── Retry loop ────────────────────────────────────────────────────────────────
MAX_REWRITE_RETRIES = int(os.getenv("MAX_REWRITE_RETRIES", "2"))

# ── Relevance threshold ───────────────────────────────────────────────────────
# Validator scores context 0-1; below this threshold → "not relevant"
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.45"))