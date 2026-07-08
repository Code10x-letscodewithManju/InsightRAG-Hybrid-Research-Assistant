"""
Global configuration for InsightRAG.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

# ==================================================
# Project Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
INDEX_DIR = DATA_DIR / "indexes"
OUTPUT_DIR = DATA_DIR / "outputs"

PROMPTS_DIR = BASE_DIR / "prompts"

# ==================================================
# LLM Configuration
# ==================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "llama-3.3-70b-versatile"

# ==================================================
# Embedding Model
# ==================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ==================================================
# Reranker
# ==================================================

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ==================================================
# Chunking
# ==================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ==================================================
# Retrieval
# ==================================================

VECTOR_TOP_K = 8
BM25_TOP_K = 8
FINAL_TOP_K = 4


# ==================================================
# Hybrid Retrieval
# ==================================================

ENSEMBLE_WEIGHTS = [0.7, 0.3]