"""
FAISS vector store for InsightRAG.
"""

from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.retrieval.embedder import Embedder


class VectorStore:
    """Creates and manages the FAISS vector index."""

    def __init__(self):
        self.embeddings = Embedder().get_embeddings()
        self.vector_db = None

    def build(self, documents: List[Document]):
        """Create FAISS index."""
        self.vector_db = FAISS.from_documents(
            documents,
            self.embeddings,
        )

    def as_retriever(self, k: int = 8):
        """Return FAISS retriever."""
        return self.vector_db.as_retriever(
            search_kwargs={"k": k}
        )