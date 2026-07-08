"""
FAISS vector store for InsightRAG.
"""

from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.retrieval.embedder import Embedder


class VectorStore:
    """Creates and manages the FAISS vector database."""

    def __init__(self):
        self.embeddings = Embedder().get_embeddings()
        self.vector_store = None

    def build(self, documents: List[Document]) -> None:
        """Build FAISS vector store."""

        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings,
        )

    def get_vector_store(self):
        return self.vector_store

    def as_retriever(self, k: int = 8):
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )