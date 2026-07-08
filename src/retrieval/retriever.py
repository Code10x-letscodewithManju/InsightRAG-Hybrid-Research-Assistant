"""
Hybrid retriever for InsightRAG.
"""

from typing import List

from langchain_core.documents import Document
from langchain_classic.retrievers import (
    EnsembleRetriever,
    BM25Retriever,
)

from config import (
    VECTOR_TOP_K,
    BM25_TOP_K,
    ENSEMBLE_WEIGHTS,
)
from src.retrieval.vector_store import VectorStore


class HybridRetriever:
    """Hybrid semantic + lexical retriever."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = None

    def build(self, documents: List[Document]) -> None:
        """Build the hybrid retriever."""

        self.vector_store.build(documents)

        vector_retriever = self.vector_store.as_retriever(
            k=VECTOR_TOP_K
        )

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = BM25_TOP_K

        self.retriever = EnsembleRetriever(
            retrievers=[
                vector_retriever,
                bm25_retriever,
            ],
            weights=ENSEMBLE_WEIGHTS,
        )

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents."""

        return self.retriever.invoke(query)