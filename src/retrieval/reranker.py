"""
Cross-encoder reranker for InsightRAG.
"""

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import (
    RERANKER_MODEL,
    FINAL_TOP_K,
)


class Reranker:
    """Cross-encoder based document reranker."""

    def __init__(self):

        self.model = CrossEncoder(
            RERANKER_MODEL
        )

    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            doc
            for _, doc in ranked[:FINAL_TOP_K]
        ]