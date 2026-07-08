"""
Pydantic models for InsightRAG.
"""

from typing import List

from pydantic import BaseModel


class Citation(BaseModel):
    filename: str
    page: int


class RetrievalStats(BaseModel):
    retrieved_documents: int
    reranked_documents: int
    embedding_model: str
    retriever: str
    reranker: str
    llm: str


class AnswerResponse(BaseModel):
    answer: str
    confidence: str
    citations: List[Citation]
    retrieval: RetrievalStats