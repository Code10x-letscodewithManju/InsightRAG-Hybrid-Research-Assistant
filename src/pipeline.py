"""
End-to-end RAG pipeline for InsightRAG.
"""

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker

from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import Reranker

from src.llm.generator import AnswerGenerator
from src.llm.citation import CitationGenerator
from src.llm.confidence import ConfidenceCalculator

from src.models.schemas import (
    AnswerResponse,
    RetrievalStats,
)


class InsightRAG:
    """
    End-to-end Hybrid RAG pipeline.
    """

    def __init__(self, document_path):

        self.parser = PDFParser(document_path)
        self.chunker = DocumentChunker()

        self.retriever = HybridRetriever()
        self.reranker = Reranker()

        self.generator = AnswerGenerator()

        self.ready = False

    def build_index(self):
        """
        Parse documents, chunk them and build the retrieval index.
        """

        documents = self.parser.load_documents()

        chunks = self.chunker.chunk_documents(documents)

        self.retriever.build(chunks)

        self.ready = True

    def ask(self, question: str) -> AnswerResponse:
        """
        Execute the complete RAG pipeline.
        """

        if not self.ready:
            raise RuntimeError(
                "Call build_index() before asking questions."
            )

        # Step 1 - Retrieve candidate documents
        retrieved = self.retriever.retrieve(question)

        # Step 2 - Rerank retrieved documents
        reranked = self.reranker.rerank(
            question,
            retrieved,
        )

        # Step 3 - Generate grounded answer
        answer = self.generator.generate(
            question,
            reranked,
        )

        # Step 4 - Generate citations
        citations = CitationGenerator.generate(
            reranked
        )

        # Step 5 - Calculate confidence
        confidence = ConfidenceCalculator.calculate(
            retrieved=len(retrieved),
            reranked=len(reranked),
            citations=len(citations),
        )

        # Step 6 - Return structured response
        return AnswerResponse(
            answer=answer,
            confidence=confidence,
            citations=citations,
            retrieval=RetrievalStats(
                retrieved_documents=len(retrieved),
                reranked_documents=len(reranked),
            ),
        )