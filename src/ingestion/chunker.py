"""
Document chunking module for InsightRAG.

Splits parsed documents into overlapping chunks while
preserving metadata required for retrieval and citations.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunker:
    """Splits LangChain Documents into retrieval-ready chunks."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.

        Args:
            documents: Parsed LangChain documents.

        Returns:
            List of chunked LangChain documents.
        """

        chunks = self.splitter.split_documents(documents)

        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["total_chunks"] = total_chunks

        return chunks