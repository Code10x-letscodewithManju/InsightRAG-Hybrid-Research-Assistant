"""
PDF document loader for InsightRAG.

This module is responsible for loading PDF documents and
returning LangChain Document objects with metadata.
"""

from pathlib import Path
from typing import List
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

logger = logging.getLogger(__name__)


class PDFParser:
    """Loads PDF documents from a directory."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def load_documents(self) -> List[Document]:
        """
        Load all PDF documents from the directory.

        Returns:
            List[Document]
        """

        documents = []

        if not self.directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {self.directory}"
            )

        pdf_files = sorted(self.directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning("No PDF files found.")

        for pdf in pdf_files:

            try:

                loader = PyMuPDFLoader(str(pdf))
                pages = loader.load()

                for page in pages:
                    page.metadata["filename"] = pdf.name

                documents.extend(pages)

                logger.info(
                    "Loaded %s (%d pages)",
                    pdf.name,
                    len(pages),
                )

            except Exception as e:

                logger.exception(
                    "Failed to load %s : %s",
                    pdf.name,
                    e,
                )

        logger.info(
            "Total pages loaded: %d",
            len(documents),
        )

        return documents