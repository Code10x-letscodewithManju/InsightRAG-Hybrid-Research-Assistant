"""
Citation utilities.
"""

from typing import List

from langchain_core.documents import Document

from src.models.schemas import Citation


class CitationGenerator:

    @staticmethod
    def generate(documents: List[Document]) -> List[Citation]:

        citations = []
        seen = set()

        for doc in documents:

            key = (
                doc.metadata["filename"],
                doc.metadata["page"],
            )

            if key not in seen:

                seen.add(key)

                citations.append(

                    Citation(
                        filename=doc.metadata["filename"],
                        page=doc.metadata["page"],
                    )

                )

        return citations