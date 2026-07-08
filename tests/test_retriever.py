from config import SAMPLE_DOCS_DIR

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.retrieval.retriever import HybridRetriever


def test_retriever():

    parser = PDFParser(SAMPLE_DOCS_DIR)

    docs = parser.load_documents()

    chunks = DocumentChunker().chunk_documents(docs)

    retriever = HybridRetriever()

    retriever.build(chunks)

    results = retriever.retrieve("What is polymorphism?")

    assert len(results) > 0

    print(f"Retrieved {len(results)} documents.")


if __name__ == "__main__":
    test_retriever()