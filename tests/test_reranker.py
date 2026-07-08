from config import SAMPLE_DOCS_DIR

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker

from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import Reranker


def test_reranker():

    parser = PDFParser(SAMPLE_DOCS_DIR)

    docs = parser.load_documents()

    chunks = DocumentChunker().chunk_documents(docs)

    retriever = HybridRetriever()

    retriever.build(chunks)

    retrieved = retriever.retrieve("What is polymorphism?")

    reranker = Reranker()

    reranked = reranker.rerank(
        "What is polymorphism?",
        retrieved,
    )

    assert len(reranked) > 0

    print(f"Top {len(reranked)} reranked documents.")


if __name__ == "__main__":
    test_reranker()