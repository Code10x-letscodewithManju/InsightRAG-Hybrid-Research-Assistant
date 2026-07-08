from config import SAMPLE_DOCS_DIR

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker

from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import Reranker

from src.llm.generator import AnswerGenerator


def test_generator():

    parser = PDFParser(SAMPLE_DOCS_DIR)

    docs = parser.load_documents()

    chunks = DocumentChunker().chunk_documents(docs)

    retriever = HybridRetriever()

    retriever.build(chunks)

    retrieved = retriever.retrieve("What is polymorphism?")

    reranked = Reranker().rerank(
        "What is polymorphism?",
        retrieved,
    )

    answer = AnswerGenerator().generate(
        "What is polymorphism?",
        reranked,
    )

    assert len(answer) > 0

    print(answer)


if __name__ == "__main__":
    test_generator()