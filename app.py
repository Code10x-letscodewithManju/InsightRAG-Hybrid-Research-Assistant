from config import SAMPLE_DOCS_DIR

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker

from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import Reranker

from src.llm.generator import AnswerGenerator


QUESTION = "What is polymorphism?"


parser = PDFParser(SAMPLE_DOCS_DIR)
documents = parser.load_documents()

chunker = DocumentChunker()
chunks = chunker.chunk_documents(documents)

retriever = HybridRetriever()
retriever.build(chunks)

retrieved_docs = retriever.retrieve(QUESTION)

reranker = Reranker()

top_docs = reranker.rerank(
    QUESTION,
    retrieved_docs,
)

generator = AnswerGenerator()

answer = generator.generate(
    QUESTION,
    top_docs,
)

print("\n")
print("=" * 100)
print("FINAL ANSWER")
print("=" * 100)
print(answer)