from config import SAMPLE_DOCS_DIR

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker

from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import Reranker


parser = PDFParser(SAMPLE_DOCS_DIR)
documents = parser.load_documents()

chunker = DocumentChunker()
chunks = chunker.chunk_documents(documents)

retriever = HybridRetriever()
retriever.build(chunks)

results = retriever.retrieve(
    "What is polymorphism?"
)

print(f"\nRetrieved : {len(results)}")

reranker = Reranker()

top_docs = reranker.rerank(
    "What is polymorphism?",
    results,
)

print(f"After Reranking : {len(top_docs)}")

for i, doc in enumerate(top_docs, 1):

    print("=" * 80)
    print(f"Rank {i}")

    print(doc.metadata["filename"])

    print(doc.metadata["page"])

    print(doc.page_content[:350])