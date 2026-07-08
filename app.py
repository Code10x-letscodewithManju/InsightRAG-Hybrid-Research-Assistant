from config import SAMPLE_DOCS_DIR

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.retrieval.retriever import HybridRetriever


parser = PDFParser(SAMPLE_DOCS_DIR)
documents = parser.load_documents()

chunker = DocumentChunker()
chunks = chunker.chunk_documents(documents)

retriever = HybridRetriever()
retriever.build(chunks)

results = retriever.retrieve(
    "What is polymorphism?"
)

print(f"\nRetrieved {len(results)} Documents\n")

for i, doc in enumerate(results, start=1):
    print("=" * 80)
    print(f"Result {i}")
    print(doc.metadata["filename"])
    print(f"Page : {doc.metadata['page']}")
    print(doc.page_content[:300])