from config import SAMPLE_DOCS_DIR

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.retrieval.vector_store import VectorStore

parser = PDFParser(SAMPLE_DOCS_DIR)
documents = parser.load_documents()

chunker = DocumentChunker()
chunks = chunker.chunk_documents(documents)

print(f"Chunks: {len(chunks)}")

vector_db = VectorStore()
vector_db.build(chunks)

retriever = vector_db.as_retriever()

results = retriever.invoke(
    "What is polymorphism?"
)

print("\nRetrieved Documents:\n")

for idx, doc in enumerate(results, start=1):
    print("=" * 80)
    print(f"Result {idx}")
    print(doc.metadata["filename"])
    print(f"Page: {doc.metadata['page']}")
    print(doc.page_content[:300])