from config import SAMPLE_DOCS_DIR
from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker

parser = PDFParser(SAMPLE_DOCS_DIR)
documents = parser.load_documents()

chunker = DocumentChunker()
chunks = chunker.chunk_documents(documents)

print(f"Pages Loaded : {len(documents)}")
print(f"Chunks Created : {len(chunks)}")

print("\nMetadata:\n")
print(chunks[0].metadata)

print("\nChunk Preview:\n")
print(chunks[0].page_content[:500])