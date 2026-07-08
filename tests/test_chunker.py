from config import SAMPLE_DOCS_DIR

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker


def test_chunker():

    parser = PDFParser(SAMPLE_DOCS_DIR)

    docs = parser.load_documents()

    chunker = DocumentChunker()

    chunks = chunker.chunk_documents(docs)

    assert len(chunks) > 0

    print(f"Created {len(chunks)} chunks.")


if __name__ == "__main__":
    test_chunker()