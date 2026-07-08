from config import SAMPLE_DOCS_DIR
from src.ingestion.parser import PDFParser


def test_pdf_parser():

    parser = PDFParser(SAMPLE_DOCS_DIR)

    documents = parser.load_documents()

    assert len(documents) > 0

    print(f"Loaded {len(documents)} pages successfully.")


if __name__ == "__main__":
    test_pdf_parser()