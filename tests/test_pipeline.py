from config import SAMPLE_DOCS_DIR

from src.pipeline import InsightRAG


def test_pipeline():

    rag = InsightRAG(SAMPLE_DOCS_DIR)

    rag.build_index()

    response = rag.ask(
        "Explain polymorphism."
    )

    assert response.answer

    assert response.citations

    print(response.answer)

    print(response.confidence)


if __name__ == "__main__":
    test_pipeline()