from config import SAMPLE_DOCS_DIR
from src.pipeline import InsightRAG

rag = InsightRAG(SAMPLE_DOCS_DIR)

print("Building index...")
rag.build_index()

print("Ready!")

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break

    response = rag.ask(question)

    print("\nAnswer\n")
    print(response.answer)

    print("\nConfidence")
    print(response.confidence)

    print("\nSources")

    for citation in response.citations:
        print(
            f"- {citation.filename} (Page {citation.page})"
        )