"""
LLM response generation using Groq.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
)


class AnswerGenerator:
    """
    Generates grounded answers using the Groq LLM.
    """

    def __init__(self):
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        # Load prompt from prompts/rag_prompt.txt
        prompt_path = (
            Path(__file__).resolve().parents[2]
            / "prompts"
            / "rag_prompt.txt"
        )

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()

        self.prompt = ChatPromptTemplate.from_template(prompt_text)

    def build_context(self, documents: List[Document]) -> str:
        """
        Build a formatted context string from retrieved documents.
        """

        context_sections = []

        for idx, doc in enumerate(documents, start=1):
            context_sections.append(
                f"""
[Document {idx}]
Source: {doc.metadata.get("filename")}
Page: {doc.metadata.get("page")}

{doc.page_content.strip()}
"""
            )

        return ("\n" + "-" * 80 + "\n").join(context_sections)

    def generate(
        self,
        question: str,
        documents: List[Document],
    ) -> str:
        """
        Generate a grounded answer from the retrieved documents.
        """

        context = self.build_context(documents)

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return response.content.strip()