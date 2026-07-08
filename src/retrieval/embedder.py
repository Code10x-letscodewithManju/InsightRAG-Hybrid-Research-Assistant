"""
Embedding model loader for InsightRAG.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


class Embedder:
    """Loads the embedding model."""

    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_embeddings(self):
        """Return embedding model."""
        return self.embedding_model