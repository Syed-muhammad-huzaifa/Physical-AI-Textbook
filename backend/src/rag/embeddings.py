"""
Embeddings wrapper for the RAG (Retrieval Augmented Generation) system.

This module provides functionality to generate embeddings using FastEmbed.
"""
from typing import List
import logging
from fastembed import TextEmbedding
from .config import rag_config


class EmbeddingModel:
    """
    Wrapper class for FastEmbed model
    """
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = rag_config.FASTEMBED_MODEL
        self.model = TextEmbedding(model_name=model_name)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized embedding model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            self.logger.warning("Empty text list provided for embedding")
            return []

        self.logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = []
        for i, embedding in enumerate(self.model.embed(texts)):
            embeddings.append(embedding.tolist())
            if (i + 1) % 10 == 0:  # Log progress every 10 embeddings
                self.logger.debug(f"Processed {i + 1}/{len(texts)} embeddings")
        self.logger.info(f"Completed embedding generation for {len(texts)} documents")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        if not query:
            self.logger.warning("Empty query provided for embedding")
            raise ValueError("Query cannot be empty")

        self.logger.debug(f"Generating embedding for query: {query[:50]}...")
        # Convert generator to list and get first item
        embedding = next(self.model.embed([query]))
        result = embedding.tolist()
        self.logger.debug(f"Generated embedding of length {len(result)} for query")
        return result


# Global instance for reuse
embedding_model = EmbeddingModel()