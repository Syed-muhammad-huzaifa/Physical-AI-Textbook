"""
Configuration module for RAG (Retrieval Augmented Generation) system.

This module handles environment variables and configuration settings for the RAG system.
"""
import os
import logging
from typing import Optional


class RAGConfig:
    """Configuration class for RAG system"""

    # Qdrant configuration
    # Default to local Qdrant for smoother dev startup
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "rag_chunks")

    # Embedding configuration
    # Use a supported fastembed model by default
    FASTEMBED_MODEL: str = os.getenv(
        "FASTEMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    # Chunking configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Directory configuration
    DOCS_DIR: str = os.getenv("DOCS_DIR", "book-source/docs")

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration values are present"""
        if not cls.QDRANT_URL:
            raise ValueError("QDRANT_URL environment variable is required")

        if not cls.QDRANT_COLLECTION:
            raise ValueError("QDRANT_COLLECTION environment variable is required")

        if not cls.FASTEMBED_MODEL:
            raise ValueError("FASTEMBED_MODEL environment variable is required")

        if cls.EMBEDDING_BATCH_SIZE <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be a positive integer")

        if cls.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be a positive integer")

        if cls.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP must be a non-negative integer")

        if not cls.DOCS_DIR:
            raise ValueError("DOCS_DIR environment variable is required")

    @classmethod
    def log_config(cls, logger: logging.Logger) -> None:
        """Log configuration values (excluding sensitive ones)"""
        logger.info("RAG Configuration:")
        logger.info(f"  QDRANT_COLLECTION: {cls.QDRANT_COLLECTION}")
        logger.info(f"  FASTEMBED_MODEL: {cls.FASTEMBED_MODEL}")
        logger.info(f"  EMBEDDING_BATCH_SIZE: {cls.EMBEDDING_BATCH_SIZE}")
        logger.info(f"  CHUNK_SIZE: {cls.CHUNK_SIZE}")
        logger.info(f"  CHUNK_OVERLAP: {cls.CHUNK_OVERLAP}")
        logger.info(f"  DOCS_DIR: {cls.DOCS_DIR}")
        logger.info(f"  QDRANT_URL set: {'Yes' if cls.QDRANT_URL else 'No'}")
        logger.info(f"  QDRANT_API_KEY set: {'Yes' if cls.QDRANT_API_KEY else 'No'}")


# Initialize configuration
rag_config = RAGConfig()

# Validate configuration at startup
rag_config.validate()
