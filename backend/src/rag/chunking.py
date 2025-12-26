"""
Text chunking utility for the RAG (Retrieval Augmented Generation) system.

This module provides functionality to split text into chunks with overlap.
"""
from typing import List, Dict, Any
from .config import rag_config


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
    """
    Split text into chunks with specified size and overlap.

    Args:
        text: The input text to chunk
        chunk_size: Size of each chunk (defaults to config value)
        overlap: Overlap between chunks (defaults to config value)

    Returns:
        List of dictionaries containing chunk_id and text
    """
    if chunk_size is None:
        chunk_size = rag_config.CHUNK_SIZE
    if overlap is None:
        overlap = rag_config.CHUNK_OVERLAP

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        # Calculate end position
        end = start + chunk_size

        # If this is the last chunk and it's smaller than chunk_size, include it
        if end > len(text):
            end = len(text)

        # Extract the chunk
        chunk_text = text[start:end]

        # Add to chunks list
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text
        })

        chunk_id += 1

        # If we've reached the end of the text, stop to avoid looping on short docs
        if end >= len(text):
            break

        # Move start position by (chunk_size - overlap) for the next chunk
        start = max(end - overlap, 0)

    return chunks


def chunk_document_text(doc_id: str, text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
    """
    Chunk document text and include document ID information.

    Args:
        doc_id: The document ID
        text: The document text to chunk
        chunk_size: Size of each chunk (defaults to config value)
        overlap: Overlap between chunks (defaults to config value)

    Returns:
        List of dictionaries containing doc_id, chunk_id, source, and text
    """
    text_chunks = chunk_text(text, chunk_size, overlap)

    # Add doc_id and source to each chunk
    for chunk in text_chunks:
        chunk["doc_id"] = doc_id
        chunk["source"] = doc_id  # source is same as doc_id

    return text_chunks
