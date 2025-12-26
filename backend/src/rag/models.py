"""
Data models for the RAG (Retrieval Augmented Generation) system.

This module defines the core data models for documents, chunks, and queries.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    A segment of markdown text with associated metadata and embedding vector
    """
    doc_id: str
    chunk_id: int
    source: str
    text: str
    embedding: Optional[List[float]] = None  # Will be populated during ingestion


class Document(BaseModel):
    """
    A markdown file identified by its relative path with associated chunks
    """
    doc_id: str
    source: str
    chunks: List[DocumentChunk] = []


class Query(BaseModel):
    """
    A text input that is embedded and used to find similar chunks in the vector database
    """
    text: str
    vector: Optional[List[float]] = None  # Will be populated during search
    top_k: int = Field(default=6, ge=1, le=100)
    filters: Optional[dict] = None