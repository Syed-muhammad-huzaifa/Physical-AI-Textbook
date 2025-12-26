"""
API request/response schemas for the RAG (Retrieval Augmented Generation) system.

This module defines the Pydantic schemas for API endpoints.
"""
from typing import List, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""
    status: str


class IngestDirRequest(BaseModel):
    """Request schema for ingest directory endpoint"""
    wipe_collection: bool = False


class IngestDirResponse(BaseModel):
    """Response schema for ingest directory endpoint"""
    files_ingested: int
    chunks_added: int


class RetrieveRequest(BaseModel):
    """Request schema for retrieve endpoint"""
    query: str
    top_k: int = 6
    filters: Optional[dict] = None


class RetrieveResult(BaseModel):
    """Schema for individual result in retrieve response"""
    score: float
    doc_id: str
    chunk_id: int
    source: str
    text: str


class RetrieveResponse(BaseModel):
    """Response schema for retrieve endpoint"""
    results: List[RetrieveResult]