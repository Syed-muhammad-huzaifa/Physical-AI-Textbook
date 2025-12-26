"""
API endpoints for the RAG (Retrieval Augmented Generation) system.

This module defines the FastAPI endpoints for health check, ingestion, and retrieval.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from ..rag.service import rag_service
from ..rag.schemas import (
    HealthResponse,
    IngestDirRequest,
    IngestDirResponse,
    RetrieveRequest,
    RetrieveResponse
)
from ..rag.qdrant import qdrant_service
from ..rag.config import rag_config


# Create router
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify system connectivity.
    """
    try:
        # Test Qdrant connectivity by trying to get collection info
        qdrant_service.client.get_collection(rag_config.QDRANT_COLLECTION)
        return HealthResponse(status="ok")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant connection failed: {str(e)}")


@router.post("/ingest-dir", response_model=IngestDirResponse)
async def ingest_directory(
    request: IngestDirRequest,
):
    """
    Ingest all markdown files from the configured directory.
    """
    try:
        result = rag_service.ingest_dir(wipe_collection=request.wipe_collection)
        return IngestDirResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_chunks(
    request: RetrieveRequest,
):
    """
    Retrieve relevant chunks for a given query.
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="Query cannot be empty")

        if request.top_k < 1 or request.top_k > 100:
            raise HTTPException(status_code=422, detail="top_k must be between 1 and 100")

        results = rag_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )

        # Convert results to the expected schema format
        formatted_results = [
            {
                "score": result["score"],
                "doc_id": result["doc_id"],
                "chunk_id": result["chunk_id"],
                "source": result["source"],
                "text": result["text"]
            }
            for result in results
        ]

        return RetrieveResponse(results=formatted_results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
