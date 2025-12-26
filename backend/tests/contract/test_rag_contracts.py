"""
API contract tests for the RAG (Retrieval Augmented Generation) system.

This module tests the API contracts for the RAG endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import create_app
from src.rag.service import rag_service


class TestRAGContract:
    """Contract tests for RAG API endpoints"""

    def setup_method(self):
        """Setup test client before each test method."""
        app = create_app()
        self.client = TestClient(app)

    @patch('src.rag.service.RAGService.ingest_dir')
    def test_ingest_dir_contract_success(self, mock_ingest_dir):
        """Test the contract for the ingest-dir endpoint."""
        # Mock return value
        mock_ingest_dir.return_value = {"files_ingested": 5, "chunks_added": 50}

        # Make request
        response = self.client.post(
            "/api/v1/rag/ingest-dir",
            json={"wipe_collection": False}
        )

        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        assert "files_ingested" in response_data
        assert "chunks_added" in response_data
        assert isinstance(response_data["files_ingested"], int)
        assert isinstance(response_data["chunks_added"], int)

    @patch('src.rag.service.RAGService.ingest_dir')
    def test_ingest_dir_contract_with_wipe(self, mock_ingest_dir):
        """Test the contract for the ingest-dir endpoint with wipe_collection=True."""
        # Mock return value
        mock_ingest_dir.return_value = {"files_ingested": 0, "chunks_added": 0}

        # Make request
        response = self.client.post(
            "/api/v1/rag/ingest-dir",
            json={"wipe_collection": True}
        )

        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        assert "files_ingested" in response_data
        assert "chunks_added" in response_data

    @patch('src.rag.service.RAGService.retrieve')
    def test_retrieve_contract_success(self, mock_retrieve):
        """Test the contract for the retrieve endpoint."""
        # Mock return value
        mock_retrieve.return_value = [
            {
                "score": 0.85,
                "doc_id": "test_doc.md",
                "chunk_id": 1,
                "source": "test_doc.md",
                "text": "This is test content"
            }
        ]

        # Make request
        response = self.client.post(
            "/api/v1/rag/retrieve",
            json={
                "query": "test query",
                "top_k": 5,
                "filters": {"doc_id": "test_doc.md"}
            }
        )

        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        assert "results" in response_data
        assert isinstance(response_data["results"], list)
        if len(response_data["results"]) > 0:
            result = response_data["results"][0]
            assert "score" in result
            assert "doc_id" in result
            assert "chunk_id" in result
            assert "source" in result
            assert "text" in result
            assert isinstance(result["score"], (int, float))
            assert isinstance(result["chunk_id"], int)

    @patch('src.rag.service.RAGService.retrieve')
    def test_retrieve_contract_without_filters(self, mock_retrieve):
        """Test the contract for the retrieve endpoint without filters."""
        # Mock return value
        mock_retrieve.return_value = []

        # Make request
        response = self.client.post(
            "/api/v1/rag/retrieve",
            json={
                "query": "another test query",
                "top_k": 3
            }
        )

        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        assert "results" in response_data
        assert response_data["results"] == []

    def test_retrieve_contract_validation_error(self):
        """Test the contract for the retrieve endpoint with validation error."""
        # Make request with empty query
        response = self.client.post(
            "/api/v1/rag/retrieve",
            json={
                "query": "",  # Empty query should cause validation error
                "top_k": 5
            }
        )

        # Should return 422 for validation error
        assert response.status_code == 422

    def test_retrieve_contract_invalid_top_k(self):
        """Test the contract for the retrieve endpoint with invalid top_k."""
        # Make request with invalid top_k
        response = self.client.post(
            "/api/v1/rag/retrieve",
            json={
                "query": "test query",
                "top_k": 101  # Should be <= 100
            }
        )

        # Should return 422 for validation error
        assert response.status_code == 422

    @patch('src.rag.qdrant.qdrant_service')
    def test_health_check_contract_success(self, mock_qdrant_service):
        """Test the contract for the health check endpoint."""
        # Mock successful connection
        mock_qdrant_service.client.get_collection.return_value = True

        # Make request
        response = self.client.get("/api/v1/rag/health")

        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        assert "status" in response_data
        assert response_data["status"] == "ok"

    @patch('src.rag.qdrant.qdrant_service')
    def test_health_check_contract_failure(self, mock_qdrant_service):
        """Test the contract for the health check endpoint when Qdrant is unavailable."""
        # Mock connection failure
        mock_qdrant_service.client.get_collection.side_effect = Exception("Connection failed")

        # Make request
        response = self.client.get("/api/v1/rag/health")

        # Should return 503 for service unavailable
        assert response.status_code == 503
        response_data = response.json()
        assert "detail" in response_data