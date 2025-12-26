"""
Integration tests for the health check functionality.

This module tests the health check endpoint in an integrated environment.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import create_app


class TestHealthIntegration:
    """Integration tests for health check functionality"""

    def setup_method(self):
        """Setup test client before each test method."""
        app = create_app()
        self.client = TestClient(app)

    @patch('src.rag.qdrant.qdrant_service')
    def test_health_check_success(self, mock_qdrant_service):
        """Test health check endpoint returns success when Qdrant is available."""
        # Mock successful Qdrant connection
        mock_qdrant_service.client.get_collection.return_value = {"status": "ok"}

        # Make request to health endpoint
        response = self.client.get("/api/v1/rag/health")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @patch('src.rag.qdrant.qdrant_service')
    def test_health_check_failure(self, mock_qdrant_service):
        """Test health check endpoint returns failure when Qdrant is unavailable."""
        # Mock failed Qdrant connection
        mock_qdrant_service.client.get_collection.side_effect = Exception("Connection failed")

        # Make request to health endpoint
        response = self.client.get("/api/v1/rag/health")

        # Assertions
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data

    @patch('src.rag.qdrant.qdrant_service')
    def test_health_check_no_auth_required(self, mock_qdrant_service):
        """Test health check endpoint does not require authentication."""
        # Mock successful Qdrant connection
        mock_qdrant_service.client.get_collection.return_value = {"status": "ok"}

        # Make request to health endpoint without any auth header
        response = self.client.get("/api/v1/rag/health")

        # Should return 200 for public access
        assert response.status_code == 200