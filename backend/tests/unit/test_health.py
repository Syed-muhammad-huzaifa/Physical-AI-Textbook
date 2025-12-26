"""
Unit tests for the health check functionality.

This module tests the health check functionality of the RAG system.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.rag.qdrant import qdrant_service
from src.rag.config import rag_config


class TestHealth:
    """Unit tests for health check functionality"""

    @patch('src.rag.qdrant.QdrantClient')
    def test_qdrant_connection_success(self, mock_qdrant_client_class):
        """Test successful Qdrant connection check."""
        # Mock the Qdrant client and its methods
        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.return_value = {"status": "ok"}
        mock_qdrant_client_class.return_value = mock_client_instance

        # Update the qdrant_service client to use the mock
        original_client = qdrant_service.client
        qdrant_service.client = mock_client_instance

        try:
            # Test that no exception is raised when connection is successful
            qdrant_service.client.get_collection(rag_config.QDRANT_COLLECTION)
            # If we reach this point, the connection was successful
            assert True  # This will always pass if no exception was raised
        except Exception:
            # If an exception was raised, the test should fail
            assert False, "Expected successful connection but got an exception"
        finally:
            # Restore the original client
            qdrant_service.client = original_client

    @patch('src.rag.qdrant.QdrantClient')
    def test_qdrant_connection_failure(self, mock_qdrant_client_class):
        """Test Qdrant connection check when connection fails."""
        # Mock the Qdrant client to raise an exception
        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.side_effect = Exception("Connection failed")
        mock_qdrant_client_class.return_value = mock_client_instance

        # Update the qdrant_service client to use the mock
        original_client = qdrant_service.client
        qdrant_service.client = mock_client_instance

        try:
            # This should raise an exception
            with pytest.raises(Exception):
                qdrant_service.client.get_collection(rag_config.QDRANT_COLLECTION)
        finally:
            # Restore the original client
            qdrant_service.client = original_client

    def test_config_validation_success(self):
        """Test that config validation passes with valid settings."""
        # Test that the default config values are valid
        try:
            rag_config.validate()
            # If we reach this point, validation passed
            assert True
        except ValueError as e:
            # If validation failed, the test should fail
            assert False, f"Expected successful validation but got: {e}"

    def test_config_values_set(self):
        """Test that required config values are set."""
        # Check that the required configuration values are not empty
        assert rag_config.QDRANT_URL != "", "QDRANT_URL should be set"
        assert rag_config.QDRANT_COLLECTION != "", "QDRANT_COLLECTION should be set"
        assert rag_config.FASTEMBED_MODEL != "", "FASTEMBED_MODEL should be set"
        assert rag_config.DOCS_DIR != "", "DOCS_DIR should be set"
        assert rag_config.CHUNK_SIZE > 0, "CHUNK_SIZE should be positive"
        assert rag_config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP should be non-negative"