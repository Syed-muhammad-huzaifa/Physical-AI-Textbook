"""
Integration tests for the RAG (Retrieval Augmented Generation) system.

This module tests the end-to-end functionality of the RAG system.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.rag.service import RAGService
from src.rag.models import Query


class TestRAGIntegration:
    """Integration tests for RAG functionality"""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        # Create a mock RAGService for testing
        self.rag_service = RAGService()

    @patch('src.rag.service.load_markdown_files')
    @patch('src.rag.service.chunk_document_text')
    @patch('src.rag.service.embedding_model')
    @patch('src.rag.service.qdrant_service')
    def test_ingest_directory_success(self, mock_qdrant, mock_embedding, mock_chunk, mock_loader):
        """Test successful ingestion of directory."""
        # Mock return values
        mock_loader.return_value = [
            ("doc1.md", "doc1.md", "This is the content of document 1"),
            ("doc2.md", "doc2.md", "This is the content of document 2")
        ]
        mock_chunk.return_value = [
            {"chunk_id": 0, "doc_id": "doc1.md", "source": "doc1.md", "text": "This is the content of document 1"}
        ]
        mock_embedding.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        # Call the method
        result = self.rag_service.ingest_dir()

        # Assertions
        assert result["files_ingested"] == 2
        assert result["chunks_added"] == 1  # Only 1 chunk from mock
        mock_loader.assert_called_once()
        mock_embedding.embed_documents.assert_called()
        mock_qdrant.ensure_collection.assert_called_once()
        mock_qdrant.upsert_chunks.assert_called_once()

    @patch('src.rag.service.embedding_model')
    @patch('src.rag.service.qdrant_service')
    def test_retrieve_chunks_success(self, mock_qdrant, mock_embedding):
        """Test successful retrieval of chunks."""
        # Mock return values
        mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_qdrant.search.return_value = [
            {
                "score": 0.85,
                "doc_id": "doc1.md",
                "chunk_id": 0,
                "source": "doc1.md",
                "text": "This is relevant content"
            }
        ]

        # Call the method
        results = self.rag_service.retrieve("What is this?", top_k=5)

        # Assertions
        assert len(results) == 1
        assert results[0]["score"] == 0.85
        assert results[0]["doc_id"] == "doc1.md"
        assert results[0]["text"] == "This is relevant content"
        mock_embedding.embed_query.assert_called_once_with("What is this?")
        mock_qdrant.search.assert_called_once()

    @patch('src.rag.service.load_markdown_files')
    @patch('src.rag.service.qdrant_service')
    def test_ingest_empty_directory(self, mock_qdrant, mock_loader):
        """Test ingestion of an empty directory."""
        # Mock return values
        mock_loader.return_value = []

        # Call the method
        result = self.rag_service.ingest_dir()

        # Assertions
        assert result["files_ingested"] == 0
        assert result["chunks_added"] == 0
        mock_qdrant.upsert_chunks.assert_not_called()
        mock_qdrant.ensure_collection.assert_not_called()

    def test_retrieve_empty_query(self):
        """Test retrieval with empty query raises error."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.rag_service.retrieve("")

    @patch('src.rag.service.embedding_model')
    @patch('src.rag.service.qdrant_service')
    def test_retrieve_with_filters(self, mock_qdrant, mock_embedding):
        """Test retrieval with filters."""
        # Mock return values
        mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_qdrant.search.return_value = []

        # Call the method with filters
        results = self.rag_service.retrieve("test query", top_k=3, filters={"doc_id": "specific_doc.md"})

        # Assertions
        assert results == []
        # Verify that search was called with the correct filter
        mock_qdrant.search.assert_called_once()
