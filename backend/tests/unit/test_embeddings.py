"""
Unit tests for the embeddings module.

This module tests the FastEmbed wrapper functionality.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.rag.embeddings import EmbeddingModel


class TestEmbeddingModel:
    """Unit tests for EmbeddingModel class"""

    @patch('src.rag.embeddings.TextEmbedding')
    def test_embed_documents_success(self, mock_text_embedding_class):
        """Test successful embedding of multiple documents."""
        # Mock the TextEmbedding instance
        mock_model_instance = MagicMock()
        mock_model_instance.embed.return_value = iter([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_text_embedding_class.return_value = mock_model_instance

        # Create EmbeddingModel instance
        embedding_model = EmbeddingModel(model_name="test-model")

        # Call the method
        texts = ["Document 1", "Document 2"]
        result = embedding_model.embed_documents(texts)

        # Assertions
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        mock_text_embedding_class.assert_called_once_with(model_name="test-model")
        mock_model_instance.embed.assert_called_once_with(texts)

    @patch('src.rag.embeddings.TextEmbedding')
    def test_embed_query_success(self, mock_text_embedding_class):
        """Test successful embedding of a single query."""
        # Mock the TextEmbedding instance
        mock_model_instance = MagicMock()
        mock_model_instance.embed.return_value = iter([[0.7, 0.8, 0.9]])
        mock_text_embedding_class.return_value = mock_model_instance

        # Create EmbeddingModel instance
        embedding_model = EmbeddingModel(model_name="test-model")

        # Call the method
        query = "Test query"
        result = embedding_model.embed_query(query)

        # Assertions
        assert result == [0.7, 0.8, 0.9]
        mock_model_instance.embed.assert_called_once_with([query])

    @patch('src.rag.embeddings.TextEmbedding')
    def test_init_with_custom_model(self, mock_text_embedding_class):
        """Test initialization with a custom model name."""
        # Mock the TextEmbedding instance
        mock_model_instance = MagicMock()
        mock_model_instance.embed.return_value = iter([[0.1, 0.2]])
        mock_text_embedding_class.return_value = mock_model_instance

        # Create EmbeddingModel instance with custom model
        embedding_model = EmbeddingModel(model_name="custom-model")

        # Verify the model was initialized with the custom name
        mock_text_embedding_class.assert_called_once_with(model_name="custom-model")

    def test_embed_documents_empty_list(self):
        """Test embedding of an empty list of documents."""
        # Create EmbeddingModel instance
        embedding_model = EmbeddingModel(model_name="test-model")

        # Call the method with empty list
        result = embedding_model.embed_documents([])

        # Should return empty list
        assert result == []

    @patch('src.rag.embeddings.TextEmbedding')
    def test_embed_single_document(self, mock_text_embedding_class):
        """Test embedding of a single document."""
        # Mock the TextEmbedding instance
        mock_model_instance = MagicMock()
        mock_model_instance.embed.return_value = iter([[0.9, 0.8, 0.7]])
        mock_text_embedding_class.return_value = mock_model_instance

        # Create EmbeddingModel instance
        embedding_model = EmbeddingModel(model_name="test-model")

        # Call the method
        texts = ["Single document"]
        result = embedding_model.embed_documents(texts)

        # Assertions
        assert len(result) == 1
        assert result[0] == [0.9, 0.8, 0.7]