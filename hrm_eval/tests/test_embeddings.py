"""
Unit tests for embedding generation.

Tests EmbeddingGenerator with various input types.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from ..rag_vector_store import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.array([[0.1] * 384])
            mock_st.return_value = mock_model
            yield mock_model
    
    def test_initialization(self, mock_sentence_transformer):
        """Test EmbeddingGenerator initialization."""
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        
        assert generator.model is not None
        assert generator.embedding_dim == 384
        assert generator.model_name == "all-MiniLM-L6-v2"
    
    def test_encode_single_text(self, mock_sentence_transformer):
        """Test encoding a single text string."""
        mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384])
        
        generator = EmbeddingGenerator()
        
        text = "Test text for embedding"
        embedding = generator.encode(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        mock_sentence_transformer.encode.assert_called_once()
    
    def test_encode_multiple_texts(self, mock_sentence_transformer):
        """Test encoding multiple texts."""
        mock_sentence_transformer.encode.return_value = np.array([
            [0.1] * 384,
            [0.2] * 384,
        ])
        
        generator = EmbeddingGenerator()
        
        texts = ["Text 1", "Text 2"]
        embeddings = generator.encode(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
    
    def test_encode_empty_list(self, mock_sentence_transformer):
        """Test encoding empty list."""
        generator = EmbeddingGenerator()
        
        embeddings = generator.encode([])
        
        assert embeddings == []
    
    def test_encode_test_case(self, mock_sentence_transformer):
        """Test encoding a test case dictionary."""
        mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384])
        
        generator = EmbeddingGenerator()
        
        test_case = {
            "description": "Test user authentication",
            "type": "positive",
            "priority": "P1",
            "labels": ["auth", "security"],
        }
        
        embedding = generator.encode_test_case(test_case)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
    
    def test_encode_test_case_minimal(self, mock_sentence_transformer):
        """Test encoding test case with minimal fields."""
        mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384])
        
        generator = EmbeddingGenerator()
        
        test_case = {"description": "Minimal test case"}
        
        embedding = generator.encode_test_case(test_case)
        
        assert isinstance(embedding, list)
    
    def test_encode_requirement(self, mock_sentence_transformer):
        """Test encoding a requirement dictionary."""
        mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384])
        
        generator = EmbeddingGenerator()
        
        requirement = {
            "summary": "User login feature",
            "description": "As a user, I want to log in securely",
            "acceptance_criteria": [
                {"criteria": "Valid credentials accepted"},
                {"criteria": "Invalid credentials rejected"},
            ],
            "tech_stack": ["FastAPI", "PostgreSQL"],
        }
        
        embedding = generator.encode_requirement(requirement)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
    
    def test_encode_batch(self, mock_sentence_transformer):
        """Test batch encoding of items."""
        mock_sentence_transformer.encode.return_value = np.array([
            [0.1] * 384,
            [0.2] * 384,
        ])
        
        generator = EmbeddingGenerator()
        
        items = [
            {"description": "Test 1", "type": "positive"},
            {"description": "Test 2", "type": "negative"},
        ]
        
        embeddings = generator.encode_batch(items, item_type="test_case")
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
    
    def test_encode_batch_requirements(self, mock_sentence_transformer):
        """Test batch encoding of requirements."""
        mock_sentence_transformer.encode.return_value = np.array([
            [0.1] * 384,
        ])
        
        generator = EmbeddingGenerator()
        
        items = [
            {"summary": "Feature 1", "description": "Description 1"},
        ]
        
        embeddings = generator.encode_batch(items, item_type="requirement")
        
        assert len(embeddings) == 1
    
    def test_encode_batch_invalid_type(self, mock_sentence_transformer):
        """Test batch encoding with invalid item type."""
        generator = EmbeddingGenerator()
        
        items = [{"test": "data"}]
        
        with pytest.raises(ValueError):
            generator.encode_batch(items, item_type="invalid")
    
    def test_get_embedding_dimension(self, mock_sentence_transformer):
        """Test getting embedding dimension."""
        generator = EmbeddingGenerator()
        
        dim = generator.get_embedding_dimension()
        
        assert dim == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
