"""
Unit tests for RAG retrieval logic.

Tests RAGRetriever with context building and similarity search.
"""

import pytest
from unittest.mock import Mock, MagicMock

from ..rag_vector_store import RAGRetriever, VectorStore, EmbeddingGenerator


class TestRAGRetriever:
    """Test RAGRetriever class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock(spec=VectorStore)
        store.search = Mock(return_value=[
            {
                "id": "test1",
                "metadata": {
                    "description": "Test case 1",
                    "type": "positive",
                    "priority": "P1",
                },
                "similarity": 0.85,
            },
            {
                "id": "test2",
                "metadata": {
                    "description": "Test case 2",
                    "type": "negative",
                    "priority": "P2",
                },
                "similarity": 0.75,
            },
        ])
        return store
    
    @pytest.fixture
    def mock_embedding_generator(self):
        """Create mock embedding generator."""
        generator = Mock(spec=EmbeddingGenerator)
        generator.encode_requirement = Mock(return_value=[0.1] * 384)
        generator.encode = Mock(return_value=[0.2] * 384)
        return generator
    
    def test_initialization(self, mock_vector_store, mock_embedding_generator):
        """Test RAGRetriever initialization."""
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        assert retriever.vector_store is not None
        assert retriever.embedding_generator is not None
    
    def test_retrieve_similar_test_cases(self, mock_vector_store, mock_embedding_generator):
        """Test retrieving similar test cases."""
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        requirement = {
            "summary": "User authentication",
            "description": "Implement login functionality",
        }
        
        results = retriever.retrieve_similar_test_cases(requirement, top_k=5)
        
        assert len(results) == 2
        assert results[0]["similarity"] >= 0.7
        mock_embedding_generator.encode_requirement.assert_called_once()
        mock_vector_store.search.assert_called_once()
    
    def test_retrieve_with_min_similarity_filter(self, mock_vector_store, mock_embedding_generator):
        """Test filtering by minimum similarity."""
        mock_vector_store.search.return_value = [
            {"id": "test1", "similarity": 0.9},
            {"id": "test2", "similarity": 0.6},
            {"id": "test3", "similarity": 0.5},
        ]
        
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        requirement = {"summary": "Test"}
        results = retriever.retrieve_similar_test_cases(requirement, min_similarity=0.7)
        
        assert len(results) == 1
        assert results[0]["similarity"] >= 0.7
    
    def test_retrieve_by_text(self, mock_vector_store, mock_embedding_generator):
        """Test retrieving by text query."""
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        text = "authentication security test"
        results = retriever.retrieve_by_text(text, top_k=5)
        
        assert len(results) == 2
        mock_embedding_generator.encode.assert_called_once_with(text)
    
    def test_build_context(self, mock_vector_store, mock_embedding_generator):
        """Test building context from retrieved tests."""
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        requirement = {"summary": "Test"}
        retrieved_tests = [
            {
                "id": "test1",
                "metadata": {
                    "description": "Authentication test",
                    "type": "positive",
                    "priority": "P1",
                    "labels": ["auth", "security"],
                },
                "similarity": 0.85,
            },
        ]
        
        context = retriever.build_context(requirement, retrieved_tests)
        
        assert isinstance(context, str)
        assert "Historical Test Case Examples" in context
        assert "Authentication test" in context
        assert "positive" in context
    
    def test_build_context_empty_results(self, mock_vector_store, mock_embedding_generator):
        """Test building context with no results."""
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        requirement = {"summary": "Test"}
        context = retriever.build_context(requirement, [])
        
        assert context == ""
    
    def test_build_context_with_metadata(self, mock_vector_store, mock_embedding_generator):
        """Test context includes metadata and similarity scores."""
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        requirement = {"summary": "Test"}
        retrieved_tests = [
            {
                "id": "test1",
                "metadata": {
                    "description": "Test case",
                    "type": "positive",
                    "priority": "P1",
                    "preconditions": ["System running", "User logged in"],
                    "test_steps": [
                        {"step": 1, "action": "Navigate to page"},
                        {"step": 2, "action": "Click button"},
                    ],
                },
                "similarity": 0.85,
            },
        ]
        
        context = retriever.build_context(requirement, retrieved_tests, include_metadata=True)
        
        assert "**Similarity Score:** 0.85" in context or "Similarity Score: 0.85" in context
        assert "Preconditions:" in context
        assert "Test Steps:" in context
    
    def test_build_compact_context(self, mock_vector_store, mock_embedding_generator):
        """Test building compact context."""
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        retrieved_tests = [
            {"id": "test1", "metadata": {"description": "Test " * 50, "type": "positive", "priority": "P1"}},
            {"id": "test2", "metadata": {"description": "Another test", "type": "negative", "priority": "P2"}},
        ]
        
        context = retriever.build_compact_context(retrieved_tests)
        
        assert isinstance(context, str)
        assert len(context) < 500
        assert "[positive/P1]" in context
    
    def test_get_retrieval_stats(self, mock_vector_store, mock_embedding_generator):
        """Test getting retrieval statistics."""
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        requirement = {"summary": "Test"}
        retrieved_tests = [
            {"id": "test1", "metadata": {"type": "positive", "priority": "P1"}, "similarity": 0.9},
            {"id": "test2", "metadata": {"type": "negative", "priority": "P2"}, "similarity": 0.8},
            {"id": "test3", "metadata": {"type": "positive", "priority": "P1"}, "similarity": 0.75},
        ]
        
        stats = retriever.get_retrieval_stats(requirement, retrieved_tests)
        
        assert stats["num_retrieved"] == 3
        assert stats["avg_similarity"] > 0.7
        assert stats["max_similarity"] == 0.9
        assert stats["min_similarity"] == 0.75
        assert "types_distribution" in stats
        assert stats["types_distribution"]["positive"] == 2
    
    def test_get_retrieval_stats_empty(self, mock_vector_store, mock_embedding_generator):
        """Test retrieval stats with no results."""
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        requirement = {"summary": "Test"}
        stats = retriever.get_retrieval_stats(requirement, [])
        
        assert stats["num_retrieved"] == 0
        assert stats["avg_similarity"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
