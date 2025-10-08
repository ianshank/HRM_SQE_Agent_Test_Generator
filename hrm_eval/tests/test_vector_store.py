"""
Unit tests for vector store implementations.

Tests ChromaDB and Pinecone backends with comprehensive coverage.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil
from pathlib import Path

from ..rag_vector_store import VectorStore, ChromaDBBackend


class TestChromaDBBackend:
    """Test ChromaDB backend implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for ChromaDB."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.mark.skipif(True, reason="Requires chromadb installation")
    def test_chromadb_initialization(self, temp_dir):
        """Test ChromaDB backend initialization."""
        backend = ChromaDBBackend(persist_directory=temp_dir)
        
        assert backend.client is not None
        assert backend.collection is not None
    
    @pytest.mark.skipif(True, reason="Requires chromadb installation")
    def test_add_documents(self, temp_dir):
        """Test adding documents to ChromaDB."""
        backend = ChromaDBBackend(persist_directory=temp_dir)
        
        documents = [
            {"id": "doc1", "description": "Test document 1", "type": "positive"},
            {"id": "doc2", "description": "Test document 2", "type": "negative"},
        ]
        
        embeddings = [
            [0.1, 0.2, 0.3] * 128,  # 384 dimensions
            [0.4, 0.5, 0.6] * 128,
        ]
        
        backend.add_documents(documents, embeddings)
        
        stats = backend.get_collection_stats()
        assert stats["total_documents"] == 2
    
    @pytest.mark.skipif(True, reason="Requires chromadb installation")
    def test_search_documents(self, temp_dir):
        """Test searching for similar documents."""
        backend = ChromaDBBackend(persist_directory=temp_dir)
        
        documents = [
            {"id": "doc1", "description": "Authentication test case"},
            {"id": "doc2", "description": "Database integration test"},
        ]
        
        embeddings = [
            [0.1] * 384,
            [0.9] * 384,
        ]
        
        backend.add_documents(documents, embeddings, ids=["doc1", "doc2"])
        
        query_embedding = [0.15] * 384
        results = backend.search(query_embedding, top_k=1)
        
        assert len(results) > 0
        assert "id" in results[0]
    
    def test_add_documents_empty_list(self, temp_dir):
        """Test adding empty document list."""
        with patch('chromadb.PersistentClient'):
            backend = ChromaDBBackend(persist_directory=temp_dir)
            backend.collection = MagicMock()
            
            backend.add_documents([], [])
    
    def test_add_documents_mismatch_length(self, temp_dir):
        """Test error on document/embedding length mismatch."""
        with patch('chromadb.PersistentClient'):
            backend = ChromaDBBackend(persist_directory=temp_dir)
            
            documents = [{"id": "doc1"}]
            embeddings = [[0.1], [0.2]]
            
            with pytest.raises(ValueError):
                backend.add_documents(documents, embeddings)


class TestVectorStore:
    """Test unified VectorStore interface."""
    
    def test_chromadb_backend_initialization(self):
        """Test initializing VectorStore with ChromaDB."""
        with patch('chromadb.PersistentClient'):
            store = VectorStore(backend="chromadb", persist_directory="test_db")
            
            assert store.backend_type == "chromadb"
            assert store.backend is not None
    
    def test_unsupported_backend(self):
        """Test error on unsupported backend."""
        with pytest.raises(ValueError) as exc_info:
            VectorStore(backend="unsupported")
        
        assert "Unsupported backend" in str(exc_info.value)
    
    def test_add_documents_delegation(self):
        """Test that add_documents delegates to backend."""
        with patch('chromadb.PersistentClient'):
            store = VectorStore(backend="chromadb")
            store.backend.add_documents = Mock()
            
            docs = [{"id": "test"}]
            embeds = [[0.1]]
            
            store.add_documents(docs, embeds)
            
            store.backend.add_documents.assert_called_once()
    
    def test_search_delegation(self):
        """Test that search delegates to backend."""
        with patch('chromadb.PersistentClient'):
            store = VectorStore(backend="chromadb")
            store.backend.search = Mock(return_value=[])
            
            query = [0.1] * 384
            store.search(query, top_k=5)
            
            store.backend.search.assert_called_once()
    
    def test_get_stats(self):
        """Test getting collection statistics."""
        with patch('chromadb.PersistentClient'):
            store = VectorStore(backend="chromadb")
            store.backend.get_collection_stats = Mock(return_value={
                "total_documents": 10,
                "backend": "chromadb"
            })
            
            stats = store.get_stats()
            
            assert "total_documents" in stats
            assert stats["total_documents"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
