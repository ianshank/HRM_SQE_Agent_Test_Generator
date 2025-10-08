"""
Unit tests for RAG vector store components.

Tests the vector store, embeddings, and retrieval functionality.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from hrm_eval.rag_vector_store.vector_store import VectorStore, ChromaDBBackend
from hrm_eval.rag_vector_store.embeddings import EmbeddingGenerator
from hrm_eval.rag_vector_store.retrieval import RAGRetriever


class TestEmbeddingGenerator:
    """Unit tests for EmbeddingGenerator."""
    
    @pytest.fixture
    def embedding_generator(self):
        """Create embedding generator instance."""
        return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    def test_initialization(self, embedding_generator):
        """Test embedding generator initializes correctly."""
        assert embedding_generator.model is not None
        assert embedding_generator.embedding_dim == 384
        assert embedding_generator.model_name == "all-MiniLM-L6-v2"
    
    def test_encode_single_text(self, embedding_generator):
        """Test encoding single text."""
        text = "Test case for file upload validation"
        embedding = embedding_generator.encode(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
    
    def test_encode_batch(self, embedding_generator):
        """Test encoding batch of texts."""
        texts = [
            "Test case 1: User authentication",
            "Test case 2: File upload",
            "Test case 3: Data validation"
        ]
        embeddings = embedding_generator.encode(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
    
    def test_encode_test_case(self, embedding_generator):
        """Test encoding test case dictionary."""
        test_case = {
            "description": "Verify successful file upload",
            "type": "positive",
            "priority": "P1",
            "preconditions": ["User logged in", "Valid file available"],
            "test_steps": [
                {"step": 1, "action": "Navigate to upload page"},
                {"step": 2, "action": "Select file"},
                {"step": 3, "action": "Click upload"}
            ]
        }
        
        embedding = embedding_generator.encode_test_case(test_case)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
    
    def test_encode_requirement(self, embedding_generator):
        """Test encoding requirement dictionary."""
        requirement = {
            "summary": "File upload functionality",
            "description": "Users should be able to upload files",
            "acceptance_criteria": [
                {"criteria": "File formats validated"},
                {"criteria": "Upload progress shown"}
            ]
        }
        
        embedding = embedding_generator.encode_requirement(requirement)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
    
    def test_empty_text_handling(self, embedding_generator):
        """Test handling of empty text."""
        embedding = embedding_generator.encode("")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384


class TestVectorStore:
    """Unit tests for VectorStore."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test database."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create vector store instance."""
        return VectorStore(
            backend="chromadb",
            persist_directory=temp_dir,
            collection_name="test_collection"
        )
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                "id": "test_1",
                "description": "Test file upload with valid format",
                "type": "positive",
                "priority": "P1"
            },
            {
                "id": "test_2",
                "description": "Test file upload with invalid format",
                "type": "negative",
                "priority": "P2"
            },
            {
                "id": "test_3",
                "description": "Test batch file upload",
                "type": "positive",
                "priority": "P1"
            }
        ]
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        np.random.seed(42)
        return [
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist()
        ]
    
    def test_initialization(self, vector_store):
        """Test vector store initializes correctly."""
        assert vector_store.backend is not None
        assert vector_store.backend_type == "chromadb"
    
    def test_add_documents(self, vector_store, sample_documents, sample_embeddings):
        """Test adding documents to vector store."""
        ids = [doc["id"] for doc in sample_documents]
        
        vector_store.add_documents(
            documents=sample_documents,
            embeddings=sample_embeddings,
            ids=ids
        )
        
        stats = vector_store.get_stats()
        assert stats["total_documents"] == 3
    
    def test_search(self, vector_store, sample_documents, sample_embeddings):
        """Test searching vector store."""
        # Add documents first
        ids = [doc["id"] for doc in sample_documents]
        vector_store.add_documents(sample_documents, sample_embeddings, ids)
        
        # Search with first embedding (should return itself as top result)
        results = vector_store.search(
            query_embedding=sample_embeddings[0],
            top_k=2
        )
        
        assert len(results) <= 2
        assert all('id' in r for r in results)
        assert all('similarity' in r or 'distance' in r for r in results)
    
    def test_delete_documents(self, vector_store, sample_documents, sample_embeddings):
        """Test deleting documents from vector store."""
        ids = [doc["id"] for doc in sample_documents]
        vector_store.add_documents(sample_documents, sample_embeddings, ids)
        
        # Delete one document
        vector_store.delete_documents([ids[0]])
        
        stats = vector_store.get_stats()
        assert stats["total_documents"] == 2
    
    def test_empty_search(self, vector_store, sample_embeddings):
        """Test search on empty vector store."""
        results = vector_store.search(
            query_embedding=sample_embeddings[0],
            top_k=5
        )
        
        assert len(results) == 0


class TestRAGRetriever:
    """Unit tests for RAGRetriever."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create vector store."""
        return VectorStore(
            backend="chromadb",
            persist_directory=temp_dir,
            collection_name="test_rag"
        )
    
    @pytest.fixture
    def embedding_generator(self):
        """Create embedding generator."""
        return EmbeddingGenerator()
    
    @pytest.fixture
    def rag_retriever(self, vector_store, embedding_generator):
        """Create RAG retriever."""
        return RAGRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )
    
    @pytest.fixture
    def populated_vector_store(self, vector_store, embedding_generator):
        """Populate vector store with test data."""
        test_cases = [
            {
                "description": "Verify file upload with valid MP4 format",
                "type": "positive",
                "priority": "P1",
                "labels": ["upload", "video"]
            },
            {
                "description": "Verify error handling for invalid file format",
                "type": "negative",
                "priority": "P2",
                "labels": ["upload", "validation"]
            },
            {
                "description": "Verify metadata validation on upload",
                "type": "positive",
                "priority": "P1",
                "labels": ["upload", "metadata"]
            }
        ]
        
        embeddings = embedding_generator.encode_batch(
            items=test_cases,
            item_type="test_case"
        )
        
        ids = [f"tc_{i}" for i in range(len(test_cases))]
        vector_store.add_documents(test_cases, embeddings, ids)
        
        return vector_store
    
    def test_initialization(self, rag_retriever):
        """Test RAG retriever initializes correctly."""
        assert rag_retriever.vector_store is not None
        assert rag_retriever.embedding_generator is not None
    
    def test_retrieve_by_text(self, rag_retriever, populated_vector_store):
        """Test retrieving by text query."""
        query = "file upload validation"
        
        results = rag_retriever.retrieve_by_text(
            text=query,
            top_k=3,
            min_similarity=0.0  # Low threshold for test
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all('metadata' in r for r in results)
    
    def test_retrieve_similar_test_cases(self, rag_retriever, populated_vector_store):
        """Test retrieving similar test cases for a requirement."""
        requirement = {
            "summary": "File upload feature",
            "description": "Users can upload files with validation"
        }
        
        results = rag_retriever.retrieve_similar_test_cases(
            requirement=requirement,
            top_k=2,
            min_similarity=0.0
        )
        
        assert isinstance(results, list)
        assert len(results) <= 2
    
    def test_build_context(self, rag_retriever):
        """Test building context from retrieved tests."""
        retrieved_tests = [
            {
                "metadata": {
                    "description": "Test 1",
                    "type": "positive",
                    "priority": "P1"
                },
                "similarity": 0.9
            },
            {
                "metadata": {
                    "description": "Test 2",
                    "type": "negative",
                    "priority": "P2"
                },
                "similarity": 0.8
            }
        ]
        
        context = rag_retriever.build_context(
            requirement={"summary": "Feature"},
            retrieved_tests=retrieved_tests
        )
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert "Test 1" in context
        assert "Test 2" in context
    
    def test_build_compact_context(self, rag_retriever):
        """Test building compact context."""
        retrieved_tests = [
            {
                "metadata": {
                    "description": "Test file upload",
                    "type": "positive",
                    "priority": "P1"
                }
            }
        ]
        
        context = rag_retriever.build_compact_context(retrieved_tests)
        
        assert isinstance(context, str)
        assert len(context) > 0
    
    def test_get_retrieval_stats(self, rag_retriever):
        """Test getting retrieval statistics."""
        retrieved_tests = [
            {"similarity": 0.9, "metadata": {"type": "positive", "priority": "P1"}},
            {"similarity": 0.8, "metadata": {"type": "negative", "priority": "P2"}},
            {"similarity": 0.7, "metadata": {"type": "positive", "priority": "P1"}}
        ]
        
        stats = rag_retriever.get_retrieval_stats(
            requirement={"summary": "Test"},
            retrieved_tests=retrieved_tests
        )
        
        assert stats["num_retrieved"] == 3
        assert 0.7 <= stats["avg_similarity"] <= 0.9
        assert stats["max_similarity"] == 0.9
        assert stats["min_similarity"] == 0.7
        assert "types_distribution" in stats
        assert "priorities_distribution" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
