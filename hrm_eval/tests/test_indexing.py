"""
Unit tests for vector indexing.

Tests VectorIndexer with batch processing and format conversion.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import json

from ..rag_vector_store import VectorIndexer, VectorStore, EmbeddingGenerator


class TestVectorIndexer:
    """Test VectorIndexer class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock(spec=VectorStore)
        store.add_documents = Mock()
        store.get_stats = Mock(return_value={"total_documents": 10})
        return store
    
    @pytest.fixture
    def mock_embedding_generator(self):
        """Create mock embedding generator."""
        generator = Mock(spec=EmbeddingGenerator)
        generator.encode_batch = Mock(return_value=[[0.1] * 384, [0.2] * 384])
        return generator
    
    def test_initialization(self, mock_vector_store, mock_embedding_generator):
        """Test VectorIndexer initialization."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        assert indexer.vector_store is not None
        assert indexer.embedding_generator is not None
        assert indexer.indexed_count == 0
    
    def test_index_test_cases(self, mock_vector_store, mock_embedding_generator):
        """Test indexing test cases."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        test_cases = [
            {"id": "test1", "description": "Test case 1", "type": "positive"},
            {"id": "test2", "description": "Test case 2", "type": "negative"},
        ]
        
        indexer.index_test_cases(test_cases, batch_size=100, show_progress=False)
        
        assert indexer.indexed_count == 2
        mock_embedding_generator.encode_batch.assert_called_once()
        mock_vector_store.add_documents.assert_called_once()
    
    def test_index_test_cases_large_batch(self, mock_vector_store, mock_embedding_generator):
        """Test indexing with multiple batches."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        test_cases = [{"id": f"test{i}", "description": f"Test {i}"} for i in range(250)]
        
        indexer.index_test_cases(test_cases, batch_size=100, show_progress=False)
        
        assert indexer.indexed_count == 250
        assert mock_vector_store.add_documents.call_count == 3
    
    def test_index_test_cases_empty_list(self, mock_vector_store, mock_embedding_generator):
        """Test indexing empty list."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        indexer.index_test_cases([], show_progress=False)
        
        assert indexer.indexed_count == 0
        mock_vector_store.add_documents.assert_not_called()
    
    def test_index_requirements(self, mock_vector_store, mock_embedding_generator):
        """Test indexing requirements."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        requirements = [
            {"id": "req1", "summary": "Requirement 1", "description": "Description 1"},
            {"id": "req2", "summary": "Requirement 2", "description": "Description 2"},
        ]
        
        indexer.index_requirements(requirements, batch_size=100, show_progress=False)
        
        assert indexer.indexed_count == 2
        mock_embedding_generator.encode_batch.assert_called_once()
    
    def test_index_from_generated_results(self, mock_vector_store, mock_embedding_generator):
        """Test indexing from HRM-generated TestCase objects."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        mock_test_case = Mock()
        mock_test_case.id = "TC-001"
        mock_test_case.description = "Test description"
        mock_test_case.type.value = "positive"
        mock_test_case.priority.value = "P1"
        mock_test_case.labels = ["auth", "security"]
        mock_test_case.preconditions = ["System running"]
        mock_test_case.test_steps = [Mock(step_number=1, action="Step 1")]
        mock_test_case.expected_results = [Mock(result="Result 1")]
        mock_test_case.source_story_id = "US-001"
        mock_test_case.test_data = "Test data"
        
        indexer.index_from_generated_results([mock_test_case], source="hrm", batch_size=100)
        
        assert indexer.indexed_count == 1
    
    def test_index_from_generated_results_with_errors(self, mock_vector_store, mock_embedding_generator):
        """Test indexing with some failed conversions."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        mock_test_case_good = Mock()
        mock_test_case_good.id = "TC-001"
        mock_test_case_good.description = "Good test"
        mock_test_case_good.type.value = "positive"
        mock_test_case_good.priority.value = "P1"
        mock_test_case_good.labels = []
        mock_test_case_good.test_steps = []
        mock_test_case_good.expected_results = []
        
        mock_test_case_bad = Mock()
        mock_test_case_bad.id = "TC-002"
        mock_test_case_bad.description = None
        mock_test_case_bad.type.value = "positive"
        mock_test_case_bad.priority = Mock(side_effect=AttributeError())
        
        indexer.index_from_generated_results(
            [mock_test_case_good, mock_test_case_bad],
            batch_size=100
        )
        
        assert indexer.indexed_count >= 1
    
    def test_index_from_jsonl(self, mock_vector_store, mock_embedding_generator):
        """Test indexing from JSONL file."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({"id": "test1", "description": "Test 1"}, f)
            f.write('\n')
            json.dump({"id": "test2", "description": "Test 2"}, f)
            f.write('\n')
            temp_path = f.name
        
        try:
            indexer.index_from_jsonl(temp_path, item_type="test_case", batch_size=100)
            
            assert indexer.indexed_count == 2
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_index_from_jsonl_invalid_json(self, mock_vector_store, mock_embedding_generator):
        """Test indexing from JSONL with invalid lines."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({"id": "test1", "description": "Test 1"}, f)
            f.write('\n')
            f.write('invalid json line\n')
            json.dump({"id": "test2", "description": "Test 2"}, f)
            f.write('\n')
            temp_path = f.name
        
        try:
            indexer.index_from_jsonl(temp_path, item_type="test_case", batch_size=100)
            
            assert indexer.indexed_count == 2
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_index_from_jsonl_file_not_found(self, mock_vector_store, mock_embedding_generator):
        """Test indexing from non-existent file."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        indexer.index_from_jsonl("non_existent_file.jsonl", item_type="test_case")
        
        assert indexer.indexed_count == 0
    
    def test_get_indexing_stats(self, mock_vector_store, mock_embedding_generator):
        """Test getting indexing statistics."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        test_cases = [{"id": f"test{i}", "description": f"Test {i}"} for i in range(5)]
        indexer.index_test_cases(test_cases, show_progress=False)
        
        stats = indexer.get_indexing_stats()
        
        assert "total_indexed_this_session" in stats
        assert stats["total_indexed_this_session"] == 5
        assert "vector_store_stats" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
