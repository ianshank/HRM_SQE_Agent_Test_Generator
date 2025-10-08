"""
Unit tests for TrainingDataCollector, focusing on data format conversion.
"""

import pytest
import json
import tempfile
from pathlib import Path

from hrm_eval.fine_tuning.data_collector import TrainingDataCollector


class TestDataCollector:
    """Test TrainingDataCollector functionality."""
    
    def test_convert_sqe_to_hrm_format(self):
        """Test conversion from SQE format to HRM format."""
        collector = TrainingDataCollector()
        
        # Create SQE format example
        sqe_example = {
            "prompt": "Generate test case for user login",
            "completion": "Test: Verify successful login with valid credentials"
        }
        
        # Convert to HRM format
        hrm_example = collector._convert_sqe_to_hrm_format(sqe_example)
        
        # Verify structure
        assert "puzzle_id" in hrm_example
        assert "input_sequence" in hrm_example
        assert "target_sequence" in hrm_example
        assert "solution_steps" in hrm_example
        assert "metadata" in hrm_example
        
        # Verify types
        assert isinstance(hrm_example["puzzle_id"], int)
        assert isinstance(hrm_example["input_sequence"], list)
        assert isinstance(hrm_example["target_sequence"], list)
        assert isinstance(hrm_example["solution_steps"], list)
        assert isinstance(hrm_example["metadata"], dict)
        
        # Verify token sequences are non-empty
        assert len(hrm_example["input_sequence"]) > 0
        assert len(hrm_example["target_sequence"]) > 0
        
        # Verify metadata
        assert hrm_example["metadata"]["source"] == "sqe_augmentation"
        assert hrm_example["metadata"]["original_format"] == "prompt_completion"
    
    def test_augment_with_sqe_data(self):
        """Test augmentation with SQE data file."""
        collector = TrainingDataCollector()
        
        # Create temporary SQE file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sqe_path = Path(f.name)
            
            # Write SQE examples in prompt/completion format
            f.write(json.dumps({
                "prompt": "Test prompt 1",
                "completion": "Test completion 1"
            }) + "\n")
            f.write(json.dumps({
                "prompt": "Test prompt 2",
                "completion": "Test completion 2"
            }) + "\n")
        
        try:
            # Augment with SQE data
            initial_count = len(collector.examples)
            collector.augment_with_sqe_data(str(sqe_path))
            
            # Verify examples were added
            assert len(collector.examples) == initial_count + 2
            
            # Verify all examples have correct format
            for example in collector.examples:
                assert "input_sequence" in example
                assert "target_sequence" in example
                assert "puzzle_id" in example
                assert "solution_steps" in example
                assert "metadata" in example
        
        finally:
            # Cleanup
            sqe_path.unlink()
    
    def test_mixed_format_augmentation(self):
        """Test augmentation with mixed format data."""
        collector = TrainingDataCollector()
        
        # Create temporary file with mixed formats
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sqe_path = Path(f.name)
            
            # Write one SQE format (prompt/completion)
            f.write(json.dumps({
                "prompt": "Test prompt",
                "completion": "Test completion"
            }) + "\n")
            
            # Write one HRM format (already converted)
            f.write(json.dumps({
                "puzzle_id": 42,
                "input_sequence": [1, 2, 3],
                "target_sequence": [4, 5, 6],
                "solution_steps": [{"action": 0, "state": None}],
                "metadata": {"source": "existing"}
            }) + "\n")
        
        try:
            # Augment
            collector.augment_with_sqe_data(str(sqe_path))
            
            # Verify both were added
            assert len(collector.examples) == 2
            
            # Verify all have correct format
            for example in collector.examples:
                assert "input_sequence" in example
                assert "target_sequence" in example
        
        finally:
            # Cleanup
            sqe_path.unlink()
    
    def test_data_format_consistency(self):
        """Test that all collected data has consistent format."""
        collector = TrainingDataCollector()
        
        # Add mock examples with different origins
        collector.examples = [
            {
                "puzzle_id": 1,
                "input_sequence": [1, 2],
                "target_sequence": [3, 4],
                "solution_steps": [],
                "metadata": {"source": "manual"}
            },
            collector._convert_sqe_to_hrm_format({
                "prompt": "Test",
                "completion": "Result"
            })
        ]
        
        # Verify all have same keys structure
        required_keys = {"puzzle_id", "input_sequence", "target_sequence", "solution_steps", "metadata"}
        
        for example in collector.examples:
            assert required_keys.issubset(set(example.keys()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
