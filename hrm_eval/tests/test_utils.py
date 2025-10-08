"""Unit tests for utility functions."""

import pytest
import torch
from pathlib import Path
import tempfile
from ..utils.checkpoint_utils import validate_checkpoint, compare_checkpoints


class TestCheckpointUtils:
    """Tests for checkpoint utilities."""
    
    def test_validate_checkpoint_valid(self):
        """Test validation of valid checkpoint."""
        checkpoint = {
            "model.weight1": torch.randn(10, 10),
            "model.weight2": torch.randn(5, 5),
        }
        
        result = validate_checkpoint(checkpoint)
        
        assert result["valid"] == True
        assert result["num_tensors"] == 2
        assert result["num_parameters"] == 125
        assert result["has_nan"] == False
        assert result["has_inf"] == False
    
    def test_validate_checkpoint_with_nan(self):
        """Test validation catches NaN values."""
        checkpoint = {
            "model.weight1": torch.tensor([1.0, float('nan'), 3.0]),
        }
        
        with pytest.raises(Exception):
            validate_checkpoint(checkpoint, check_nan=True)
    
    def test_validate_checkpoint_with_inf(self):
        """Test validation catches Inf values."""
        checkpoint = {
            "model.weight1": torch.tensor([1.0, float('inf'), 3.0]),
        }
        
        with pytest.raises(Exception):
            validate_checkpoint(checkpoint, check_inf=True)
    
    def test_compare_checkpoints_same(self):
        """Test comparison of identical checkpoints."""
        ckpt1 = {
            "weight1": torch.randn(10, 10),
            "weight2": torch.randn(5, 5),
        }
        ckpt2 = {
            "weight1": torch.randn(10, 10),
            "weight2": torch.randn(5, 5),
        }
        
        comparison = compare_checkpoints(ckpt1, ckpt2)
        
        assert comparison["same_structure"] == True
        assert comparison["compatible"] == True
        assert len(comparison["shape_mismatches"]) == 0
    
    def test_compare_checkpoints_different_shapes(self):
        """Test comparison detects shape mismatches."""
        ckpt1 = {
            "weight1": torch.randn(10, 10),
        }
        ckpt2 = {
            "weight1": torch.randn(10, 5),
        }
        
        comparison = compare_checkpoints(ckpt1, ckpt2)
        
        assert comparison["compatible"] == False
        assert len(comparison["shape_mismatches"]) == 1

