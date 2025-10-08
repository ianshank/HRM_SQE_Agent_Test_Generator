"""
Checkpoint loading and validation utilities.

Provides robust checkpoint loading with validation, error handling,
and compatibility checking.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class CheckpointValidationError(Exception):
    """Exception raised when checkpoint validation fails."""
    pass


def validate_checkpoint(
    checkpoint: Dict[str, Any],
    expected_keys: Optional[List[str]] = None,
    check_nan: bool = True,
    check_inf: bool = True,
) -> Dict[str, Any]:
    """
    Validate checkpoint integrity and structure.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        expected_keys: List of expected keys in checkpoint
        check_nan: Check for NaN values in tensors
        check_inf: Check for Inf values in tensors
        
    Returns:
        Validation results dictionary
        
    Raises:
        CheckpointValidationError: If validation fails
    """
    validation_results = {
        "valid": True,
        "num_parameters": 0,
        "num_tensors": 0,
        "has_nan": False,
        "has_inf": False,
        "missing_keys": [],
        "extra_keys": [],
    }
    
    if not isinstance(checkpoint, dict):
        raise CheckpointValidationError(
            f"Checkpoint must be a dictionary, got {type(checkpoint)}"
        )
    
    if expected_keys:
        checkpoint_keys = set(checkpoint.keys())
        expected_keys_set = set(expected_keys)
        
        missing = expected_keys_set - checkpoint_keys
        extra = checkpoint_keys - expected_keys_set
        
        if missing:
            validation_results["missing_keys"] = list(missing)
            logger.warning(f"Missing expected keys: {missing}")
        
        if extra:
            validation_results["extra_keys"] = list(extra)
            logger.debug(f"Extra keys found: {extra}")
    
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            validation_results["num_tensors"] += 1
            validation_results["num_parameters"] += value.numel()
            
            if check_nan and torch.isnan(value).any():
                validation_results["has_nan"] = True
                validation_results["valid"] = False
                logger.error(f"NaN values found in tensor: {key}")
            
            if check_inf and torch.isinf(value).any():
                validation_results["has_inf"] = True
                validation_results["valid"] = False
                logger.error(f"Inf values found in tensor: {key}")
    
    logger.info(
        f"Checkpoint validation complete: {validation_results['num_tensors']} tensors, "
        f"{validation_results['num_parameters']:,} parameters"
    )
    
    if not validation_results["valid"]:
        raise CheckpointValidationError(
            f"Checkpoint validation failed: {validation_results}"
        )
    
    return validation_results


def load_checkpoint(
    checkpoint_path: Path,
    device: str = "cpu",
    strict: bool = True,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Load PyTorch checkpoint with validation and error handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        strict: Strictly enforce checkpoint structure
        validate: Run validation checks
        
    Returns:
        Loaded checkpoint dictionary
        
    Raises:
        FileNotFoundError: If checkpoint file not found
        CheckpointValidationError: If validation fails
        RuntimeError: If loading fails
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    logger.info(f"Checkpoint loaded successfully, size: {checkpoint_path.stat().st_size / (1024**2):.2f} MB")
    
    if validate:
        validation_results = validate_checkpoint(checkpoint)
        logger.info(f"Checkpoint validation: {validation_results}")
    
    return checkpoint


def extract_model_state(
    checkpoint: Dict[str, Any],
    state_dict_key: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract model state dictionary from checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint
        state_dict_key: Key containing model state dict (auto-detected if None)
        
    Returns:
        Model state dictionary
        
    Raises:
        ValueError: If state dict cannot be found
    """
    if state_dict_key:
        if state_dict_key not in checkpoint:
            raise ValueError(f"State dict key '{state_dict_key}' not found in checkpoint")
        return checkpoint[state_dict_key]
    
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    
    if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        logger.info("Checkpoint appears to be a raw state dict")
        return checkpoint
    
    raise ValueError("Could not find model state dict in checkpoint")


def get_checkpoint_info(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata and information from checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint
        
    Returns:
        Dictionary with checkpoint information
    """
    info = {
        "num_tensors": 0,
        "total_parameters": 0,
        "keys": list(checkpoint.keys()),
        "layer_info": [],
    }
    
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            info["num_tensors"] += 1
            info["total_parameters"] += value.numel()
            
            info["layer_info"].append({
                "name": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "num_params": value.numel(),
            })
    
    return info


def compare_checkpoints(
    checkpoint1: Dict[str, Any],
    checkpoint2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two checkpoints for compatibility.
    
    Args:
        checkpoint1: First checkpoint
        checkpoint2: Second checkpoint
        
    Returns:
        Comparison results
    """
    keys1 = set(checkpoint1.keys())
    keys2 = set(checkpoint2.keys())
    
    comparison = {
        "same_structure": keys1 == keys2,
        "keys_only_in_first": list(keys1 - keys2),
        "keys_only_in_second": list(keys2 - keys1),
        "common_keys": list(keys1 & keys2),
        "shape_mismatches": [],
    }
    
    for key in comparison["common_keys"]:
        val1 = checkpoint1[key]
        val2 = checkpoint2[key]
        
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            if val1.shape != val2.shape:
                comparison["shape_mismatches"].append({
                    "key": key,
                    "shape1": list(val1.shape),
                    "shape2": list(val2.shape),
                })
    
    comparison["compatible"] = (
        comparison["same_structure"] and
        len(comparison["shape_mismatches"]) == 0
    )
    
    return comparison

