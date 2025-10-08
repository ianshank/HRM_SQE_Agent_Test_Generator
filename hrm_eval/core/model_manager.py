"""
Model Management Module.

Provides centralized model loading, checkpoint handling, and caching to eliminate
duplication across workflows and ensure consistent model management.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import hashlib

from ..models import HRMModel
from ..models.hrm_model import HRMConfig
from ..utils.checkpoint_utils import load_checkpoint, validate_checkpoint
from ..utils.unified_config import SystemConfig, get_checkpoint_path

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model: HRMModel
    checkpoint_path: Path
    device: torch.device
    checkpoint_step: Optional[int] = None
    validation_results: Optional[Dict[str, Any]] = None
    cache_key: Optional[str] = None


class ModelManager:
    """
    Centralized model and checkpoint management.
    
    Handles:
    - Model loading with consistent checkpoint handling
    - State dict prefix stripping and key mapping
    - Model caching to avoid reloading
    - Checkpoint validation and verification
    - Device management
    
    Example:
        >>> manager = ModelManager(config, device="cuda")
        >>> model_info = manager.load_model("step_7566", use_cache=True)
        >>> print(f"Model loaded: {model_info.checkpoint_path}")
        >>> 
        >>> available = manager.list_available_checkpoints()
        >>> print(f"Found {len(available)} checkpoints")
    """
    
    def __init__(
        self,
        config: SystemConfig,
        device: Optional[str] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize model manager.
        
        Args:
            config: System configuration
            device: Device to load models on (cuda/cpu/mps), auto-detects if None
            enable_cache: Enable model caching
        """
        self.config = config
        self.cache: Dict[str, ModelInfo] = {}
        self.enable_cache = enable_cache
        
        if device is None:
            if config.device.auto_select:
                self.device = self._auto_select_device()
            else:
                self.device = torch.device(config.device.preferred_device)
        else:
            self.device = torch.device(device)
        
        logger.info(f"ModelManager initialized on device: {self.device}")
        logger.info(f"Caching: {'enabled' if enable_cache else 'disabled'}")
    
    def _auto_select_device(self) -> torch.device:
        """Auto-select best available device."""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.device.device_id}")
            logger.info(f"Auto-selected CUDA device: {device}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Auto-selected MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Auto-selected CPU device")
        
        return device
    
    def load_model(
        self,
        checkpoint_name: str,
        model_config: Optional[HRMConfig] = None,
        base_dir: Optional[Path] = None,
        use_cache: bool = True,
        validate: bool = True,
    ) -> ModelInfo:
        """
        Load HRM model from checkpoint.
        
        Args:
            checkpoint_name: Checkpoint identifier (e.g., "step_7566")
            model_config: Model configuration (loads from config if None)
            base_dir: Base directory for checkpoints
            use_cache: Use cached model if available
            validate: Validate checkpoint before loading
            
        Returns:
            ModelInfo object with loaded model and metadata
            
        Raises:
            FileNotFoundError: If checkpoint not found
            ValueError: If checkpoint validation fails
            
        Example:
            >>> manager = ModelManager(config)
            >>> model_info = manager.load_model("step_7566")
            >>> predictions = model_info.model(input_ids, puzzle_ids)
        """
        cache_key = self._generate_cache_key(checkpoint_name, str(self.device))
        
        if use_cache and self.enable_cache and cache_key in self.cache:
            logger.info(f"Using cached model: {checkpoint_name}")
            return self.cache[cache_key]
        
        logger.info(f"Loading model from checkpoint: {checkpoint_name}")
        
        checkpoint_path = self.get_checkpoint_path(checkpoint_name, base_dir)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.debug(f"Loading checkpoint from: {checkpoint_path}")
        
        if model_config is None:
            from ..utils.config_utils import load_config
            base_path = Path(__file__).parent.parent
            full_config = load_config(
                model_config_path=base_path / self.config.paths.configs_dir / "model_config.yaml",
                eval_config_path=base_path / self.config.paths.configs_dir / "eval_config.yaml"
            )
            model_config = HRMConfig.from_yaml_config(full_config)
            logger.debug("Loaded model config from YAML")
        
        model = HRMModel(model_config)
        logger.debug(f"Initialized HRM model")
        
        checkpoint = load_checkpoint(str(checkpoint_path), device=str(self.device))
        
        validation_results = None
        if validate:
            logger.debug("Validating checkpoint...")
            validation_results = validate_checkpoint(checkpoint)
            if not validation_results["valid"]:
                raise ValueError(f"Checkpoint validation failed: {validation_results['errors']}")
        
        state_dict = self._extract_state_dict(checkpoint)
        
        state_dict = self._process_state_dict(state_dict)
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=self.config.model.strict_loading)
        
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
        
        model.to(self.device)
        
        if self.config.model.eval_mode:
            model.eval()
        
        if not self.config.model.requires_grad:
            for param in model.parameters():
                param.requires_grad = False
        
        checkpoint_step = self._extract_checkpoint_step(checkpoint_name)
        
        model_info = ModelInfo(
            model=model,
            checkpoint_path=checkpoint_path,
            device=self.device,
            checkpoint_step=checkpoint_step,
            validation_results=validation_results,
            cache_key=cache_key,
        )
        
        if self.enable_cache:
            self.cache[cache_key] = model_info
            logger.debug(f"Cached model with key: {cache_key}")
        
        logger.info(f"Model loaded successfully from {checkpoint_path.name}")
        return model_info
    
    def _extract_state_dict(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract state dict from checkpoint."""
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        else:
            return checkpoint
    
    def _process_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process state dict by stripping prefixes and applying key mappings."""
        processed = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            for prefix in self.config.model.strip_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break
            
            for old_key, mapped_key in self.config.model.key_mappings.items():
                if old_key in new_key:
                    new_key = new_key.replace(old_key, mapped_key)
            
            processed[new_key] = value
        
        if processed != state_dict:
            logger.debug(f"Processed state dict: {len(state_dict)} -> {len(processed)} keys")
        
        return processed
    
    def _generate_cache_key(self, checkpoint_name: str, device: str) -> str:
        """Generate cache key for model."""
        key_string = f"{checkpoint_name}_{device}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _extract_checkpoint_step(self, checkpoint_name: str) -> Optional[int]:
        """
        Extract step number from checkpoint name.
        
        Delegates to centralized function in common_utils to avoid duplication.
        
        Args:
            checkpoint_name: Checkpoint name
            
        Returns:
            Step number or None
        """
        from .common_utils import extract_checkpoint_step
        return extract_checkpoint_step(checkpoint_name)
    
    def get_checkpoint_path(self, checkpoint_name: str, base_dir: Optional[Path] = None) -> Path:
        """
        Get full checkpoint path from name.
        
        Args:
            checkpoint_name: Checkpoint identifier
            base_dir: Base directory (defaults to current directory)
            
        Returns:
            Full checkpoint path
            
        Example:
            >>> manager = ModelManager(config)
            >>> path = manager.get_checkpoint_path("step_7566")
            >>> print(path)
            /path/to/checkpoints_hrm_v9_optimized_step_7566
        """
        return get_checkpoint_path(self.config, checkpoint_name, base_dir)
    
    def list_available_checkpoints(self, base_dir: Optional[Path] = None) -> List[Tuple[str, Path]]:
        """
        List all available checkpoints.
        
        Args:
            base_dir: Base directory to search (defaults to current directory)
            
        Returns:
            List of (checkpoint_name, checkpoint_path) tuples
            
        Example:
            >>> manager = ModelManager(config)
            >>> checkpoints = manager.list_available_checkpoints()
            >>> for name, path in checkpoints:
            ...     print(f"{name}: {path}")
        """
        if base_dir is None:
            base_dir = Path.cwd()
        
        checkpoints = []
        pattern = self.config.paths.base_checkpoint_dir + "*"
        
        for checkpoint_dir in base_dir.glob(pattern):
            if checkpoint_dir.is_dir():
                checkpoint_name = checkpoint_dir.name
                checkpoints.append((checkpoint_name, checkpoint_dir))
        
        checkpoints.sort(key=lambda x: self._extract_checkpoint_step(x[0]) or 0)
        
        logger.debug(f"Found {len(checkpoints)} checkpoints in {base_dir}")
        return checkpoints
    
    def validate_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Validate checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Validation results dictionary
            
        Example:
            >>> manager = ModelManager(config)
            >>> results = manager.validate_checkpoint(Path("checkpoint.pt"))
            >>> if results["valid"]:
            ...     print("Checkpoint is valid")
        """
        if not checkpoint_path.exists():
            return {
                "valid": False,
                "errors": [f"Checkpoint not found: {checkpoint_path}"],
            }
        
        try:
            checkpoint = load_checkpoint(str(checkpoint_path), device="cpu")
            validation_results = validate_checkpoint(checkpoint)
            return validation_results
        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
            }
    
    def clear_cache(self):
        """Clear the model cache."""
        self.cache.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached models.
        
        Returns:
            Dictionary with cache statistics
            
        Example:
            >>> manager = ModelManager(config)
            >>> info = manager.get_cache_info()
            >>> print(f"Cached models: {info['count']}")
        """
        return {
            "count": len(self.cache),
            "enabled": self.enable_cache,
            "models": [
                {
                    "cache_key": info.cache_key,
                    "checkpoint": str(info.checkpoint_path.name),
                    "device": str(info.device),
                    "step": info.checkpoint_step,
                }
                for info in self.cache.values()
            ],
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ModelManager(device={self.device}, cached={len(self.cache)})"


__all__ = ["ModelManager", "ModelInfo"]

