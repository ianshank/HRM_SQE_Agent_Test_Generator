"""
Configuration management utilities.

Provides type-safe loading and validation of YAML configuration files
using Pydantic models.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field, validator


class ModelLevelConfig(BaseModel):
    """Configuration for transformer level (H or L)."""
    
    num_layers: int = Field(..., gt=0, description="Number of transformer layers")
    hidden_size: int = Field(..., gt=0, description="Hidden dimension size")
    intermediate_size: int = Field(..., gt=0, description="Intermediate layer size")
    num_attention_heads: int = Field(..., gt=0, description="Number of attention heads")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout probability")
    
    @validator("hidden_size")
    def validate_hidden_size(cls, v, values):
        """Validate hidden size is divisible by num_attention_heads."""
        return v


class LMHeadConfig(BaseModel):
    """Configuration for language modeling head."""
    
    vocab_size: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)


class QHeadConfig(BaseModel):
    """Configuration for Q-value head (RL)."""
    
    num_actions: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)


class ModelConfig(BaseModel):
    """Complete model configuration."""
    
    name: str
    vocab_size: int = Field(..., gt=0)
    embed_dim: int = Field(..., gt=0)
    num_puzzles: int = Field(..., gt=0)
    h_level: ModelLevelConfig
    l_level: ModelLevelConfig
    lm_head: LMHeadConfig
    q_head: QHeadConfig


class CheckpointConfig(BaseModel):
    """Checkpoint configuration."""
    
    base_dir: str
    primary: str
    secondary: Optional[str] = None


class DeviceConfig(BaseModel):
    """Device configuration."""
    
    type: str = Field(default="cuda", pattern="^(cuda|cpu|mps)$")
    device_id: int = Field(default=0, ge=0)
    mixed_precision: bool = False


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    
    batch_size: int = Field(..., gt=0)
    num_workers: int = Field(default=4, ge=0)
    max_steps_per_puzzle: int = Field(..., gt=0)
    timeout_seconds: int = Field(..., gt=0)
    metrics: List[str]
    save_predictions: bool = True
    save_trajectories: bool = False
    output_dir: str


class DataConfig(BaseModel):
    """Data configuration."""
    
    validation_set: str
    test_set: str
    data_format: str = Field(default="jsonl", pattern="^(jsonl|csv|pickle)$")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = Field(default="json", pattern="^(json|text)$")
    log_dir: str
    console_output: bool = True
    file_output: bool = True


class WandBConfig(BaseModel):
    """Weights & Biases configuration."""
    
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    tags: List[str] = []


class EnsembleConfig(BaseModel):
    """Ensemble configuration."""
    
    enabled: bool = False
    strategy: str = Field(
        default="weighted_average",
        pattern="^(weighted_average|voting|stacking)$"
    )
    weights: Dict[str, float] = {}
    voting_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @validator("weights")
    def validate_weights_sum(cls, v):
        """Validate that weights sum to 1.0."""
        if v and abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError("Ensemble weights must sum to 1.0")
        return v


class Config(BaseModel):
    """Root configuration combining all sub-configs."""
    
    model: ModelConfig
    checkpoint: CheckpointConfig
    device: DeviceConfig
    evaluation: EvaluationConfig
    data: DataConfig
    logging: LoggingConfig
    wandb: WandBConfig
    ensemble: EnsembleConfig
    
    class Config:
        """Pydantic config."""
        extra = "forbid"  # Raise error for unknown fields


def load_config(
    model_config_path: Path,
    eval_config_path: Path
) -> Config:
    """
    Load and merge configuration files.
    
    Args:
        model_config_path: Path to model configuration YAML
        eval_config_path: Path to evaluation configuration YAML
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config files not found
        ValueError: If config validation fails
    """
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    
    if not eval_config_path.exists():
        raise FileNotFoundError(f"Eval config not found: {eval_config_path}")
    
    with open(model_config_path, "r") as f:
        model_config_data = yaml.safe_load(f)
    
    with open(eval_config_path, "r") as f:
        eval_config_data = yaml.safe_load(f)
    
    merged_config = {**model_config_data, **eval_config_data}
    
    try:
        config = Config(**merged_config)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")
    
    return config


def save_config(config: Config, output_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        output_path: Path to save configuration
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(
            config.dict(),
            f,
            default_flow_style=False,
            sort_keys=False,
        )

