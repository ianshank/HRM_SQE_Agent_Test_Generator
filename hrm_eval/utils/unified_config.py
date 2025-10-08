"""
Unified Configuration Management System.

Provides centralized access to all configuration values, eliminating hard-coded
values throughout the codebase. Supports configuration profiles, validation,
and environment-specific overrides.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class PathConfig(BaseModel):
    """Path configuration."""
    base_checkpoint_dir: str
    checkpoint_pattern: str
    fine_tuned_dir: str
    configs_dir: str
    results_dir: str
    logs_dir: str
    temp_dir: str
    vector_store_dir: str
    test_cases_filename: str
    requirements_filename: str
    report_filename: str
    metadata_filename: str


class RAGConfig(BaseModel):
    """RAG configuration."""
    top_k_retrieval: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1)
    
    context_slicing: Dict[str, int] = Field(
        default_factory=lambda: {
            "acceptance_criteria_max": 3,
            "test_steps_max": 3,
            "expected_results_max": 2,
            "preconditions_max": 2,
        }
    )
    
    backend: str = Field(default="chromadb", pattern="^(chromadb|pinecone)$")
    collection_name: str = "test_cases"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    distance_metric: str = Field(default="cosine", pattern="^(cosine|euclidean|dot)$")


class GenerationConfig(BaseModel):
    """Test generation configuration."""
    max_input_length: int = Field(default=100, ge=1)
    max_sequence_length: int = Field(default=512, ge=1)
    truncation: bool = True
    padding: bool = True
    
    batch_size: int = Field(default=8, ge=1)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    num_beams: int = Field(default=4, ge=1)
    max_new_tokens: int = Field(default=256, ge=1)
    
    min_test_cases_per_story: int = Field(default=1, ge=1)
    max_test_cases_per_story: int = Field(default=10, ge=1)
    default_priority: str = "P2"
    default_test_type: str = "positive"


class FineTuningConfig(BaseModel):
    """Fine-tuning configuration."""
    num_epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=2.0e-5, gt=0.0)
    warmup_steps: int = Field(default=100, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    
    optimizer: str = Field(default="adamw", pattern="^(adam|adamw|sgd)$")
    adam_beta1: float = Field(default=0.9, ge=0.0, le=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, le=1.0)
    adam_epsilon: float = Field(default=1.0e-8, gt=0.0)
    
    lr_scheduler_type: str = "linear"
    train_test_split: float = Field(default=0.8, ge=0.0, le=1.0)
    shuffle_data: bool = True
    seed: int = 42
    
    save_steps: int = Field(default=500, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    logging_steps: int = Field(default=50, ge=1)


class OutputConfig(BaseModel):
    """Output configuration."""
    formatting_width: int = Field(default=80, ge=40)
    separator_char: str = "="
    indent_spaces: int = Field(default=2, ge=0)
    
    report_formats: List[str] = ["json", "markdown"]
    include_metadata: bool = True
    include_timestamps: bool = True
    include_statistics: bool = True
    
    use_timestamps: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    compress_large_outputs: bool = False
    compression_threshold_mb: int = Field(default=10, ge=1)


class SecurityConfig(BaseModel):
    """Security configuration."""
    validate_paths: bool = True
    allow_absolute_paths: bool = False
    allowed_extensions: List[str] = [".json", ".jsonl", ".txt", ".md", ".yaml", ".yml"]
    max_file_size_mb: int = Field(default=100, ge=1)
    
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(default=10, ge=1)
    rate_limit_burst: int = Field(default=3, ge=1)
    
    sanitize_inputs: bool = True
    max_input_length: int = Field(default=10000, ge=1)
    reject_suspicious_patterns: bool = True


class DebugConfig(BaseModel):
    """Debug configuration."""
    enabled: bool = False
    verbose: bool = False
    
    profile_performance: bool = False
    profile_memory: bool = False
    profile_gpu: bool = False
    
    log_model_inputs: bool = False
    log_model_outputs: bool = False
    log_intermediate_states: bool = False
    
    breakpoint_on_error: bool = False
    breakpoint_on_warning: bool = False
    
    profiling_output_dir: str = "profiling_results"
    save_flamegraph: bool = False
    save_memory_profile: bool = False
    
    checkpoint_stages: List[str] = []


class LoggingConfig(BaseModel):
    """Logging configuration."""
    default_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    module_levels: Dict[str, str] = Field(default_factory=dict)
    
    console_output: bool = True
    file_output: bool = True
    json_format: bool = False
    
    log_dir: str = "logs"
    log_filename: Optional[str] = None
    max_file_size_mb: int = Field(default=100, ge=1)
    backup_count: int = Field(default=5, ge=0)
    
    format_string: str = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


class DeviceConfig(BaseModel):
    """Device configuration."""
    auto_select: bool = True
    preferred_device: str = Field(default="cuda", pattern="^(cuda|cpu|mps)$")
    device_id: int = Field(default=0, ge=0)
    
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    max_memory_mb: Optional[int] = None
    
    data_parallel: bool = False
    distributed: bool = False


class ModelConfigOverrides(BaseModel):
    """Model configuration overrides."""
    strict_loading: bool = False
    load_optimizer_state: bool = False
    map_location: Optional[str] = None
    
    strip_prefixes: List[str] = ["model.inner.", "model.", "module."]
    key_mappings: Dict[str, str] = {"embedding_weight": "weight"}
    
    eval_mode: bool = True
    requires_grad: bool = False


class WorkflowConfig(BaseModel):
    """Workflow configuration."""
    max_retries: int = Field(default=3, ge=0)
    retry_delay_seconds: int = Field(default=5, ge=1)
    timeout_seconds: int = Field(default=3600, ge=1)
    
    cache_models: bool = True
    cache_embeddings: bool = True
    cache_rag_results: bool = False
    
    validate_inputs: bool = True
    validate_outputs: bool = True
    
    continue_on_error: bool = False
    save_partial_results: bool = True


class DropFolderConfigOverrides(BaseModel):
    """Drop folder configuration overrides."""
    watch_interval_seconds: int = Field(default=5, ge=1)
    debounce_delay_seconds: int = Field(default=2, ge=0)
    
    process_in_parallel: bool = False
    max_concurrent_files: int = Field(default=1, ge=1)
    
    archive_after_processing: bool = True
    archive_on_error: bool = True
    delete_after_days: int = Field(default=30, ge=0)


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1)
    reload: bool = False
    
    request_timeout_seconds: int = Field(default=300, ge=1)
    keepalive_timeout_seconds: int = Field(default=5, ge=1)
    
    max_request_size_mb: int = Field(default=50, ge=1)
    max_concurrent_requests: int = Field(default=100, ge=1)
    
    cors_enabled: bool = True
    cors_origins: List[str] = ["*"]


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    enabled: bool = False
    metrics_backend: str = Field(default="prometheus", pattern="^(prometheus|statsd|cloudwatch)$")
    
    track_latency: bool = True
    track_throughput: bool = True
    track_resource_usage: bool = True
    
    alert_on_errors: bool = False
    alert_threshold_error_rate: float = Field(default=0.05, ge=0.0, le=1.0)


class FeaturesConfig(BaseModel):
    """Feature flags."""
    enable_rag: bool = True
    enable_fine_tuning: bool = True
    enable_ensemble: bool = False
    enable_caching: bool = True
    enable_profiling: bool = False
    experimental_features: bool = False


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""
    wandb_enabled: bool = False
    wandb_project: str = "hrm-test-generation"
    wandb_entity: Optional[str] = None
    
    mlflow_enabled: bool = False
    mlflow_tracking_uri: Optional[str] = None
    
    experiment_name: Optional[str] = None
    tags: List[str] = []
    notes: str = ""


class OverridesConfig(BaseModel):
    """Configuration overrides."""
    allow_override: bool = True
    override_keys: List[str] = [
        "debug.enabled",
        "logging.default_level",
        "device.preferred_device",
        "rag.top_k_retrieval",
        "generation.batch_size",
        "fine_tuning.num_epochs",
        "paths.base_checkpoint_dir",
    ]


class SystemConfig(BaseModel):
    """Complete system configuration."""
    paths: PathConfig
    rag: RAGConfig
    generation: GenerationConfig
    fine_tuning: FineTuningConfig
    output: OutputConfig
    security: SecurityConfig
    debug: DebugConfig
    logging: LoggingConfig
    device: DeviceConfig
    model: ModelConfigOverrides
    workflow: WorkflowConfig
    drop_folder: DropFolderConfigOverrides
    api: APIConfig
    monitoring: MonitoringConfig
    features: FeaturesConfig
    experiment: ExperimentConfig
    overrides: OverridesConfig
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"
        validate_assignment = True


def load_system_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    _use_cache: bool = True,
) -> SystemConfig:
    """
    Load system configuration from YAML file.
    
    Args:
        config_path: Path to system_config.yaml (defaults to configs/system_config.yaml)
        overrides: Dictionary of override values (dot notation keys)
        
    Returns:
        Validated SystemConfig object
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config validation fails
        
    Example:
        >>> config = load_system_config()
        >>> print(config.rag.top_k_retrieval)
        5
        
        >>> config = load_system_config(overrides={"rag.top_k_retrieval": 10})
        >>> print(config.rag.top_k_retrieval)
        10
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "system_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"System config not found: {config_path}")
    
    logger.debug(f"Loading system config from {config_path}")
    
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    if overrides:
        config_data = apply_overrides(config_data, overrides)
    
    env_overrides = load_env_overrides()
    if env_overrides:
        config_data = apply_overrides(config_data, env_overrides)
    
    try:
        config = SystemConfig(**config_data)
        logger.info("System configuration loaded and validated successfully")
        return config
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply override values to configuration dictionary.
    
    Supports dot notation for nested keys (e.g., "rag.top_k_retrieval").
    
    Args:
        config: Base configuration dictionary
        overrides: Override values with dot-notation keys
        
    Returns:
        Updated configuration dictionary
        
    Example:
        >>> config = {"rag": {"top_k": 5}}
        >>> overrides = {"rag.top_k": 10}
        >>> updated = apply_overrides(config, overrides)
        >>> print(updated["rag"]["top_k"])
        10
    """
    import copy
    config = copy.deepcopy(config)
    
    for key, value in overrides.items():
        keys = key.split(".")
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        logger.debug(f"Applied override: {key} = {value}")
    
    return config


def load_env_overrides() -> Dict[str, Any]:
    """
    Load configuration overrides from environment variables.
    
    Environment variables should be prefixed with HRM_CONFIG_ and use
    double underscores for nesting (e.g., HRM_CONFIG_RAG__TOP_K=10).
    
    Returns:
        Dictionary of override values
        
    Example:
        >>> os.environ["HRM_CONFIG_RAG__TOP_K"] = "10"
        >>> overrides = load_env_overrides()
        >>> print(overrides)
        {"rag.top_k": 10}
    """
    overrides = {}
    prefix = "HRM_CONFIG_"
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower().replace("__", ".")
            
            try:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif "." in value:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            except (ValueError, AttributeError):
                pass
            
            overrides[config_key] = value
            logger.debug(f"Loaded env override: {config_key} = {value}")
    
    return overrides


def get_checkpoint_path(config: SystemConfig, checkpoint_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    Get checkpoint path from configuration.
    
    Args:
        config: System configuration
        checkpoint_name: Checkpoint identifier (e.g., "step_7566" or full path)
        base_dir: Base directory (defaults to project root)
        
    Returns:
        Full checkpoint path
        
    Example:
        >>> config = load_system_config()
        >>> path = get_checkpoint_path(config, "step_7566")
        >>> print(path)
        /path/to/checkpoints_hrm_v9_optimized_step_7566
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    if Path(checkpoint_name).exists():
        return Path(checkpoint_name)
    
    if "step_" in checkpoint_name:
        checkpoint_dir = config.paths.checkpoint_pattern.format(step=checkpoint_name.replace("step_", ""))
    else:
        checkpoint_dir = config.paths.checkpoint_pattern.format(step=checkpoint_name)
    
    return base_dir / checkpoint_dir


def create_output_directory(config: SystemConfig, workflow_name: str, base_dir: Optional[Path] = None) -> Path:
    """
    Create timestamped output directory based on configuration.
    
    Args:
        config: System configuration
        workflow_name: Name of workflow for directory prefix
        base_dir: Base directory (defaults to results_dir from config)
        
    Returns:
        Created output directory path
        
    Example:
        >>> config = load_system_config()
        >>> output_dir = create_output_directory(config, "test_generation")
        >>> print(output_dir)
        results/test_generation_20251008_123456
    """
    if base_dir is None:
        base_dir = Path.cwd() / config.paths.results_dir
    
    from datetime import datetime
    
    if config.output.use_timestamps:
        timestamp = datetime.now().strftime(config.output.timestamp_format)
        dir_name = f"{workflow_name}_{timestamp}"
    else:
        dir_name = workflow_name
    
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def get_config_value(config: SystemConfig, key: str, default: Any = None) -> Any:
    """
    Get configuration value by dot-notation key.
    
    Args:
        config: System configuration
        key: Dot-notation key (e.g., "rag.top_k_retrieval")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
        
    Example:
        >>> config = load_system_config()
        >>> value = get_config_value(config, "rag.top_k_retrieval")
        >>> print(value)
        5
    """
    keys = key.split(".")
    current = config
    
    for k in keys:
        if hasattr(current, k):
            current = getattr(current, k)
        else:
            return default
    
    return current


__all__ = [
    "SystemConfig",
    "PathConfig",
    "RAGConfig",
    "GenerationConfig",
    "FineTuningConfig",
    "OutputConfig",
    "SecurityConfig",
    "DebugConfig",
    "LoggingConfig",
    "DeviceConfig",
    "ModelConfigOverrides",
    "WorkflowConfig",
    "DropFolderConfigOverrides",
    "APIConfig",
    "MonitoringConfig",
    "FeaturesConfig",
    "ExperimentConfig",
    "OverridesConfig",
    "load_system_config",
    "apply_overrides",
    "load_env_overrides",
    "get_checkpoint_path",
    "create_output_directory",
    "get_config_value",
]

