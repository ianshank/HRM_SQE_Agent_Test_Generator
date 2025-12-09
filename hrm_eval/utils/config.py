"""
Configuration management module for HRM SQE Agent Test Generator.

Provides centralized, environment-aware configuration with:
- Environment variable overrides (HRM_* prefix)
- YAML file configuration
- Pydantic validation
- Type-safe access

NO HARDCODED VALUES - all configuration through files or environment.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
import logging

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================

class APIConfig(BaseModel):
    """API server configuration."""

    host: str = Field(default="0.0.0.0", description="API host address")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    workers: int = Field(default=4, ge=1, le=32, description="Number of workers")
    request_timeout: int = Field(default=300, ge=1, description="Request timeout in seconds")
    rate_limit_per_minute: int = Field(default=100, ge=1, description="Rate limit per minute")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class ModelConfig(BaseModel):
    """HRM model configuration."""

    model_path: str = Field(
        default="checkpoints_hrm_v9_optimized_step_7566",
        description="Path to model checkpoint"
    )
    default_checkpoint: str = Field(
        default="step_7566",
        description="Default checkpoint to use"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device: cuda, cpu, mps (auto-detected if None)"
    )
    mixed_precision: bool = Field(
        default=False,
        description="Enable mixed precision inference"
    )


class RAGConfig(BaseModel):
    """RAG vector store configuration."""

    backend: str = Field(
        default="chromadb",
        description="Vector store backend: chromadb or pinecone"
    )
    persist_directory: str = Field(
        default="vector_store_db",
        description="Directory for vector store persistence"
    )
    collection_name: str = Field(
        default="test_cases_requirements",
        description="Collection name"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    top_k: int = Field(default=5, ge=1, le=100, description="Number of similar items to retrieve")
    min_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )


class LLMConfig(BaseModel):
    """LLM configuration for SQE Agent."""

    provider: str = Field(
        default="openai",
        description="LLM provider: openai or anthropic"
    )
    model: str = Field(default="gpt-4", description="Model name")
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Generation temperature"
    )
    max_tokens: int = Field(
        default=2000,
        ge=1,
        le=128000,
        description="Maximum tokens to generate"
    )


class GenerationConfig(BaseModel):
    """Test generation configuration."""

    mode: str = Field(
        default="hybrid",
        description="Generation mode: hrm_only, sqe_only, hybrid"
    )
    merge_strategy: str = Field(
        default="weighted",
        description="Merge strategy: weighted, union, intersection"
    )
    hrm_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="HRM weight in hybrid mode"
    )
    sqe_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="SQE weight in hybrid mode"
    )
    batch_size: int = Field(default=8, ge=1, le=128, description="Batch size")
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Generation temperature"
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid_modes = {"hrm_only", "sqe_only", "hybrid"}
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        return v


class DropFolderConfig(BaseModel):
    """Drop folder configuration."""

    directory: str = Field(default="drop_folder", description="Drop folder base directory")
    watch_interval: int = Field(default=5, ge=1, description="Watch interval in seconds")
    debounce_delay: int = Field(default=2, ge=0, description="Debounce delay in seconds")
    archive_processed: bool = Field(default=True, description="Archive processed files")


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""

    enabled: bool = Field(default=True, description="Enable metrics collection")
    port: int = Field(default=9090, ge=1, le=65535, description="Metrics port")
    tracing_enabled: bool = Field(default=False, description="Enable distributed tracing")
    tracing_endpoint: Optional[str] = Field(
        default=None,
        description="Tracing endpoint (Jaeger/OTLP)"
    )


class SecurityConfig(BaseModel):
    """Security configuration."""

    secret_key: Optional[str] = Field(
        default=None,
        description="API secret key for authentication"
    )
    jwt_expiry_hours: int = Field(default=24, ge=1, description="JWT token expiry in hours")
    validate_paths: bool = Field(default=True, description="Enable path validation")
    max_file_size_mb: int = Field(default=100, ge=1, description="Maximum file size in MB")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        description="Log format string"
    )
    json_output: bool = Field(default=False, description="Output logs in JSON format")


# =============================================================================
# Main Settings Class
# =============================================================================

class Settings(BaseSettings):
    """
    Main application settings.

    Configuration is loaded from:
    1. Environment variables (HRM_* prefix)
    2. .env file
    3. YAML configuration files
    4. Default values

    Environment variables take precedence over file configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="HRM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Environment
    env: str = Field(
        default="development",
        description="Environment: development, staging, production"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Log level")

    # Nested configurations
    api: APIConfig = Field(default_factory=APIConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    drop_folder: DropFolderConfig = Field(default_factory=DropFolderConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # API Keys (loaded from environment only for security)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key"
    )
    pinecone_environment: Optional[str] = Field(
        default="us-west1-gcp",
        description="Pinecone environment"
    )
    wandb_api_key: Optional[str] = Field(
        default=None,
        description="Weights & Biases API key"
    )

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Settings":
        """
        Load settings from YAML file with environment variable overrides.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Settings instance
        """
        config_path = Path(config_path)

        if config_path.exists():
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            yaml_config = {}

        # Merge with environment variables
        return cls(**yaml_config)

    def get_device(self) -> str:
        """
        Get the compute device to use.

        Returns:
            Device string: cuda, mps, or cpu
        """
        import torch

        if self.model.device:
            return self.model.device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_model_path(self) -> Path:
        """
        Get the full model checkpoint path.

        Returns:
            Path to model checkpoint
        """
        return Path(self.model.model_path)

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env.lower() == "development"


# =============================================================================
# Global Configuration Access
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Settings are loaded once and cached for performance.
    Call `get_settings.cache_clear()` to reload.

    Returns:
        Settings instance
    """
    # Check for config file
    config_paths = [
        Path("config.yaml"),
        Path("config.yml"),
        Path("hrm_eval/configs/system_config.yaml"),
    ]

    for config_path in config_paths:
        if config_path.exists():
            logger.info(f"Loading configuration from: {config_path}")
            return Settings.from_yaml(config_path)

    logger.info("Loading configuration from environment variables and defaults")
    return Settings()


def reload_settings() -> Settings:
    """
    Reload settings (clears cache).

    Returns:
        Fresh Settings instance
    """
    get_settings.cache_clear()
    return get_settings()


# =============================================================================
# Environment-specific helpers
# =============================================================================

def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with HRM_ prefix.

    Args:
        key: Variable name (without HRM_ prefix)
        default: Default value if not set

    Returns:
        Environment variable value or default
    """
    return os.getenv(f"HRM_{key.upper()}", default)


def require_env_var(key: str) -> str:
    """
    Get required environment variable.

    Args:
        key: Variable name (without HRM_ prefix)

    Returns:
        Environment variable value

    Raises:
        ValueError: If variable not set
    """
    value = get_env_var(key)
    if value is None:
        raise ValueError(f"Required environment variable HRM_{key.upper()} is not set")
    return value


# =============================================================================
# Configuration validation
# =============================================================================

def validate_configuration(settings: Settings) -> List[str]:
    """
    Validate configuration and return list of warnings.

    Args:
        settings: Settings to validate

    Returns:
        List of warning messages
    """
    warnings = []

    # Check API keys for production
    if settings.is_production():
        if not settings.openai_api_key and not settings.anthropic_api_key:
            warnings.append("No LLM API key configured (OPENAI_API_KEY or ANTHROPIC_API_KEY)")

        if not settings.security.secret_key:
            warnings.append("No secret key configured for production (HRM_SECRET_KEY)")

    # Check model path exists
    model_path = settings.get_model_path()
    if not model_path.exists():
        warnings.append(f"Model checkpoint not found: {model_path}")

    # Check RAG backend configuration
    if settings.rag.backend == "pinecone" and not settings.pinecone_api_key:
        warnings.append("Pinecone backend selected but PINECONE_API_KEY not set")

    return warnings
