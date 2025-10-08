"""
Unit tests for unified configuration system.

Tests configuration loading, validation, overrides, and helper functions.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from hrm_eval.utils.unified_config import (
    SystemConfig,
    load_system_config,
    apply_overrides,
    load_env_overrides,
    get_checkpoint_path,
    create_output_directory,
    get_config_value,
)


class TestSystemConfigLoading:
    """Test configuration loading and validation."""
    
    def test_load_system_config_default(self):
        """Test loading default system configuration."""
        config = load_system_config()
        
        assert isinstance(config, SystemConfig)
        assert config.rag.top_k_retrieval == 5
        assert config.generation.batch_size == 8
        assert config.debug.enabled == False
    
    def test_config_validation_rag(self):
        """Test RAG configuration validation."""
        config = load_system_config()
        
        assert config.rag.top_k_retrieval >= 1
        assert 0.0 <= config.rag.similarity_threshold <= 1.0
        assert config.rag.backend in ["chromadb", "pinecone"]
    
    def test_config_validation_generation(self):
        """Test generation configuration validation."""
        config = load_system_config()
        
        assert config.generation.max_input_length >= 1
        assert 0.0 <= config.generation.temperature <= 2.0
        assert config.generation.batch_size >= 1
    
    def test_config_validation_security(self):
        """Test security configuration validation."""
        config = load_system_config()
        
        assert config.security.max_file_size_mb >= 1
        assert config.security.rate_limit_per_minute >= 1
        assert ".json" in config.security.allowed_extensions


class TestConfigOverrides:
    """Test configuration override mechanisms."""
    
    def test_apply_overrides_simple(self):
        """Test simple override application."""
        config = {"rag": {"top_k": 5}}
        overrides = {"rag.top_k": 10}
        
        updated = apply_overrides(config, overrides)
        
        assert updated["rag"]["top_k"] == 10
    
    def test_apply_overrides_nested(self):
        """Test nested override application."""
        config = {
            "rag": {
                "context_slicing": {
                    "acceptance_criteria_max": 3
                }
            }
        }
        overrides = {"rag.context_slicing.acceptance_criteria_max": 5}
        
        updated = apply_overrides(config, overrides)
        
        assert updated["rag"]["context_slicing"]["acceptance_criteria_max"] == 5
    
    def test_apply_overrides_creates_missing_keys(self):
        """Test override creates missing nested keys."""
        config = {}
        overrides = {"new.nested.key": "value"}
        
        updated = apply_overrides(config, overrides)
        
        assert updated["new"]["nested"]["key"] == "value"
    
    def test_load_env_overrides(self):
        """Test loading overrides from environment variables."""
        with patch.dict(os.environ, {
            "HRM_CONFIG_RAG__TOP_K": "10",
            "HRM_CONFIG_DEBUG__ENABLED": "true",
            "HRM_CONFIG_GENERATION__TEMPERATURE": "0.9"
        }):
            overrides = load_env_overrides()
            
            assert overrides["rag.top_k"] == 10
            assert overrides["debug.enabled"] == True
            assert overrides["generation.temperature"] == 0.9
    
    def test_env_overrides_type_conversion(self):
        """Test environment variable type conversion."""
        with patch.dict(os.environ, {
            "HRM_CONFIG_VALUE_INT": "42",
            "HRM_CONFIG_VALUE_FLOAT": "3.14",
            "HRM_CONFIG_VALUE_BOOL_TRUE": "true",
            "HRM_CONFIG_VALUE_BOOL_FALSE": "false",
            "HRM_CONFIG_VALUE_STRING": "hello"
        }):
            overrides = load_env_overrides()
            
            assert overrides["value_int"] == 42
            assert overrides["value_float"] == 3.14
            assert overrides["value_bool_true"] == True
            assert overrides["value_bool_false"] == False
            assert overrides["value_string"] == "hello"


class TestCheckpointPathResolution:
    """Test checkpoint path resolution."""
    
    def test_get_checkpoint_path_with_step(self):
        """Test checkpoint path resolution with step number."""
        config = load_system_config()
        
        path = get_checkpoint_path(config, "step_7566")
        
        assert "step_7566" in str(path)
        assert "checkpoints_hrm_v9_optimized" in str(path)
    
    def test_get_checkpoint_path_absolute(self):
        """Test checkpoint path with absolute path."""
        config = load_system_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            abs_path = Path(tmpdir) / "checkpoint.pt"
            abs_path.touch()
            
            path = get_checkpoint_path(config, str(abs_path))
            
            assert path == abs_path


class TestOutputDirectoryCreation:
    """Test output directory creation."""
    
    def test_create_output_directory_with_timestamp(self):
        """Test output directory creation with timestamp."""
        config = load_system_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_output_directory(
                config,
                "test_workflow",
                base_dir=Path(tmpdir)
            )
            
            assert output_dir.exists()
            assert output_dir.is_dir()
            assert "test_workflow" in output_dir.name
    
    def test_create_output_directory_without_timestamp(self):
        """Test output directory creation without timestamp."""
        config = load_system_config()
        config.output.use_timestamps = False
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_output_directory(
                config,
                "test_workflow",
                base_dir=Path(tmpdir)
            )
            
            assert output_dir.exists()
            assert output_dir.name == "test_workflow"


class TestConfigValueAccess:
    """Test configuration value access."""
    
    def test_get_config_value_simple(self):
        """Test getting simple config value."""
        config = load_system_config()
        
        value = get_config_value(config, "rag.top_k_retrieval")
        
        assert value == 5
    
    def test_get_config_value_nested(self):
        """Test getting nested config value."""
        config = load_system_config()
        
        value = get_config_value(config, "rag.context_slicing")
        
        assert isinstance(value, dict)
        assert "acceptance_criteria_max" in value
    
    def test_get_config_value_default(self):
        """Test getting config value with default."""
        config = load_system_config()
        
        value = get_config_value(config, "nonexistent.key", default="default_value")
        
        assert value == "default_value"
    
    def test_get_config_value_missing_no_default(self):
        """Test getting missing config value without default."""
        config = load_system_config()
        
        value = get_config_value(config, "nonexistent.key")
        
        assert value is None


class TestConfigPydanticValidation:
    """Test Pydantic validation in configuration."""
    
    def test_invalid_rag_top_k(self):
        """Test validation fails for invalid top_k."""
        from pydantic import ValidationError
        from hrm_eval.utils.unified_config import RAGConfig
        
        with pytest.raises(ValidationError):
            RAGConfig(top_k_retrieval=0)  # Should be >= 1
    
    def test_invalid_similarity_threshold(self):
        """Test validation fails for invalid similarity threshold."""
        from pydantic import ValidationError
        from hrm_eval.utils.unified_config import RAGConfig
        
        with pytest.raises(ValidationError):
            RAGConfig(similarity_threshold=1.5)  # Should be <= 1.0
    
    def test_invalid_backend(self):
        """Test validation fails for invalid backend."""
        from pydantic import ValidationError
        from hrm_eval.utils.unified_config import RAGConfig
        
        with pytest.raises(ValidationError):
            RAGConfig(backend="invalid")  # Should be chromadb or pinecone
    
    def test_invalid_batch_size(self):
        """Test validation fails for invalid batch size."""
        from pydantic import ValidationError
        from hrm_eval.utils.unified_config import GenerationConfig
        
        with pytest.raises(ValidationError):
            GenerationConfig(batch_size=0)  # Should be >= 1


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_load_with_overrides(self):
        """Test loading config with overrides."""
        overrides = {
            "rag.top_k_retrieval": 10,
            "debug.enabled": True,
        }
        
        config = load_system_config(overrides=overrides)
        
        assert config.rag.top_k_retrieval == 10
        assert config.debug.enabled == True
    
    def test_config_immutability_attempt(self):
        """Test config validation on assignment."""
        config = load_system_config()
        
        config.rag.top_k_retrieval = 15
        assert config.rag.top_k_retrieval == 15
    
    def test_all_config_sections_present(self):
        """Test all config sections are loaded."""
        config = load_system_config()
        
        assert hasattr(config, 'paths')
        assert hasattr(config, 'rag')
        assert hasattr(config, 'generation')
        assert hasattr(config, 'fine_tuning')
        assert hasattr(config, 'output')
        assert hasattr(config, 'security')
        assert hasattr(config, 'debug')
        assert hasattr(config, 'logging')
        assert hasattr(config, 'device')
        assert hasattr(config, 'model')
        assert hasattr(config, 'workflow')
        assert hasattr(config, 'features')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

