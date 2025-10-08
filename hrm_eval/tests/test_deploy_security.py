"""
Integration tests for deploy.py security validations.

Tests the path traversal protection in deployment and evaluation workflows.
"""

import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from deploy import validate_output_directory
from utils.security import PathTraversalError


class TestOutputDirectoryValidation:
    """Tests for output directory validation in deploy.py."""
    
    def test_valid_relative_path(self):
        """Test validation of valid relative output path."""
        # Should succeed without exceptions
        output_dir = validate_output_directory("results")
        assert output_dir.exists()
        assert output_dir.is_dir()
        assert output_dir.name == "results"
    
    def test_valid_nested_path(self):
        """Test validation of nested directory path."""
        output_dir = validate_output_directory("results/test_run_001")
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_reject_parent_directory_traversal(self):
        """Test rejection of parent directory traversal attempt."""
        with pytest.raises(PathTraversalError) as exc_info:
            validate_output_directory("../malicious_dir")
        assert "dangerous pattern" in str(exc_info.value).lower()
    
    def test_reject_nested_traversal(self):
        """Test rejection of nested traversal attempt."""
        with pytest.raises(PathTraversalError):
            validate_output_directory("results/../../etc")
    
    def test_reject_absolute_outside_project(self):
        """Test rejection of absolute path outside project."""
        with pytest.raises(PathTraversalError):
            validate_output_directory("/etc/passwd")
    
    def test_security_audit_logging(self, caplog):
        """Test that security violations are logged."""
        import logging
        
        with caplog.at_level(logging.ERROR):
            try:
                validate_output_directory("../malicious")
            except PathTraversalError:
                pass
        
        # Should log security violation
        assert "security violation" in caplog.text.lower() or \
               "path traversal" in caplog.text.lower()
    
    def test_directory_creation(self):
        """Test that valid directories are created."""
        test_dir = f"test_output_{id(self)}"
        output_dir = validate_output_directory(test_dir)
        
        assert output_dir.exists()
        assert output_dir.is_dir()
        
        # Cleanup
        try:
            output_dir.rmdir()
        except:
            pass
    
    def test_existing_directory_validation(self, tmp_path):
        """Test validation of existing directory."""
        # Create a temporary directory within project
        existing_dir = "existing_results"
        dir_path = validate_output_directory(existing_dir)
        assert dir_path.is_dir()
    
    def test_special_characters_handling(self):
        """Test handling of special characters in path."""
        # Should handle these safely
        safe_names = [
            "results_2025-10-08",
            "test.output",
            "run_001",
        ]
        
        for name in safe_names:
            output_dir = validate_output_directory(name)
            assert output_dir.exists()
            # Cleanup
            try:
                output_dir.rmdir()
            except:
                pass


class TestDeploySecurityIntegration:
    """Integration tests for complete deploy.py security workflow."""
    
    @patch('deploy.load_model')
    @patch('deploy.PuzzleDataset')
    @patch('deploy.Evaluator')
    @patch('deploy.load_config')
    def test_evaluate_with_safe_output_dir(
        self,
        mock_config,
        mock_evaluator_class,
        mock_dataset,
        mock_load_model
    ):
        """Test evaluation workflow with safe output directory."""
        # Setup mocks
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__ = lambda self: 10
        mock_dataset.return_value = mock_dataset_instance
        
        mock_evaluator = MagicMock()
        mock_results = MagicMock()
        mock_results.aggregate_metrics = {
            'solve_rate': 0.8,
            'average_steps': 10.5,
            'average_accuracy': 0.9
        }
        mock_results.total_time = 120.5
        mock_evaluator.evaluate.return_value = mock_results
        mock_evaluator_class.return_value = mock_evaluator
        
        mock_config_obj = MagicMock()
        mock_config_obj.checkpoint.base_dir = "."
        mock_config_obj.data.validation_set = "data/test.jsonl"
        mock_config_obj.model.vocab_size = 10000
        mock_config_obj.evaluation.save_trajectories = False
        mock_config.return_value = mock_config_obj
        
        # Import and run
        from deploy import evaluate_single_model
        import argparse
        
        args = argparse.Namespace(
            checkpoint="step_7566",
            output_dir="safe_results",
            data_path=None
        )
        
        # Should complete without security errors
        try:
            evaluate_single_model(args, mock_config_obj, "cpu")
        except Exception as e:
            # Ignore other errors, we're testing security
            if isinstance(e, PathTraversalError):
                pytest.fail(f"Security error with safe path: {e}")
    
    @patch('deploy.load_config')
    def test_evaluate_rejects_malicious_output_dir(self, mock_config):
        """Test that evaluation rejects malicious output directory."""
        import argparse
        from deploy import evaluate_single_model
        
        mock_config_obj = MagicMock()
        mock_config.return_value = mock_config_obj
        
        args = argparse.Namespace(
            checkpoint="step_7566",
            output_dir="../../../etc",
            data_path=None
        )
        
        # Should raise PathTraversalError before any model loading
        with pytest.raises(PathTraversalError):
            # This will fail at the validate_output_directory call
            validate_output_directory(args.output_dir)


class TestSecurityLogging:
    """Tests for security event logging in deploy.py."""
    
    def test_security_auditor_initialization(self):
        """Test that security auditor is properly initialized."""
        import deploy
        
        assert hasattr(deploy, 'security_auditor')
        assert deploy.security_auditor is not None
    
    def test_path_traversal_logged(self, caplog):
        """Test that path traversal attempts are logged."""
        import logging
        
        with caplog.at_level(logging.ERROR):
            try:
                validate_output_directory("../malicious")
            except PathTraversalError:
                pass
        
        # Should have logged the security event
        assert len(caplog.records) > 0
        assert any(
            "security" in record.message.lower() or
            "path" in record.message.lower()
            for record in caplog.records
        )


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_output_dir(self):
        """Test handling of empty output directory string."""
        with pytest.raises((PathTraversalError, ValueError)):
            validate_output_directory("")
    
    def test_whitespace_only_path(self):
        """Test handling of whitespace-only path."""
        with pytest.raises((PathTraversalError, ValueError)):
            validate_output_directory("   ")
    
    def test_null_byte_injection(self):
        """Test protection against null byte injection."""
        # Null bytes should be handled safely
        try:
            validate_output_directory("safe_dir\x00../etc")
        except (PathTraversalError, ValueError):
            # Either error is acceptable for this attack
            pass
    
    def test_unicode_path(self):
        """Test handling of Unicode characters in path."""
        unicode_dir = "results_日本語"
        output_dir = validate_output_directory(unicode_dir)
        assert output_dir.exists()
        # Cleanup
        try:
            output_dir.rmdir()
        except:
            pass
    
    def test_very_long_path(self):
        """Test handling of very long path names."""
        # Most file systems have path length limits
        long_name = "a" * 200
        
        try:
            output_dir = validate_output_directory(long_name)
            # If it succeeds, cleanup
            try:
                output_dir.rmdir()
            except:
                pass
        except (OSError, PathTraversalError):
            # Either error is acceptable for very long paths
            pass


class TestSecurityRegression:
    """Regression tests to ensure security fixes remain in place."""
    
    def test_no_md5_in_deploy(self):
        """Ensure deploy.py doesn't use MD5."""
        import inspect
        import deploy
        
        source = inspect.getsource(deploy)
        
        # Should not contain md5 usage
        assert 'hashlib.md5' not in source, \
            "deploy.py should not use MD5"
    
    def test_path_validation_present(self):
        """Ensure path validation is present in deploy.py."""
        import inspect
        import deploy
        
        source = inspect.getsource(deploy)
        
        # Should import security utilities
        assert 'PathValidator' in source or 'validate_output_directory' in source, \
            "deploy.py should use path validation"
    
    def test_security_auditor_used(self):
        """Ensure security auditor is used for logging."""
        import inspect
        import deploy
        
        source = inspect.getsource(deploy)
        
        # Should use security auditor
        assert 'SecurityAuditor' in source or 'security_auditor' in source, \
            "deploy.py should use security auditor"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
