"""
Unit tests for DebugManager.

Tests profiling, checkpoints, logging, and state management.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from hrm_eval.utils.debug_manager import DebugManager
from hrm_eval.utils.unified_config import load_system_config


class TestDebugManagerInitialization:
    """Test DebugManager initialization."""
    
    def test_init_disabled(self):
        """Test initialization when debug is disabled."""
        config = load_system_config()
        config.debug.enabled = False
        
        debug = DebugManager(config)
        
        assert debug.enabled == False
        assert debug.output_dir is None
    
    def test_init_enabled(self):
        """Test initialization when debug is enabled."""
        config = load_system_config()
        config.debug.enabled = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.debug.profiling_output_dir = tmpdir
            debug = DebugManager(config)
            
            assert debug.enabled == True
            assert debug.output_dir is not None
            assert debug.output_dir.exists()
    
    def test_init_data_structures(self):
        """Test initialization of data structures."""
        config = load_system_config()
        
        debug = DebugManager(config)
        
        assert isinstance(debug.profiling_data, dict)
        assert isinstance(debug.checkpoint_data, dict)
        assert isinstance(debug.model_io_log, list)


class TestProfilingSection:
    """Test profiling section functionality."""
    
    def test_profile_section_disabled(self):
        """Test profiling when disabled."""
        config = load_system_config()
        config.debug.enabled = False
        debug = DebugManager(config)
        
        with debug.profile_section("test_section"):
            time.sleep(0.01)
        
        assert "test_section" not in debug.profiling_data
    
    def test_profile_section_enabled(self):
        """Test profiling when enabled."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.profile_performance = True
        debug = DebugManager(config)
        
        with debug.profile_section("test_section"):
            time.sleep(0.01)
        
        assert "test_section" in debug.profiling_data
        assert debug.profiling_data["test_section"]["elapsed_seconds"] >= 0.01
    
    def test_profile_section_exception_handling(self):
        """Test profiling handles exceptions."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.profile_performance = True
        debug = DebugManager(config)
        
        with pytest.raises(ValueError):
            with debug.profile_section("test_section"):
                raise ValueError("Test error")
        
        assert "test_section" in debug.profiling_data
    
    @patch('hrm_eval.utils.debug_manager.DebugManager._get_memory_usage')
    def test_profile_section_with_memory(self, mock_memory):
        """Test profiling with memory tracking."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.profile_performance = True
        config.debug.profile_memory = True
        debug = DebugManager(config)
        
        mock_memory.side_effect = [100.0, 150.0]  # Start and end memory
        
        with debug.profile_section("test_section"):
            pass
        
        assert debug.profiling_data["test_section"]["memory_delta_mb"] == 50.0


class TestDebugCheckpoint:
    """Test debug checkpoint functionality."""
    
    def test_debug_checkpoint_disabled(self):
        """Test checkpoint when debug disabled."""
        config = load_system_config()
        config.debug.enabled = False
        debug = DebugManager(config)
        
        with debug.debug_checkpoint("test_checkpoint"):
            pass
        
        assert "test_checkpoint" not in debug.checkpoint_data
    
    def test_debug_checkpoint_not_configured(self):
        """Test checkpoint not in configured stages."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.checkpoint_stages = []
        debug = DebugManager(config)
        
        with debug.debug_checkpoint("test_checkpoint"):
            pass
        
        assert "test_checkpoint" not in debug.checkpoint_data
    
    def test_debug_checkpoint_configured(self):
        """Test checkpoint in configured stages."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.checkpoint_stages = ["test_checkpoint"]
        debug = DebugManager(config)
        
        with debug.debug_checkpoint("test_checkpoint"):
            pass
        
        assert "test_checkpoint" in debug.checkpoint_data
        assert "timestamp" in debug.checkpoint_data["test_checkpoint"]
    
    def test_debug_checkpoint_with_error(self):
        """Test checkpoint with error."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.checkpoint_stages = ["test_checkpoint"]
        config.debug.breakpoint_on_error = False
        debug = DebugManager(config)
        
        with pytest.raises(ValueError):
            with debug.debug_checkpoint("test_checkpoint"):
                raise ValueError("Test error")


class TestModelIOLogging:
    """Test model I/O logging."""
    
    def test_log_model_io_disabled(self):
        """Test I/O logging when disabled."""
        config = load_system_config()
        config.debug.enabled = False
        debug = DebugManager(config)
        
        debug.log_model_input_output(
            {"input": "test"},
            {"output": "test"}
        )
        
        assert len(debug.model_io_log) == 0
    
    def test_log_model_io_enabled(self):
        """Test I/O logging when enabled."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.log_model_inputs = True
        config.debug.log_model_outputs = True
        debug = DebugManager(config)
        
        import torch
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        outputs = {"logits": torch.tensor([[0.1, 0.9]])}
        
        debug.log_model_input_output(inputs, outputs, "TestModel")
        
        assert len(debug.model_io_log) == 1
        assert debug.model_io_log[0]["model_name"] == "TestModel"
        assert "inputs" in debug.model_io_log[0]
        assert "outputs" in debug.model_io_log[0]


class TestIntermediateStateDump:
    """Test intermediate state dumping."""
    
    def test_dump_state_disabled(self):
        """Test state dump when disabled."""
        config = load_system_config()
        config.debug.enabled = False
        debug = DebugManager(config)
        
        debug.dump_intermediate_state({"key": "value"}, "test_stage")
        
        assert "state_test_stage" not in debug.checkpoint_data
    
    def test_dump_state_enabled_no_file(self):
        """Test state dump without file saving."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.log_intermediate_states = True
        debug = DebugManager(config)
        
        debug.dump_intermediate_state(
            {"key": "value"},
            "test_stage",
            save_to_file=False
        )
        
        assert "state_test_stage" in debug.checkpoint_data
    
    def test_dump_state_with_file(self):
        """Test state dump with file saving."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.log_intermediate_states = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.debug.profiling_output_dir = tmpdir
            debug = DebugManager(config)
            
            debug.dump_intermediate_state(
                {"key": "value"},
                "test_stage",
                save_to_file=True
            )
            
            state_file = Path(tmpdir) / "state_test_stage.json"
            assert state_file.exists()


class TestPerformanceReport:
    """Test performance report generation."""
    
    def test_get_performance_report_empty(self):
        """Test report when no profiling data."""
        config = load_system_config()
        debug = DebugManager(config)
        
        report = debug.get_performance_report()
        
        assert "message" in report
    
    def test_get_performance_report_with_data(self):
        """Test report with profiling data."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.profile_performance = True
        debug = DebugManager(config)
        
        with debug.profile_section("section1"):
            time.sleep(0.01)
        
        with debug.profile_section("section2"):
            time.sleep(0.02)
        
        report = debug.get_performance_report()
        
        assert "total_time" in report
        assert report["total_time"] >= 0.03
        assert "sections" in report
        assert "slowest_sections" in report
        assert len(report["slowest_sections"]) > 0


class TestDebugManagerUtilities:
    """Test utility methods."""
    
    def test_enable_breakpoint_on_error(self):
        """Test enabling breakpoint on error."""
        config = load_system_config()
        config.debug.enabled = True
        debug = DebugManager(config)
        
        debug.enable_breakpoint_on_error(True)
        
        assert debug.config.debug.breakpoint_on_error == True
    
    def test_clear_debug_data(self):
        """Test clearing debug data."""
        config = load_system_config()
        config.debug.enabled = True
        debug = DebugManager(config)
        
        debug.profiling_data["test"] = {}
        debug.checkpoint_data["test"] = {}
        debug.model_io_log.append({})
        
        debug.clear()
        
        assert len(debug.profiling_data) == 0
        assert len(debug.checkpoint_data) == 0
        assert len(debug.model_io_log) == 0
    
    def test_save_report(self):
        """Test saving debug report."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.profile_performance = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.debug.profiling_output_dir = tmpdir
            debug = DebugManager(config)
            
            with debug.profile_section("test"):
                pass
            
            debug.save_report("test_report.json")
            
            report_file = Path(tmpdir) / "test_report.json"
            assert report_file.exists()


class TestDebugManagerRepr:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.profile_performance = True
        debug = DebugManager(config)
        
        with debug.profile_section("test"):
            pass
        
        repr_str = repr(debug)
        
        assert "DebugManager" in repr_str
        assert "enabled=True" in repr_str
        assert "sections_profiled=1" in repr_str


class TestDebugManagerIntegration:
    """Integration tests for DebugManager."""
    
    def test_complete_debug_workflow(self):
        """Test complete debugging workflow."""
        config = load_system_config()
        config.debug.enabled = True
        config.debug.profile_performance = True
        config.debug.profile_memory = True
        config.debug.log_intermediate_states = True
        config.debug.checkpoint_stages = ["stage1", "stage2"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.debug.profiling_output_dir = tmpdir
            debug = DebugManager(config)
            
            with debug.profile_section("data_loading"):
                time.sleep(0.01)
            
            with debug.debug_checkpoint("stage1"):
                debug.dump_intermediate_state({"data": [1, 2, 3]}, "after_loading")
            
            with debug.profile_section("processing"):
                time.sleep(0.01)
            
            with debug.debug_checkpoint("stage2"):
                pass
            
            report = debug.get_performance_report()
            
            assert len(debug.profiling_data) == 2
            assert len(debug.checkpoint_data) >= 2
            assert report["total_time"] >= 0.02
            
            debug.save_report()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

