"""
Debug Manager Module.

Provides centralized debugging and profiling utilities for development and
troubleshooting.
"""

import logging
import time
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from datetime import datetime

from .unified_config import SystemConfig

logger = logging.getLogger(__name__)


class DebugManager:
    """
    Centralized debugging and profiling utilities.
    
    Features:
    - Performance profiling with context managers
    - Debug checkpoints for state inspection
    - Model I/O logging
    - Intermediate state dumps
    - Conditional breakpoints
    - Performance reporting
    
    Example:
        >>> debug = DebugManager(config)
        >>> with debug.profile_section("data_loading"):
        ...     data = load_data()
        >>> 
        >>> debug.dump_intermediate_state({"data": data}, "after_loading")
        >>> report = debug.get_performance_report()
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize debug manager.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.enabled = config.debug.enabled
        self.verbose = config.debug.verbose
        
        self.profiling_data: Dict[str, Dict[str, Any]] = {}
        self.checkpoint_data: Dict[str, Any] = {}
        self.model_io_log: list = []
        
        if self.enabled:
            self.output_dir = Path(config.debug.profiling_output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"DebugManager initialized (enabled, output: {self.output_dir})")
        else:
            self.output_dir = None
            logger.debug("DebugManager initialized (disabled)")
    
    @contextmanager
    def profile_section(self, section_name: str):
        """
        Profile a code section.
        
        Args:
            section_name: Name of the section being profiled
            
        Yields:
            Section name
            
        Example:
            >>> with debug.profile_section("model_inference"):
            ...     output = model(input_data)
        """
        if not self.enabled or not self.config.debug.profile_performance:
            yield section_name
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        logger.debug(f"Profiling started: {section_name}")
        
        try:
            yield section_name
        
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            elapsed = end_time - start_time
            memory_delta = end_memory - start_memory if start_memory and end_memory else None
            
            self.profiling_data[section_name] = {
                "start_time": start_time,
                "end_time": end_time,
                "elapsed_seconds": elapsed,
                "start_memory_mb": start_memory,
                "end_memory_mb": end_memory,
                "memory_delta_mb": memory_delta,
                "timestamp": datetime.now().isoformat(),
            }
            
            logger.debug(f"Profiling complete: {section_name} ({elapsed:.2f}s)")
            
            if self.verbose:
                self._print_profile_summary(section_name)
    
    @contextmanager
    def debug_checkpoint(self, checkpoint_name: str):
        """
        Create a debug checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Yields:
            Checkpoint name
            
        Example:
            >>> with debug.debug_checkpoint("before_generation"):
            ...     # Code to debug
            ...     pass
        """
        if not self.enabled:
            yield checkpoint_name
            return
        
        if checkpoint_name not in self.config.debug.checkpoint_stages:
            yield checkpoint_name
            return
        
        logger.debug(f"Debug checkpoint: {checkpoint_name}")
        
        self.checkpoint_data[checkpoint_name] = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_name": checkpoint_name,
        }
        
        try:
            yield checkpoint_name
        
        except Exception as e:
            logger.error(f"Exception at checkpoint {checkpoint_name}: {e}")
            
            if self.config.debug.breakpoint_on_error:
                logger.warning("Breakpoint on error enabled - entering debugger")
                import pdb
                pdb.set_trace()
            
            raise
    
    def log_model_input_output(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        model_name: str = "model",
    ):
        """
        Log model inputs and outputs for debugging.
        
        Args:
            inputs: Input dictionary
            outputs: Output dictionary
            model_name: Name of the model
            
        Example:
            >>> debug.log_model_input_output(
            ...     {"input_ids": input_ids},
            ...     {"logits": logits},
            ...     "HRM"
            ... )
        """
        if not self.enabled:
            return
        
        if not self.config.debug.log_model_inputs and not self.config.debug.log_model_outputs:
            return
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
        }
        
        if self.config.debug.log_model_inputs:
            log_entry["inputs"] = self._summarize_tensor_dict(inputs)
        
        if self.config.debug.log_model_outputs:
            log_entry["outputs"] = self._summarize_tensor_dict(outputs)
        
        self.model_io_log.append(log_entry)
        
        if self.verbose:
            logger.debug(f"Model I/O logged for {model_name}")
    
    def dump_intermediate_state(
        self,
        state: Dict[str, Any],
        stage: str,
        save_to_file: bool = True,
    ):
        """
        Dump intermediate state for inspection.
        
        Args:
            state: State dictionary to dump
            stage: Name of the pipeline stage
            save_to_file: Save to file in addition to memory
            
        Example:
            >>> debug.dump_intermediate_state(
            ...     {"contexts": contexts, "examples": rag_examples},
            ...     "after_rag_retrieval"
            ... )
        """
        if not self.enabled or not self.config.debug.log_intermediate_states:
            return
        
        self.checkpoint_data[f"state_{stage}"] = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "state_summary": self._summarize_state(state),
        }
        
        if save_to_file and self.output_dir:
            state_file = self.output_dir / f"state_{stage}.json"
            
            try:
                with open(state_file, "w") as f:
                    json.dump(
                        self._make_json_serializable(state),
                        f,
                        indent=2,
                        default=str
                    )
                logger.debug(f"State dumped to {state_file}")
            
            except Exception as e:
                logger.error(f"Failed to dump state: {e}")
    
    def enable_breakpoint_on_error(self, enabled: bool = True):
        """
        Enable/disable automatic breakpoint on error.
        
        Args:
            enabled: Enable breakpoints
            
        Example:
            >>> debug.enable_breakpoint_on_error(True)
        """
        if self.enabled:
            self.config.debug.breakpoint_on_error = enabled
            logger.info(f"Breakpoint on error: {enabled}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.
        
        Returns:
            Performance report dictionary
            
        Example:
            >>> report = debug.get_performance_report()
            >>> print(f"Total time: {report['total_time']:.2f}s")
        """
        if not self.profiling_data:
            return {"message": "No profiling data available"}
        
        total_time = sum(
            data["elapsed_seconds"]
            for data in self.profiling_data.values()
        )
        
        report = {
            "total_time": total_time,
            "sections": self.profiling_data,
            "slowest_sections": self._get_slowest_sections(5),
            "memory_intensive_sections": self._get_memory_intensive_sections(5),
        }
        
        return report
    
    def _get_slowest_sections(self, top_n: int = 5) -> list:
        """Get slowest profiled sections."""
        sections = [
            {"name": name, "time": data["elapsed_seconds"]}
            for name, data in self.profiling_data.items()
        ]
        sections.sort(key=lambda x: x["time"], reverse=True)
        return sections[:top_n]
    
    def _get_memory_intensive_sections(self, top_n: int = 5) -> list:
        """Get most memory-intensive sections."""
        sections = [
            {"name": name, "memory_delta": data.get("memory_delta_mb", 0) or 0}
            for name, data in self.profiling_data.items()
        ]
        sections.sort(key=lambda x: x["memory_delta"], reverse=True)
        return sections[:top_n]
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if not self.config.debug.profile_memory:
            return None
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except (ImportError, Exception):
            return None
    
    def _summarize_tensor_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize dictionary containing tensors."""
        import torch
        
        summary = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                summary[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "requires_grad": value.requires_grad,
                }
            else:
                summary[key] = type(value).__name__
        
        return summary
    
    def _summarize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize state dictionary."""
        summary = {}
        
        for key, value in state.items():
            if isinstance(value, list):
                summary[key] = {
                    "type": "list",
                    "length": len(value),
                    "sample": str(value[0])[:100] if value else None,
                }
            elif isinstance(value, dict):
                summary[key] = {
                    "type": "dict",
                    "keys": list(value.keys())[:10],
                }
            else:
                summary[key] = {
                    "type": type(value).__name__,
                    "value": str(value)[:100],
                }
        
        return summary
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        import torch
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return {
                "tensor_shape": list(obj.shape),
                "tensor_dtype": str(obj.dtype),
            }
        elif hasattr(obj, 'dict'):
            return self._make_json_serializable(obj.dict())
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return str(obj)
    
    def _print_profile_summary(self, section_name: str):
        """Print profile summary for a section."""
        data = self.profiling_data[section_name]
        
        print(f"\n{'='*60}")
        print(f"Profile: {section_name}")
        print(f"{'='*60}")
        print(f"Time: {data['elapsed_seconds']:.4f}s")
        
        if data.get('memory_delta_mb') is not None:
            print(f"Memory: {data['memory_delta_mb']:+.2f} MB")
        
        print(f"{'='*60}\n")
    
    def save_report(self, filename: Optional[str] = None):
        """
        Save performance report to file.
        
        Args:
            filename: Output filename (auto-generated if None)
            
        Example:
            >>> debug.save_report("performance_report.json")
        """
        if not self.enabled or not self.output_dir:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"debug_report_{timestamp}.json"
        
        report_path = self.output_dir / filename
        
        report = {
            "performance": self.get_performance_report(),
            "checkpoints": self.checkpoint_data,
            "model_io_log": self.model_io_log,
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Debug report saved to {report_path}")
    
    def clear(self):
        """Clear all debug data."""
        self.profiling_data.clear()
        self.checkpoint_data.clear()
        self.model_io_log.clear()
        logger.debug("Debug data cleared")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DebugManager(enabled={self.enabled}, sections_profiled={len(self.profiling_data)})"


__all__ = ["DebugManager"]

