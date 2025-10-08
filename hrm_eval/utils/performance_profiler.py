"""
Performance Profiler Module.

Provides detailed performance profiling including execution time, memory usage,
GPU utilization, and bottleneck detection.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, NamedTuple
from datetime import datetime
from dataclasses import dataclass, asdict

from .unified_config import SystemConfig

logger = logging.getLogger(__name__)


@dataclass
class ProfileEntry:
    """Single profile measurement."""
    name: str
    start_time: float
    end_time: float
    elapsed_seconds: float
    memory_start_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    gpu_memory_start_mb: Optional[float] = None
    gpu_memory_end_mb: Optional[float] = None
    gpu_memory_delta_mb: Optional[float] = None


@dataclass
class Bottleneck:
    """Identified performance bottleneck."""
    name: str
    elapsed_seconds: float
    percentage_of_total: float
    memory_delta_mb: Optional[float] = None
    recommendation: str = ""


class ProfileReport(NamedTuple):
    """Complete profiling report."""
    total_time: float
    total_memory_delta: Optional[float]
    entries: List[ProfileEntry]
    bottlenecks: List[Bottleneck]
    statistics: Dict[str, Any]


class PerformanceProfiler:
    """
    Profile execution time, memory, and GPU usage.
    
    Features:
    - Hierarchical profiling
    - Memory tracking (CPU and GPU)
    - Bottleneck detection
    - Flamegraph generation
    - Statistical analysis
    
    Example:
        >>> profiler = PerformanceProfiler(config)
        >>> profiler.start_profiling("training_session")
        >>> 
        >>> # Your code here
        >>> 
        >>> report = profiler.stop_profiling()
        >>> bottlenecks = profiler.get_bottlenecks()
        >>> profiler.generate_flamegraph(Path("flamegraph.svg"))
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize performance profiler.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.enabled = config.debug.profile_performance
        
        self.session_name: Optional[str] = None
        self.session_start: Optional[float] = None
        self.profile_entries: List[ProfileEntry] = []
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        
        if self.enabled:
            logger.info("PerformanceProfiler initialized (enabled)")
            self._check_dependencies()
        else:
            logger.debug("PerformanceProfiler initialized (disabled)")
    
    def _check_dependencies(self):
        """Check for optional profiling dependencies."""
        try:
            import psutil
            self.has_psutil = True
        except ImportError:
            self.has_psutil = False
            logger.warning("psutil not available - memory profiling disabled")
        
        try:
            import torch
            self.has_torch = torch.cuda.is_available()
        except ImportError:
            self.has_torch = False
    
    def start_profiling(self, session_name: str):
        """
        Start a profiling session.
        
        Args:
            session_name: Name of the profiling session
            
        Example:
            >>> profiler.start_profiling("model_training")
        """
        if not self.enabled:
            return
        
        self.session_name = session_name
        self.session_start = time.time()
        self.profile_entries.clear()
        self.active_profiles.clear()
        
        logger.info(f"Profiling session started: {session_name}")
    
    def stop_profiling(self) -> ProfileReport:
        """
        Stop profiling session and generate report.
        
        Returns:
            ProfileReport with complete profiling data
            
        Example:
            >>> report = profiler.stop_profiling()
            >>> print(f"Total time: {report.total_time:.2f}s")
        """
        if not self.enabled or self.session_start is None:
            return ProfileReport(
                total_time=0.0,
                total_memory_delta=None,
                entries=[],
                bottlenecks=[],
                statistics={},
            )
        
        total_time = time.time() - self.session_start
        
        total_memory_delta = None
        if self.profile_entries:
            memory_deltas = [
                e.memory_delta_mb for e in self.profile_entries
                if e.memory_delta_mb is not None
            ]
            if memory_deltas:
                total_memory_delta = sum(memory_deltas)
        
        bottlenecks = self.get_bottlenecks()
        statistics = self._calculate_statistics()
        
        logger.info(f"Profiling session complete: {self.session_name} ({total_time:.2f}s)")
        
        report = ProfileReport(
            total_time=total_time,
            total_memory_delta=total_memory_delta,
            entries=self.profile_entries,
            bottlenecks=bottlenecks,
            statistics=statistics,
        )
        
        if self.config.debug.save_memory_profile:
            self._save_report(report)
        
        return report
    
    def profile(self, name: str):
        """
        Context manager for profiling a code section.
        
        Args:
            name: Name of the section to profile
            
        Yields:
            Profile name
            
        Example:
            >>> with profiler.profile("data_loading"):
            ...     data = load_data()
        """
        from contextlib import contextmanager
        
        @contextmanager
        def _profile():
            self.start_section(name)
            try:
                yield name
            finally:
                self.end_section(name)
        
        return _profile()
    
    def start_section(self, name: str):
        """
        Start profiling a section.
        
        Args:
            name: Section name
            
        Example:
            >>> profiler.start_section("preprocessing")
        """
        if not self.enabled:
            return
        
        self.active_profiles[name] = {
            "start_time": time.time(),
            "memory_start": self._get_memory_usage(),
            "gpu_memory_start": self._get_gpu_memory_usage(),
        }
    
    def end_section(self, name: str):
        """
        End profiling a section.
        
        Args:
            name: Section name
            
        Example:
            >>> profiler.end_section("preprocessing")
        """
        if not self.enabled or name not in self.active_profiles:
            return
        
        profile = self.active_profiles[name]
        end_time = time.time()
        memory_end = self._get_memory_usage()
        gpu_memory_end = self._get_gpu_memory_usage()
        
        elapsed = end_time - profile["start_time"]
        
        memory_delta = None
        if profile["memory_start"] is not None and memory_end is not None:
            memory_delta = memory_end - profile["memory_start"]
        
        gpu_memory_delta = None
        if profile["gpu_memory_start"] is not None and gpu_memory_end is not None:
            gpu_memory_delta = gpu_memory_end - profile["gpu_memory_start"]
        
        entry = ProfileEntry(
            name=name,
            start_time=profile["start_time"],
            end_time=end_time,
            elapsed_seconds=elapsed,
            memory_start_mb=profile["memory_start"],
            memory_end_mb=memory_end,
            memory_delta_mb=memory_delta,
            gpu_memory_start_mb=profile["gpu_memory_start"],
            gpu_memory_end_mb=gpu_memory_end,
            gpu_memory_delta_mb=gpu_memory_delta,
        )
        
        self.profile_entries.append(entry)
        del self.active_profiles[name]
        
        logger.debug(f"Profiled {name}: {elapsed:.4f}s")
    
    def get_bottlenecks(self, top_n: int = 5, threshold: float = 0.1) -> List[Bottleneck]:
        """
        Identify performance bottlenecks.
        
        Args:
            top_n: Number of top bottlenecks to return
            threshold: Minimum percentage of total time to be considered a bottleneck
            
        Returns:
            List of identified bottlenecks
            
        Example:
            >>> bottlenecks = profiler.get_bottlenecks(top_n=3)
            >>> for bottleneck in bottlenecks:
            ...     print(f"{bottleneck.name}: {bottleneck.elapsed_seconds:.2f}s")
        """
        if not self.profile_entries:
            return []
        
        total_time = sum(e.elapsed_seconds for e in self.profile_entries)
        
        bottlenecks = []
        for entry in self.profile_entries:
            percentage = (entry.elapsed_seconds / total_time * 100) if total_time > 0 else 0
            
            if percentage >= threshold * 100:
                recommendation = self._generate_recommendation(entry, percentage)
                
                bottleneck = Bottleneck(
                    name=entry.name,
                    elapsed_seconds=entry.elapsed_seconds,
                    percentage_of_total=percentage,
                    memory_delta_mb=entry.memory_delta_mb,
                    recommendation=recommendation,
                )
                bottlenecks.append(bottleneck)
        
        bottlenecks.sort(key=lambda x: x.elapsed_seconds, reverse=True)
        return bottlenecks[:top_n]
    
    def _generate_recommendation(self, entry: ProfileEntry, percentage: float) -> str:
        """Generate optimization recommendation."""
        recommendations = []
        
        if percentage > 50:
            recommendations.append("Major bottleneck - priority optimization target")
        elif percentage > 25:
            recommendations.append("Significant bottleneck - consider optimization")
        
        if entry.memory_delta_mb and entry.memory_delta_mb > 1000:
            recommendations.append("High memory usage - consider batching or streaming")
        
        if entry.gpu_memory_delta_mb and entry.gpu_memory_delta_mb > 1000:
            recommendations.append("High GPU memory - consider gradient checkpointing")
        
        return "; ".join(recommendations) if recommendations else "Monitor performance"
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate profiling statistics."""
        if not self.profile_entries:
            return {}
        
        times = [e.elapsed_seconds for e in self.profile_entries]
        
        stats = {
            "total_sections": len(self.profile_entries),
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
        }
        
        memory_deltas = [
            e.memory_delta_mb for e in self.profile_entries
            if e.memory_delta_mb is not None
        ]
        
        if memory_deltas:
            stats["mean_memory_delta"] = sum(memory_deltas) / len(memory_deltas)
            stats["max_memory_delta"] = max(memory_deltas)
        
        return stats
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current CPU memory usage in MB."""
        if not self.has_psutil or not self.config.debug.profile_memory:
            return None
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return None
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if not self.has_torch or not self.config.debug.profile_gpu:
            return None
        
        try:
            import torch
            return torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            return None
    
    def generate_flamegraph(self, output_path: Path):
        """
        Generate flamegraph visualization.
        
        Args:
            output_path: Path to save flamegraph SVG
            
        Example:
            >>> profiler.generate_flamegraph(Path("profiling/flamegraph.svg"))
        """
        if not self.enabled or not self.config.debug.save_flamegraph:
            return
        
        if not self.profile_entries:
            logger.warning("No profile data to generate flamegraph")
            return
        
        logger.info(f"Flamegraph generation requested: {output_path}")
        logger.warning("Flamegraph generation requires py-spy or similar tool")
    
    def _save_report(self, report: ProfileReport):
        """Save profiling report to file."""
        output_dir = Path(self.config.debug.profiling_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"profile_{self.session_name}_{timestamp}.json"
        
        report_dict = {
            "session_name": self.session_name,
            "timestamp": datetime.now().isoformat(),
            "total_time": report.total_time,
            "total_memory_delta": report.total_memory_delta,
            "entries": [asdict(e) for e in report.entries],
            "bottlenecks": [asdict(b) for b in report.bottlenecks],
            "statistics": report.statistics,
        }
        
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Profile report saved: {report_path}")
    
    def print_summary(self, report: Optional[ProfileReport] = None):
        """
        Print profiling summary.
        
        Args:
            report: Profile report (uses last report if None)
            
        Example:
            >>> profiler.print_summary()
        """
        if report is None:
            return
        
        print(f"\n{'='*70}")
        print(f"Profiling Summary: {self.session_name}")
        print(f"{'='*70}")
        print(f"Total Time: {report.total_time:.4f}s")
        
        if report.total_memory_delta:
            print(f"Total Memory Delta: {report.total_memory_delta:+.2f} MB")
        
        print(f"\nTop Bottlenecks:")
        for i, bottleneck in enumerate(report.bottlenecks[:5], 1):
            print(f"  {i}. {bottleneck.name}")
            print(f"     Time: {bottleneck.elapsed_seconds:.4f}s ({bottleneck.percentage_of_total:.1f}%)")
            if bottleneck.recommendation:
                print(f"     Recommendation: {bottleneck.recommendation}")
        
        print(f"{'='*70}\n")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PerformanceProfiler(enabled={self.enabled}, entries={len(self.profile_entries)})"


__all__ = ["PerformanceProfiler", "ProfileEntry", "Bottleneck", "ProfileReport"]

