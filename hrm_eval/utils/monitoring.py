"""
Monitoring and metrics tracking for test generation.

Provides structured logging and performance metrics tracking.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class GenerationEvent:
    """Event for test case generation."""
    
    def __init__(
        self,
        event_type: str,
        epic_id: str,
        num_test_cases: int,
        generation_time: float,
        coverage: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize generation event.
        
        Args:
            event_type: Type of event (e.g., 'generation', 'refinement')
            epic_id: Epic identifier
            num_test_cases: Number of test cases generated
            generation_time: Time taken (seconds)
            coverage: Coverage percentage
            metadata: Additional metadata
        """
        self.event_type = event_type
        self.epic_id = epic_id
        self.num_test_cases = num_test_cases
        self.generation_time = generation_time
        self.coverage = coverage
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()


class ModelMetrics:
    """Metrics for model performance."""
    
    def __init__(
        self,
        solve_rate: float,
        accuracy: float,
        avg_generation_time: float,
        total_generations: int,
    ):
        """
        Initialize model metrics.
        
        Args:
            solve_rate: Percentage of successful generations
            accuracy: Accuracy of generated test cases
            avg_generation_time: Average generation time
            total_generations: Total number of generations
        """
        self.solve_rate = solve_rate
        self.accuracy = accuracy
        self.avg_generation_time = avg_generation_time
        self.total_generations = total_generations
        self.timestamp = datetime.now().isoformat()


class TestGenerationMonitor:
    """
    Monitor for test case generation processes.
    
    Tracks metrics, logs events, and provides performance monitoring.
    """
    
    def __init__(self, log_dir: str = "logs/monitoring"):
        """
        Initialize monitor.
        
        Args:
            log_dir: Directory for monitoring logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.events_log = self.log_dir / "events.jsonl"
        self.metrics_log = self.log_dir / "metrics.jsonl"
        
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.start_time = time.time()
        
        logger.info(f"TestGenerationMonitor initialized (log_dir={log_dir})")
    
    def log_generation_event(self, event: GenerationEvent):
        """
        Log a generation event.
        
        Args:
            event: Generation event to log
        """
        self.generation_count += 1
        self.total_generation_time += event.generation_time
        
        event_data = {
            "event_type": event.event_type,
            "epic_id": event.epic_id,
            "num_test_cases": event.num_test_cases,
            "generation_time": event.generation_time,
            "coverage": event.coverage,
            "metadata": event.metadata,
            "timestamp": event.timestamp,
        }
        
        with open(self.events_log, "a") as f:
            f.write(json.dumps(event_data) + "\n")
        
        logger.info(
            f"Generation event: {event.event_type} | epic={event.epic_id} | "
            f"cases={event.num_test_cases} | time={event.generation_time:.2f}s | "
            f"coverage={event.coverage:.1f}%"
        )
        
        if event.coverage < 80:
            logger.warning(f"Low coverage for epic {event.epic_id}: {event.coverage:.1f}%")
    
    def track_model_performance(self, metrics: ModelMetrics):
        """
        Track model performance metrics.
        
        Args:
            metrics: Model performance metrics
        """
        metrics_data = {
            "solve_rate": metrics.solve_rate,
            "accuracy": metrics.accuracy,
            "avg_generation_time": metrics.avg_generation_time,
            "total_generations": metrics.total_generations,
            "timestamp": metrics.timestamp,
        }
        
        with open(self.metrics_log, "a") as f:
            f.write(json.dumps(metrics_data) + "\n")
        
        logger.info(
            f"Model metrics: solve_rate={metrics.solve_rate:.2%} | "
            f"accuracy={metrics.accuracy:.2%} | "
            f"avg_time={metrics.avg_generation_time:.2f}s"
        )
        
        if metrics.solve_rate < 0.8:
            logger.warning(f"Low solve rate: {metrics.solve_rate:.2%}")
        
        if metrics.avg_generation_time > 10.0:
            logger.warning(f"High generation time: {metrics.avg_generation_time:.2f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get monitoring summary.
        
        Returns:
            Summary of monitored metrics
        """
        uptime = time.time() - self.start_time
        avg_time = (
            self.total_generation_time / self.generation_count
            if self.generation_count > 0 else 0.0
        )
        
        summary = {
            "uptime_seconds": uptime,
            "total_generations": self.generation_count,
            "avg_generation_time": avg_time,
            "events_log": str(self.events_log),
            "metrics_log": str(self.metrics_log),
        }
        
        return summary
    
    def alert_on_failure(self, error_message: str, context: Dict[str, Any]):
        """
        Alert on generation failure.
        
        Args:
            error_message: Error message
            context: Failure context
        """
        alert_data = {
            "alert_type": "generation_failure",
            "error_message": error_message,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        }
        
        alert_log = self.log_dir / "alerts.jsonl"
        with open(alert_log, "a") as f:
            f.write(json.dumps(alert_data) + "\n")
        
        logger.error(f"Generation failure alert: {error_message}", extra=context)

