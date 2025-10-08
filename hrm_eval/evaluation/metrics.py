"""
Evaluation metrics for puzzle solving.

Provides comprehensive metrics including solve rate, accuracy,
step efficiency, and time-based measurements.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PuzzleMetrics:
    """Metrics for a single puzzle evaluation."""
    
    puzzle_id: int
    solved: bool
    num_steps: int
    time_elapsed: float
    accuracy: float = 0.0
    correct_actions: int = 0
    total_actions: int = 0
    final_q_value: float = 0.0
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "puzzle_id": self.puzzle_id,
            "solved": self.solved,
            "num_steps": self.num_steps,
            "time_elapsed": self.time_elapsed,
            "accuracy": self.accuracy,
            "correct_actions": self.correct_actions,
            "total_actions": self.total_actions,
            "final_q_value": self.final_q_value,
            "error_message": self.error_message,
        }


class MetricsCalculator:
    """
    Calculator for aggregate evaluation metrics.
    
    Computes various statistics from individual puzzle results including
    solve rate, average steps, success rate, and efficiency metrics.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.puzzle_metrics: List[PuzzleMetrics] = []
        logger.info("Initialized MetricsCalculator")
    
    def add_puzzle_result(self, metrics: PuzzleMetrics) -> None:
        """
        Add result for a single puzzle.
        
        Args:
            metrics: Metrics for the puzzle
        """
        self.puzzle_metrics.append(metrics)
    
    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """
        Compute aggregate metrics across all puzzles.
        
        Returns:
            Dictionary of aggregate metrics
        """
        if not self.puzzle_metrics:
            logger.warning("No puzzle metrics to compute")
            return {}
        
        total_puzzles = len(self.puzzle_metrics)
        solved_puzzles = sum(1 for m in self.puzzle_metrics if m.solved)
        
        metrics = {
            "total_puzzles": total_puzzles,
            "solved_puzzles": solved_puzzles,
            "solve_rate": solved_puzzles / total_puzzles if total_puzzles > 0 else 0.0,
        }
        
        steps = [m.num_steps for m in self.puzzle_metrics]
        metrics["average_steps"] = np.mean(steps)
        metrics["median_steps"] = np.median(steps)
        metrics["std_steps"] = np.std(steps)
        metrics["min_steps"] = np.min(steps)
        metrics["max_steps"] = np.max(steps)
        
        times = [m.time_elapsed for m in self.puzzle_metrics]
        metrics["average_time"] = np.mean(times)
        metrics["median_time"] = np.median(times)
        metrics["total_time"] = np.sum(times)
        
        solved_metrics = [m for m in self.puzzle_metrics if m.solved]
        if solved_metrics:
            solved_steps = [m.num_steps for m in solved_metrics]
            metrics["average_steps_solved"] = np.mean(solved_steps)
            metrics["median_steps_solved"] = np.median(solved_steps)
            
            solved_times = [m.time_elapsed for m in solved_metrics]
            metrics["average_time_solved"] = np.mean(solved_times)
        else:
            metrics["average_steps_solved"] = 0.0
            metrics["median_steps_solved"] = 0.0
            metrics["average_time_solved"] = 0.0
        
        accuracies = [m.accuracy for m in self.puzzle_metrics]
        metrics["average_accuracy"] = np.mean(accuracies)
        metrics["median_accuracy"] = np.median(accuracies)
        
        total_actions = sum(m.total_actions for m in self.puzzle_metrics)
        correct_actions = sum(m.correct_actions for m in self.puzzle_metrics)
        metrics["overall_accuracy"] = (
            correct_actions / total_actions if total_actions > 0 else 0.0
        )
        
        if solved_puzzles > 0:
            avg_steps_solved = metrics["average_steps_solved"]
            min_steps = metrics["min_steps"]
            metrics["step_efficiency"] = (
                min_steps / avg_steps_solved if avg_steps_solved > 0 else 0.0
            )
        else:
            metrics["step_efficiency"] = 0.0
        
        q_values = [m.final_q_value for m in self.puzzle_metrics]
        metrics["average_q_value"] = np.mean(q_values)
        metrics["std_q_value"] = np.std(q_values)
        
        errors = sum(1 for m in self.puzzle_metrics if m.error_message)
        metrics["error_rate"] = errors / total_puzzles if total_puzzles > 0 else 0.0
        
        logger.info(f"Computed aggregate metrics for {total_puzzles} puzzles")
        logger.info(f"Solve rate: {metrics['solve_rate']:.2%}")
        logger.info(f"Average steps: {metrics['average_steps']:.2f}")
        logger.info(f"Average accuracy: {metrics['average_accuracy']:.2%}")
        
        return metrics
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics with additional analysis.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        aggregate = self.compute_aggregate_metrics()
        
        summary = {
            "aggregate_metrics": aggregate,
            "per_puzzle_metrics": [m.to_dict() for m in self.puzzle_metrics],
        }
        
        if self.puzzle_metrics:
            solve_distribution = self._compute_solve_distribution()
            summary["solve_distribution"] = solve_distribution
            
            performance_by_steps = self._compute_performance_by_steps()
            summary["performance_by_steps"] = performance_by_steps
        
        return summary
    
    def _compute_solve_distribution(self) -> Dict[str, int]:
        """Compute distribution of solve outcomes."""
        distribution = {
            "solved": sum(1 for m in self.puzzle_metrics if m.solved),
            "unsolved": sum(1 for m in self.puzzle_metrics if not m.solved),
            "with_errors": sum(1 for m in self.puzzle_metrics if m.error_message),
        }
        return distribution
    
    def _compute_performance_by_steps(self) -> Dict[str, List[float]]:
        """Compute performance metrics binned by number of steps."""
        bins = [0, 10, 50, 100, 500, 1000, float('inf')]
        bin_labels = ['0-10', '10-50', '50-100', '100-500', '500-1000', '1000+']
        
        performance = {label: {'count': 0, 'solve_rate': 0.0} for label in bin_labels}
        
        for metrics in self.puzzle_metrics:
            for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
                if low <= metrics.num_steps < high:
                    label = bin_labels[i]
                    performance[label]['count'] += 1
                    if metrics.solved:
                        performance[label]['solve_rate'] += 1
                    break
        
        for label in bin_labels:
            count = performance[label]['count']
            if count > 0:
                performance[label]['solve_rate'] /= count
        
        return performance
    
    def reset(self) -> None:
        """Reset all stored metrics."""
        self.puzzle_metrics.clear()
        logger.info("Metrics calculator reset")

