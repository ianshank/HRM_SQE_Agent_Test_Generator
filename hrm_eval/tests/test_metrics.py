"""Unit tests for metrics computation."""

import pytest
from ..evaluation.metrics import MetricsCalculator, PuzzleMetrics


class TestPuzzleMetrics:
    """Tests for PuzzleMetrics dataclass."""
    
    def test_creation(self):
        """Test metrics creation."""
        metrics = PuzzleMetrics(
            puzzle_id=1,
            solved=True,
            num_steps=50,
            time_elapsed=1.5,
            accuracy=0.95,
        )
        
        assert metrics.puzzle_id == 1
        assert metrics.solved == True
        assert metrics.num_steps == 50
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PuzzleMetrics(
            puzzle_id=1,
            solved=True,
            num_steps=50,
            time_elapsed=1.5,
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["puzzle_id"] == 1
        assert metrics_dict["solved"] == True


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create metrics calculator."""
        return MetricsCalculator()
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample puzzle metrics."""
        return [
            PuzzleMetrics(
                puzzle_id=i,
                solved=i % 2 == 0,
                num_steps=10 + i * 5,
                time_elapsed=1.0 + i * 0.1,
                accuracy=0.8 + i * 0.02,
            )
            for i in range(10)
        ]
    
    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert len(calculator.puzzle_metrics) == 0
    
    def test_add_puzzle_result(self, calculator):
        """Test adding puzzle result."""
        metrics = PuzzleMetrics(
            puzzle_id=1,
            solved=True,
            num_steps=50,
            time_elapsed=1.5,
        )
        
        calculator.add_puzzle_result(metrics)
        
        assert len(calculator.puzzle_metrics) == 1
    
    def test_compute_aggregate_metrics(self, calculator, sample_metrics):
        """Test aggregate metrics computation."""
        for metrics in sample_metrics:
            calculator.add_puzzle_result(metrics)
        
        aggregate = calculator.compute_aggregate_metrics()
        
        assert "total_puzzles" in aggregate
        assert aggregate["total_puzzles"] == 10
        assert "solve_rate" in aggregate
        assert aggregate["solve_rate"] == 0.5  # 5 out of 10 solved
        assert "average_steps" in aggregate
        assert aggregate["average_steps"] > 0
    
    def test_reset(self, calculator, sample_metrics):
        """Test calculator reset."""
        for metrics in sample_metrics:
            calculator.add_puzzle_result(metrics)
        
        assert len(calculator.puzzle_metrics) > 0
        
        calculator.reset()
        
        assert len(calculator.puzzle_metrics) == 0

