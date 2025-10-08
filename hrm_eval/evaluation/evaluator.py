"""
Main evaluator for HRM models.

Orchestrates model evaluation with comprehensive metrics tracking,
logging, and result aggregation.
"""

import torch
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import logging
from tqdm import tqdm

from ..models import HRMModel
from ..data import PuzzleDataset, create_dataloader, PuzzleEnvironment
from .metrics import MetricsCalculator, PuzzleMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    aggregate_metrics: Dict[str, float]
    per_puzzle_metrics: list
    total_time: float
    config: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, output_path: Path) -> None:
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            "aggregate_metrics": self.aggregate_metrics,
            "per_puzzle_metrics": self.per_puzzle_metrics,
            "total_time": self.total_time,
            "config": self.config,
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


class Evaluator:
    """
    Main evaluator for HRM models.
    
    Handles complete evaluation pipeline including model inference,
    environment interaction, metrics computation, and result tracking.
    """
    
    def __init__(
        self,
        model: HRMModel,
        device: torch.device,
        config: Any,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: HRM model to evaluate
            device: Device for computation
            config: Evaluation configuration
        """
        self.model = model
        self.device = device
        self.config = config
        
        self.model.to(device)
        self.model.eval()
        
        self.metrics_calculator = MetricsCalculator()
        
        logger.info(f"Initialized Evaluator on device: {device}")
    
    @torch.no_grad()
    def evaluate(
        self,
        dataset: PuzzleDataset,
        save_trajectories: bool = False,
    ) -> EvaluationResults:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: Dataset to evaluate on
            save_trajectories: Whether to save full trajectories
            
        Returns:
            EvaluationResults object
        """
        logger.info(f"Starting evaluation on {len(dataset)} puzzles")
        start_time = time.time()
        
        self.metrics_calculator.reset()
        
        dataloader = create_dataloader(
            dataset,
            batch_size=self.config.evaluation.batch_size,
            num_workers=self.config.evaluation.num_workers,
            shuffle=False,
        )
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                batch_metrics = self._evaluate_batch(batch, save_trajectories)
                
                for metrics in batch_metrics:
                    self.metrics_calculator.add_puzzle_result(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating batch {batch_idx}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        aggregate_metrics = self.metrics_calculator.compute_aggregate_metrics()
        summary_stats = self.metrics_calculator.get_summary_stats()
        
        results = EvaluationResults(
            aggregate_metrics=aggregate_metrics,
            per_puzzle_metrics=summary_stats["per_puzzle_metrics"],
            total_time=total_time,
            config={
                "batch_size": self.config.evaluation.batch_size,
                "max_steps": self.config.evaluation.max_steps_per_puzzle,
                "device": str(self.device),
            },
        )
        
        logger.info(f"Evaluation complete in {total_time:.2f}s")
        logger.info(f"Solve rate: {aggregate_metrics['solve_rate']:.2%}")
        
        return results
    
    def _evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        save_trajectories: bool,
    ) -> list:
        """
        Evaluate a single batch.
        
        Args:
            batch: Batch of puzzles
            save_trajectories: Whether to save trajectories
            
        Returns:
            List of PuzzleMetrics for the batch
        """
        puzzle_ids = batch["puzzle_ids"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        batch_metrics = []
        
        for i in range(len(puzzle_ids)):
            puzzle_id = puzzle_ids[i].item()
            
            try:
                metrics = self._evaluate_single_puzzle(
                    puzzle_id=puzzle_id,
                    input_ids=input_ids[i:i+1],
                    target_ids=target_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    save_trajectory=save_trajectories,
                )
                batch_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating puzzle {puzzle_id}: {e}")
                
                error_metrics = PuzzleMetrics(
                    puzzle_id=puzzle_id,
                    solved=False,
                    num_steps=0,
                    time_elapsed=0.0,
                    error_message=str(e),
                )
                batch_metrics.append(error_metrics)
        
        return batch_metrics
    
    def _evaluate_single_puzzle(
        self,
        puzzle_id: int,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        save_trajectory: bool,
    ) -> PuzzleMetrics:
        """
        Evaluate model on a single puzzle.
        
        Args:
            puzzle_id: Puzzle ID
            input_ids: Input token IDs
            target_ids: Target token IDs
            attention_mask: Attention mask
            save_trajectory: Whether to save trajectory
            
        Returns:
            PuzzleMetrics for the puzzle
        """
        start_time = time.time()
        
        env = PuzzleEnvironment(
            puzzle_id=puzzle_id,
            max_steps=self.config.evaluation.max_steps_per_puzzle,
            timeout=self.config.evaluation.timeout_seconds,
        )
        
        env.reset()
        
        h_state = None
        l_state = None
        
        num_steps = 0
        correct_actions = 0
        total_actions = 0
        
        done = False
        while not done and num_steps < self.config.evaluation.max_steps_per_puzzle:
            outputs = self.model(
                input_ids=input_ids,
                puzzle_ids=puzzle_id,
                h_state=h_state,
                l_state=l_state,
                attention_mask=attention_mask,
                return_states=True,
            )
            
            q_values = outputs["q_values"]
            action = torch.argmax(q_values, dim=-1).item()
            
            _, reward, done, info = env.step(action)
            
            lm_logits = outputs["lm_logits"]
            predictions = torch.argmax(lm_logits, dim=-1)
            
            if num_steps < target_ids.size(1):
                target = target_ids[0, num_steps].item()
                pred = predictions[0, num_steps].item()
                if pred == target:
                    correct_actions += 1
                total_actions += 1
            
            h_state = outputs.get("h_state")
            l_state = outputs.get("l_state")
            
            num_steps += 1
        
        time_elapsed = time.time() - start_time
        
        accuracy = correct_actions / total_actions if total_actions > 0 else 0.0
        
        final_q_value = q_values.max().item()
        
        metrics = PuzzleMetrics(
            puzzle_id=puzzle_id,
            solved=env.solved,
            num_steps=num_steps,
            time_elapsed=time_elapsed,
            accuracy=accuracy,
            correct_actions=correct_actions,
            total_actions=total_actions,
            final_q_value=final_q_value,
            trajectory=env.get_trajectory() if save_trajectory else [],
        )
        
        return metrics

