"""
Evaluate HRM model on converted SQE agent data.

Runs the HRM v9 Optimized model on the SQE test data and generates
comprehensive evaluation metrics.
"""

import sys
sys.path.insert(0, '.')

import torch
from pathlib import Path
import json
import logging
from datetime import datetime

from models import HRMModel, HRMConfig
from data import PuzzleDataset, create_dataloader
from evaluation import Evaluator, MetricsCalculator
from utils import load_checkpoint, load_config, setup_logging


def main():
    """Main evaluation function."""
    
    setup_logging(
        level="INFO",
        log_dir="logs",
        console_output=True,
        file_output=True,
        json_format=False,
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("HRM v9 Optimized - SQE Agent Data Evaluation")
    logger.info("="*80)
    
    config = load_config(
        model_config_path=Path("configs/model_config.yaml"),
        eval_config_path=Path("configs/eval_config.yaml"),
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("\nLoading checkpoint step_7566 (best performing)...")
    checkpoint_path = Path("../checkpoints_hrm_v9_optimized_step_7566")
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Please ensure checkpoints are in the parent directory")
        return
    
    hrm_config = HRMConfig.from_yaml_config(config)
    model = HRMModel(hrm_config)
    
    checkpoint = load_checkpoint(checkpoint_path, device=str(device))
    model.load_from_checkpoint(checkpoint)
    
    model.to(device)
    model.eval()
    
    param_counts = model.get_num_params()
    logger.info(f"Model loaded: {param_counts['total']:,} parameters")
    
    logger.info("\nLoading SQE agent dataset...")
    data_path = Path("data/sqe_agent_hrm_format.jsonl")
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run convert_sqe_data.py first")
        return
    
    dataset = PuzzleDataset(
        data_path=data_path,
        max_seq_len=512,
        vocab_size=config.model.vocab_size,
    )
    
    logger.info(f"Loaded {len(dataset)} SQE test examples")
    
    logger.info("\nStarting evaluation...")
    logger.info("-" * 80)
    
    evaluator = Evaluator(model=model, device=device, config=config)
    
    results = evaluator.evaluate(
        dataset=dataset,
        save_trajectories=False,  # Set to True for detailed analysis
    )
    
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"sqe_evaluation_{timestamp}.json"
    results.save(results_file)
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    metrics = results.aggregate_metrics
    
    logger.info(f"\nDataset: SQE Agent Test Data")
    logger.info(f"Model: HRM v9 Optimized (step_7566)")
    logger.info(f"Total Puzzles: {metrics['total_puzzles']}")
    logger.info(f"Evaluation Time: {results.total_time:.2f}s")
    
    logger.info(f"\nPERFORMANCE METRICS:")
    logger.info(f"  Solve Rate:        {metrics['solve_rate']:.2%}")
    logger.info(f"  Average Steps:     {metrics['average_steps']:.2f}")
    logger.info(f"  Median Steps:      {metrics['median_steps']:.0f}")
    logger.info(f"  Min Steps:         {metrics['min_steps']:.0f}")
    logger.info(f"  Max Steps:         {metrics['max_steps']:.0f}")
    
    logger.info(f"\nACCURACY METRICS:")
    logger.info(f"  Average Accuracy:  {metrics['average_accuracy']:.2%}")
    logger.info(f"  Median Accuracy:   {metrics['median_accuracy']:.2%}")
    logger.info(f"  Overall Accuracy:  {metrics['overall_accuracy']:.2%}")
    
    logger.info(f"\nTIME METRICS:")
    logger.info(f"  Average Time:      {metrics['average_time']:.3f}s")
    logger.info(f"  Median Time:       {metrics['median_time']:.3f}s")
    logger.info(f"  Total Time:        {metrics['total_time']:.2f}s")
    
    if metrics['solved_puzzles'] > 0:
        logger.info(f"\nSOLVED PUZZLE METRICS:")
        logger.info(f"  Solved Count:      {metrics['solved_puzzles']}")
        logger.info(f"  Avg Steps (Solved): {metrics['average_steps_solved']:.2f}")
        logger.info(f"  Avg Time (Solved):  {metrics['average_time_solved']:.3f}s")
        logger.info(f"  Step Efficiency:    {metrics['step_efficiency']:.2%}")
    
    logger.info(f"\nREINFORCEMENT LEARNING METRICS:")
    logger.info(f"  Average Q-Value:   {metrics['average_q_value']:.4f}")
    logger.info(f"  Q-Value Std Dev:   {metrics['std_q_value']:.4f}")
    
    if metrics['error_rate'] > 0:
        logger.info(f"\nERROR METRICS:")
        logger.info(f"  Error Rate:        {metrics['error_rate']:.2%}")
    
    logger.info("\n" + "="*80)
    logger.info(f"Results saved to: {results_file}")
    logger.info("="*80)
    
    logger.info("\nDETAILED RESULTS BY DIFFICULTY:")
    
    per_puzzle = results.per_puzzle_metrics
    by_difficulty = {}
    
    for puzzle in per_puzzle:
        difficulty = "unknown"
        if puzzle.get("metadata"):
            difficulty = puzzle["metadata"].get("difficulty", "unknown")
        
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {"solved": 0, "total": 0, "avg_steps": []}
        
        by_difficulty[difficulty]["total"] += 1
        if puzzle["solved"]:
            by_difficulty[difficulty]["solved"] += 1
        by_difficulty[difficulty]["avg_steps"].append(puzzle["num_steps"])
    
    for difficulty, stats in sorted(by_difficulty.items()):
        solve_rate = stats["solved"] / stats["total"] if stats["total"] > 0 else 0
        avg_steps = sum(stats["avg_steps"]) / len(stats["avg_steps"]) if stats["avg_steps"] else 0
        
        logger.info(f"\n  {difficulty.upper()}:")
        logger.info(f"    Count: {stats['total']}")
        logger.info(f"    Solve Rate: {solve_rate:.2%}")
        logger.info(f"    Avg Steps: {avg_steps:.2f}")
    
    logger.info("\n" + "="*80)
    logger.info("Evaluation complete!")
    logger.info("="*80)
    
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review detailed results: {results_file}")
    logger.info(f"  2. Analyze failed puzzles for patterns")
    logger.info(f"  3. Consider ensemble evaluation: step_7566 + step_3607")
    logger.info(f"  4. Tune evaluation parameters if needed")


if __name__ == "__main__":
    main()

