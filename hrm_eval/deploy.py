"""
Main deployment and evaluation script for HRM v9 Optimized.

Usage:
    python deploy.py --mode evaluate --checkpoint step_7566
    python deploy.py --mode ensemble --checkpoints step_7566 step_3607
    python deploy.py --mode test --checkpoint step_7566
"""

import argparse
import torch
from pathlib import Path
import sys
import logging

from models import HRMModel, HRMConfig
from evaluation import Evaluator
from ensemble import EnsembleModel, EnsembleStrategy
from data import PuzzleDataset, create_dataloader
from utils import load_checkpoint, load_config, setup_logging


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy and evaluate HRM v9 Optimized model"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["evaluate", "ensemble", "test"],
        help="Deployment mode",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Primary checkpoint name (e.g., step_7566)",
    )
    
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="Multiple checkpoint names for ensemble",
    )
    
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration",
    )
    
    parser.add_argument(
        "--eval-config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to evaluation configuration",
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to evaluation dataset",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu/mps)",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: Path, config, device: torch.device) -> HRMModel:
    """
    Load HRM model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device to load model on
        
    Returns:
        Loaded HRMModel
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    hrm_config = HRMConfig.from_yaml_config(config)
    model = HRMModel(hrm_config)
    
    checkpoint = load_checkpoint(checkpoint_path, device=str(device))
    model.load_from_checkpoint(checkpoint)
    
    model.to(device)
    model.eval()
    
    param_counts = model.get_num_params()
    logger.info(f"Model loaded: {param_counts['total']:,} parameters")
    
    return model


def evaluate_single_model(args, config, device):
    """
    Evaluate a single model checkpoint.
    
    Args:
        args: Command line arguments
        config: Configuration object
        device: Device to use
    """
    logger.info("="*80)
    logger.info("SINGLE MODEL EVALUATION")
    logger.info("="*80)
    
    checkpoint_path = Path(config.checkpoint.base_dir) / f"checkpoints_hrm_v9_optimized_{args.checkpoint}"
    
    model = load_model(checkpoint_path, config, device)
    
    data_path = args.data_path if args.data_path else config.data.validation_set
    dataset = PuzzleDataset(
        data_path=Path(data_path),
        max_seq_len=512,
        vocab_size=config.model.vocab_size,
    )
    
    logger.info(f"Loaded dataset: {len(dataset)} examples")
    
    evaluator = Evaluator(model=model, device=device, config=config)
    
    results = evaluator.evaluate(
        dataset=dataset,
        save_trajectories=config.evaluation.save_trajectories,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results.save(output_dir / f"results_{args.checkpoint}.json")
    
    logger.info("="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Solve Rate: {results.aggregate_metrics['solve_rate']:.2%}")
    logger.info(f"Average Steps: {results.aggregate_metrics['average_steps']:.2f}")
    logger.info(f"Average Accuracy: {results.aggregate_metrics['average_accuracy']:.2%}")
    logger.info(f"Total Time: {results.total_time:.2f}s")
    logger.info("="*80)


def evaluate_ensemble(args, config, device):
    """
    Evaluate ensemble of multiple checkpoints.
    
    Args:
        args: Command line arguments
        config: Configuration object
        device: Device to use
    """
    logger.info("="*80)
    logger.info("ENSEMBLE MODEL EVALUATION")
    logger.info("="*80)
    
    if not args.checkpoints or len(args.checkpoints) < 2:
        raise ValueError("Ensemble mode requires at least 2 checkpoints")
    
    checkpoint_paths = [
        Path(config.checkpoint.base_dir) / f"checkpoints_hrm_v9_optimized_{ckpt}"
        for ckpt in args.checkpoints
    ]
    
    logger.info(f"Creating ensemble from {len(checkpoint_paths)} checkpoints")
    
    weights = config.ensemble.weights if config.ensemble.enabled else None
    if weights:
        weights = [weights.get(ckpt, 1.0 / len(args.checkpoints)) for ckpt in args.checkpoints]
    
    ensemble = EnsembleModel.create_from_checkpoints(
        checkpoint_paths=[str(p) for p in checkpoint_paths],
        config=config,
        device=device,
        strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
        weights=weights,
    )
    
    data_path = args.data_path if args.data_path else config.data.validation_set
    dataset = PuzzleDataset(
        data_path=Path(data_path),
        max_seq_len=512,
        vocab_size=config.model.vocab_size,
    )
    
    evaluator = Evaluator(model=ensemble, device=device, config=config)
    
    results = evaluator.evaluate(
        dataset=dataset,
        save_trajectories=config.evaluation.save_trajectories,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ensemble_name = "_".join(args.checkpoints)
    results.save(output_dir / f"results_ensemble_{ensemble_name}.json")
    
    logger.info("="*80)
    logger.info("ENSEMBLE EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Solve Rate: {results.aggregate_metrics['solve_rate']:.2%}")
    logger.info(f"Average Steps: {results.aggregate_metrics['average_steps']:.2f}")
    logger.info(f"Average Accuracy: {results.aggregate_metrics['average_accuracy']:.2%}")
    logger.info("="*80)


def run_tests():
    """Run test suite."""
    import pytest
    
    logger.info("Running test suite...")
    
    test_dir = Path(__file__).parent / "tests"
    exit_code = pytest.main([str(test_dir), "-v", "--tb=short"])
    
    if exit_code == 0:
        logger.info("All tests passed!")
    else:
        logger.error(f"Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)


def main():
    """Main entry point."""
    args = parse_args()
    
    setup_logging(
        level=args.log_level,
        log_dir="logs",
        console_output=True,
        file_output=True,
        json_format=False,
    )
    
    logger.info("HRM v9 Optimized Deployment System")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    
    if args.mode == "test":
        run_tests()
        return
    
    config = load_config(
        model_config_path=Path(args.model_config),
        eval_config_path=Path(args.eval_config),
    )
    
    device = torch.device(args.device)
    
    try:
        if args.mode == "evaluate":
            if not args.checkpoint:
                raise ValueError("--checkpoint required for evaluate mode")
            evaluate_single_model(args, config, device)
        
        elif args.mode == "ensemble":
            evaluate_ensemble(args, config, device)
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Deployment complete!")


if __name__ == "__main__":
    main()

