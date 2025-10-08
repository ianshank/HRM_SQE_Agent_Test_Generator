"""
Evaluate fine-tuned HRM model against base model.

Compares test generation quality between base and fine-tuned models on validation data.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from hrm_eval.models import HRMModel
from hrm_eval.models.hrm_model import HRMConfig
from hrm_eval.utils.config_utils import load_config
from hrm_eval.utils.checkpoint_utils import load_checkpoint
from hrm_eval.data.dataset import PuzzleDataset
from torch.utils.data import DataLoader
from hrm_eval.data.dataset import collate_fn
from hrm_eval.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def evaluate_model_perplexity(
    model: HRMModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model perplexity on dataset.
    
    Args:
        model: HRM model
        dataloader: Data loader
        device: Computation device
        
    Returns:
        Dictionary with metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            puzzle_ids = batch["puzzle_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, puzzle_ids=puzzle_ids)
            lm_logits = outputs["lm_logits"]
            
            # Calculate loss only on non-padded tokens
            mask = attention_mask.bool()
            loss = criterion(
                lm_logits[mask].reshape(-1, lm_logits.size(-1)),
                target_ids[mask].reshape(-1)
            )
            
            total_loss += loss.item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
    }


def compare_models(
    base_model: HRMModel,
    finetuned_model: HRMModel,
    val_dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Compare base and fine-tuned models.
    
    Args:
        base_model: Base HRM model
        finetuned_model: Fine-tuned model
        val_dataloader: Validation data loader
        device: Computation device
        
    Returns:
        Comparison results
    """
    logger.info("Evaluating base model...")
    base_metrics = evaluate_model_perplexity(base_model, val_dataloader, device)
    
    logger.info("Evaluating fine-tuned model...")
    finetuned_metrics = evaluate_model_perplexity(finetuned_model, val_dataloader, device)
    
    # Calculate improvements
    loss_improvement = ((base_metrics["loss"] - finetuned_metrics["loss"]) / 
                       base_metrics["loss"] * 100)
    perplexity_improvement = ((base_metrics["perplexity"] - finetuned_metrics["perplexity"]) /
                             base_metrics["perplexity"] * 100)
    
    comparison = {
        "base_model": base_metrics,
        "finetuned_model": finetuned_metrics,
        "improvements": {
            "loss_reduction_percent": loss_improvement,
            "perplexity_reduction_percent": perplexity_improvement,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    return comparison


def run_evaluation():
    """Run full evaluation workflow."""
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("Fine-Tuned Model Evaluation")
    logger.info("=" * 80)
    
    base_path = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    config = load_config(
        model_config_path=base_path / "configs" / "model_config.yaml",
        eval_config_path=base_path / "configs" / "eval_config.yaml"
    )
    hrm_config = HRMConfig.from_yaml_config(config)
    
    # Load base model
    logger.info("\n" + "=" * 80)
    logger.info("Loading Base Model")
    logger.info("=" * 80)
    
    base_checkpoint_path = base_path.parent / "checkpoints_hrm_v9_optimized_step_7566"
    base_model = HRMModel(hrm_config)
    base_checkpoint = load_checkpoint(str(base_checkpoint_path))
    
    # Handle checkpoint format
    if "model_state_dict" in base_checkpoint:
        state_dict = base_checkpoint["model_state_dict"]
    elif "state_dict" in base_checkpoint:
        state_dict = base_checkpoint["state_dict"]
    else:
        state_dict = base_checkpoint
    
    # Strip prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ["model.inner.", "model.", "module."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        if new_k == "embedding_weight":
            new_k = "weight"
        new_state_dict[new_k] = v
    
    base_model.load_state_dict(new_state_dict, strict=False)
    base_model.to(device)
    logger.info("Base model loaded successfully")
    
    # Load fine-tuned model
    logger.info("\n" + "=" * 80)
    logger.info("Loading Fine-Tuned Model")
    logger.info("=" * 80)
    
    finetuned_checkpoint_path = base_path / "fine_tuned_checkpoints" / "media_fulfillment" / "checkpoint_epoch_3_best.pt"
    finetuned_model = HRMModel(hrm_config)
    finetuned_checkpoint = load_checkpoint(str(finetuned_checkpoint_path))
    
    # Load state dict
    if "model_state_dict" in finetuned_checkpoint:
        finetuned_state_dict = finetuned_checkpoint["model_state_dict"]
        logger.info(f"Loaded model_state_dict with {len(finetuned_state_dict)} keys")
    elif "state_dict" in finetuned_checkpoint:
        finetuned_state_dict = finetuned_checkpoint["state_dict"]
        logger.info(f"Loaded state_dict with {len(finetuned_state_dict)} keys")
    else:
        finetuned_state_dict = finetuned_checkpoint
        logger.info(f"Using checkpoint directly with {len(finetuned_state_dict)} keys")
    
    # Load weights and verify
    missing_keys, unexpected_keys = finetuned_model.load_state_dict(finetuned_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    finetuned_model.to(device)
    logger.info(f"Fine-tuned model loaded successfully (epoch {finetuned_checkpoint.get('epoch', 'unknown')})")
    
    # Load validation data
    logger.info("\n" + "=" * 80)
    logger.info("Loading Validation Data")
    logger.info("=" * 80)
    
    val_data_path = base_path / "training_data" / "media_fulfillment_fine_tuning" / "validation_data.jsonl"
    val_dataset = PuzzleDataset(val_data_path)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    logger.info(f"Loaded {len(val_dataset)} validation examples")
    
    # Compare models
    logger.info("\n" + "=" * 80)
    logger.info("Comparing Models")
    logger.info("=" * 80)
    
    comparison_results = compare_models(
        base_model,
        finetuned_model,
        val_dataloader,
        device,
    )
    
    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    
    base_metrics = comparison_results["base_model"]
    ft_metrics = comparison_results["finetuned_model"]
    improvements = comparison_results["improvements"]
    
    logger.info(f"\nBase Model:")
    logger.info(f"  Loss: {base_metrics['loss']:.4f}")
    logger.info(f"  Perplexity: {base_metrics['perplexity']:.2f}")
    
    logger.info(f"\nFine-Tuned Model:")
    logger.info(f"  Loss: {ft_metrics['loss']:.4f}")
    logger.info(f"  Perplexity: {ft_metrics['perplexity']:.2f}")
    
    logger.info(f"\nImprovements:")
    logger.info(f"  Loss Reduction: {improvements['loss_reduction_percent']:.2f}%")
    logger.info(f"  Perplexity Reduction: {improvements['perplexity_reduction_percent']:.2f}%")
    
    # Save results
    output_dir = base_path / "fine_tuned_checkpoints" / "media_fulfillment"
    output_path = output_dir / "evaluation_results.json"
    
    with open(output_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"\nSaved evaluation results to {output_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
    
    return comparison_results


if __name__ == "__main__":
    run_evaluation()
