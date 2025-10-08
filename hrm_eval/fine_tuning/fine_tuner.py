"""
HRM model fine-tuner for requirements-to-test-cases task.

Fine-tunes the HRM v9 Optimized model on collected training data
to improve test case generation quality.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time
from tqdm import tqdm

from ..models import HRMModel
from ..data.dataset import PuzzleDataset, collate_fn

logger = logging.getLogger(__name__)


class FineTuningConfig:
    """Configuration for fine-tuning."""
    
    def __init__(
        self,
        learning_rate: float = 1e-5,
        epochs: int = 3,
        batch_size: int = 16,
        validation_split: float = 0.2,
        warmup_steps: int = 100,
        gradient_clip: float = 1.0,
        save_every_n_steps: int = 500,
        eval_every_n_steps: int = 100,
    ):
        """Initialize configuration."""
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.warmup_steps = warmup_steps
        self.gradient_clip = gradient_clip
        self.save_every_n_steps = save_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps


class HRMFineTuner:
    """
    Fine-tunes HRM model on requirements-to-test-cases data.
    
    Implements training loop with:
    - Learning rate scheduling
    - Gradient clipping
    - Validation monitoring
    - Checkpoint saving
    """
    
    def __init__(
        self,
        model: HRMModel,
        device: torch.device,
        config: FineTuningConfig,
        output_dir: str = "fine_tuned_checkpoints",
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model: HRM model to fine-tune
            device: Device for computation
            config: Fine-tuning configuration
            output_dir: Directory for saving checkpoints
        """
        self.model = model
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        
        self.lm_criterion = nn.CrossEntropyLoss()
        self.q_criterion = nn.MSELoss()
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"HRMFineTuner initialized (lr={config.learning_rate}, epochs={config.epochs})")
    
    def fine_tune(
        self,
        training_data_path: str,
        val_data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fine-tune model on training data.
        
        Args:
            training_data_path: Path to training data JSONL
            val_data_path: Optional path to validation data
            
        Returns:
            Training metrics and results
        """
        logger.info(f"Starting fine-tuning from {training_data_path}")
        
        train_dataset = PuzzleDataset(Path(training_data_path))
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
        
        val_loader = None
        if val_data_path:
            val_dataset = PuzzleDataset(Path(val_data_path))
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )
        
        training_metrics = {
            "total_epochs": self.config.epochs,
            "total_steps": 0,
            "train_loss_history": [],
            "val_loss_history": [],
            "best_val_loss": float('inf'),
            "best_checkpoint": None,
        }
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
            
            train_loss = self._train_epoch(train_loader, epoch)
            
            training_metrics["train_loss_history"].append(train_loss)
            
            if val_loader:
                val_loss = self._validate(val_loader)
                training_metrics["val_loss_history"].append(val_loss)
                
                if val_loss < training_metrics["best_val_loss"]:
                    training_metrics["best_val_loss"] = val_loss
                    checkpoint_path = self._save_checkpoint(
                        epoch=epoch,
                        val_loss=val_loss,
                        is_best=True,
                    )
                    training_metrics["best_checkpoint"] = checkpoint_path
                    logger.info(f"New best validation loss: {val_loss:.4f}")
            
            checkpoint_path = self._save_checkpoint(
                epoch=epoch,
                val_loss=val_loss if val_loader else None,
                is_best=False,
            )
        
        training_time = time.time() - start_time
        training_metrics["total_steps"] = self.global_step
        training_metrics["training_time_seconds"] = training_time
        
        logger.info(f"Fine-tuning complete in {training_time:.2f}s")
        logger.info(f"Best validation loss: {training_metrics['best_val_loss']:.4f}")
        
        return training_metrics
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch+1}")
        
        for batch in pbar:
            self.global_step += 1
            
            loss = self._train_step(batch)
            
            total_loss += loss
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            if self.global_step % self.config.eval_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                logger.debug(f"Step {self.global_step}: avg_loss={avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        logger.info(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Loss value
        """
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        puzzle_ids = batch["puzzle_ids"].to(self.device)
        
        self.optimizer.zero_grad()
        
        outputs = self.model(
            input_ids=input_ids,
            puzzle_ids=puzzle_ids,
        )
        
        lm_logits = outputs["lm_logits"]
        q_values = outputs["q_values"]
        
        lm_loss = self.lm_criterion(
            lm_logits.view(-1, lm_logits.size(-1)),
            target_ids.view(-1),
        )
        
        q_targets = torch.zeros_like(q_values)
        q_loss = self.q_criterion(q_values, q_targets)
        
        loss = lm_loss + 0.1 * q_loss
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip,
        )
        
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            puzzle_ids = batch["puzzle_ids"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                puzzle_ids=puzzle_ids,
            )
            
            lm_logits = outputs["lm_logits"]
            
            loss = self.lm_criterion(
                lm_logits.view(-1, lm_logits.size(-1)),
                target_ids.view(-1),
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        logger.info(f"Validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: Optional[float] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss (if available)
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_epoch_{epoch+1}"
        if is_best:
            checkpoint_name += "_best"
        checkpoint_name += ".pt"
        
        checkpoint_path = self.output_dir / checkpoint_name
        
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
            },
        }
        
        if val_loss is not None:
            checkpoint["val_loss"] = val_loss
        
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return str(checkpoint_path)

