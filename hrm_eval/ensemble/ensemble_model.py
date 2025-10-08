"""
Ensemble model implementation for combining multiple checkpoints.

Supports weighted averaging, voting, and stacking strategies.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from enum import Enum
import logging

from ..models import HRMModel

logger = logging.getLogger(__name__)


class EnsembleStrategy(Enum):
    """Ensemble combination strategies."""
    
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKING = "stacking"


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple HRM models.
    
    Combines predictions from multiple model checkpoints using
    specified aggregation strategy.
    """
    
    def __init__(
        self,
        models: List[HRMModel],
        strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE,
        weights: Optional[List[float]] = None,
        voting_threshold: float = 0.5,
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of HRM models
            strategy: Ensemble strategy to use
            weights: Optional weights for weighted average (must sum to 1.0)
            voting_threshold: Threshold for voting strategy
            
        Raises:
            ValueError: If weights invalid or strategy unsupported
        """
        super().__init__()
        
        if not models:
            raise ValueError("At least one model required for ensemble")
        
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        self.num_models = len(models)
        
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            if len(weights) != self.num_models:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of models ({self.num_models})"
                )
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights
        
        self.voting_threshold = voting_threshold
        
        for i, model in enumerate(self.models):
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        
        logger.info(
            f"Initialized EnsembleModel with {self.num_models} models "
            f"using {strategy.value} strategy"
        )
        logger.info(f"Weights: {self.weights}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        puzzle_ids: torch.Tensor,
        h_state: Optional[torch.Tensor] = None,
        l_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            input_ids: Token IDs
            puzzle_ids: Puzzle IDs
            h_state: Optional H-level state
            l_state: Optional L-level state
            attention_mask: Optional attention mask
            return_states: Whether to return hidden states
            
        Returns:
            Dictionary with ensemble predictions
        """
        model_outputs = []
        
        for model in self.models:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    puzzle_ids=puzzle_ids,
                    h_state=h_state,
                    l_state=l_state,
                    attention_mask=attention_mask,
                    return_states=return_states,
                )
                model_outputs.append(outputs)
        
        if self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            combined = self._weighted_average(model_outputs)
        elif self.strategy == EnsembleStrategy.VOTING:
            combined = self._voting(model_outputs)
        elif self.strategy == EnsembleStrategy.STACKING:
            combined = self._stacking(model_outputs)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
        
        return combined
    
    def _weighted_average(
        self,
        model_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Combine outputs using weighted average.
        
        Args:
            model_outputs: List of model outputs
            
        Returns:
            Combined output dictionary
        """
        combined = {}
        
        lm_logits_list = [out["lm_logits"] for out in model_outputs]
        combined["lm_logits"] = sum(
            w * logits for w, logits in zip(self.weights, lm_logits_list)
        )
        
        q_values_list = [out["q_values"] for out in model_outputs]
        combined["q_values"] = sum(
            w * q_vals for w, q_vals in zip(self.weights, q_values_list)
        )
        
        if "h_state" in model_outputs[0]:
            h_states = [out["h_state"] for out in model_outputs]
            combined["h_state"] = sum(
                w * state for w, state in zip(self.weights, h_states)
            )
        
        if "l_state" in model_outputs[0]:
            l_states = [out["l_state"] for out in model_outputs]
            combined["l_state"] = sum(
                w * state for w, state in zip(self.weights, l_states)
            )
        
        return combined
    
    def _voting(
        self,
        model_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Combine outputs using majority voting.
        
        Args:
            model_outputs: List of model outputs
            
        Returns:
            Combined output dictionary
        """
        lm_logits_list = [out["lm_logits"] for out in model_outputs]
        predictions = [torch.argmax(logits, dim=-1) for logits in lm_logits_list]
        
        batch_size, seq_len = predictions[0].shape
        vocab_size = lm_logits_list[0].size(-1)
        
        vote_counts = torch.zeros(
            batch_size, seq_len, vocab_size,
            device=predictions[0].device
        )
        
        for pred in predictions:
            vote_counts.scatter_add_(
                2,
                pred.unsqueeze(-1),
                torch.ones_like(pred.unsqueeze(-1), dtype=torch.float)
            )
        
        combined_lm_logits = vote_counts / len(predictions)
        
        q_values_list = [out["q_values"] for out in model_outputs]
        q_predictions = [torch.argmax(q_vals, dim=-1) for q_vals in q_values_list]
        
        num_actions = q_values_list[0].size(-1)
        q_vote_counts = torch.zeros(
            batch_size, num_actions,
            device=q_predictions[0].device
        )
        
        for q_pred in q_predictions:
            q_vote_counts.scatter_add_(
                1,
                q_pred.unsqueeze(-1),
                torch.ones_like(q_pred.unsqueeze(-1), dtype=torch.float)
            )
        
        combined_q_values = q_vote_counts / len(predictions)
        
        return {
            "lm_logits": combined_lm_logits,
            "q_values": combined_q_values,
        }
    
    def _stacking(
        self,
        model_outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Combine outputs using stacking (learned combination).
        
        Note: Currently falls back to weighted average.
        Full stacking requires training a meta-learner.
        
        Args:
            model_outputs: List of model outputs
            
        Returns:
            Combined output dictionary
        """
        logger.warning("Stacking strategy not fully implemented, using weighted average")
        return self._weighted_average(model_outputs)
    
    @staticmethod
    def create_from_checkpoints(
        checkpoint_paths: List[str],
        config,
        device: torch.device,
        strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE,
        weights: Optional[List[float]] = None,
    ) -> 'EnsembleModel':
        """
        Create ensemble from checkpoint files.
        
        Args:
            checkpoint_paths: List of checkpoint file paths
            config: Model configuration
            device: Device to load models on
            strategy: Ensemble strategy
            weights: Optional ensemble weights
            
        Returns:
            EnsembleModel instance
        """
        from ..utils import load_checkpoint
        from ..models import HRMConfig, HRMModel
        
        models = []
        
        for i, ckpt_path in enumerate(checkpoint_paths):
            logger.info(f"Loading checkpoint {i+1}/{len(checkpoint_paths)}: {ckpt_path}")
            
            hrm_config = HRMConfig.from_yaml_config(config)
            model = HRMModel(hrm_config)
            
            checkpoint = load_checkpoint(ckpt_path, device=str(device))
            model.load_from_checkpoint(checkpoint)
            
            model.to(device)
            model.eval()
            
            models.append(model)
        
        ensemble = EnsembleModel(
            models=models,
            strategy=strategy,
            weights=weights,
        )
        
        return ensemble

