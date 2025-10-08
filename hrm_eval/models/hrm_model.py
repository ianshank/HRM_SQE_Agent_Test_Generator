"""
HRM v9 Optimized Model Architecture.

Implements the complete Hierarchical Recurrent Model with dual-level transformers,
puzzle embeddings, and reinforcement learning heads.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import logging

from .transformer_layers import TransformerStack

logger = logging.getLogger(__name__)


@dataclass
class HRMConfig:
    """Configuration for HRM model architecture."""
    
    vocab_size: int
    embed_dim: int
    num_puzzles: int
    
    h_num_layers: int
    h_hidden_size: int
    h_intermediate_size: int
    h_num_attention_heads: int
    h_dropout: float
    
    l_num_layers: int
    l_hidden_size: int
    l_intermediate_size: int
    l_num_attention_heads: int
    l_dropout: float
    
    num_actions: int = 2
    
    @classmethod
    def from_yaml_config(cls, config):
        """Create HRMConfig from YAML config object."""
        return cls(
            vocab_size=config.model.vocab_size,
            embed_dim=config.model.embed_dim,
            num_puzzles=config.model.num_puzzles,
            h_num_layers=config.model.h_level.num_layers,
            h_hidden_size=config.model.h_level.hidden_size,
            h_intermediate_size=config.model.h_level.intermediate_size,
            h_num_attention_heads=config.model.h_level.num_attention_heads,
            h_dropout=config.model.h_level.dropout,
            l_num_layers=config.model.l_level.num_layers,
            l_hidden_size=config.model.l_level.hidden_size,
            l_intermediate_size=config.model.l_level.intermediate_size,
            l_num_attention_heads=config.model.l_level.num_attention_heads,
            l_dropout=config.model.l_level.dropout,
            num_actions=config.model.q_head.num_actions,
        )


class PuzzleEmbedding(nn.Module):
    """Learnable puzzle embeddings for puzzle-specific representations."""
    
    def __init__(self, num_puzzles: int, embed_dim: int):
        """
        Initialize puzzle embeddings.
        
        Args:
            num_puzzles: Number of unique puzzles
            embed_dim: Embedding dimension
        """
        super().__init__()
        
        self.num_puzzles = num_puzzles
        self.embed_dim = embed_dim
        
        self.weights = nn.Parameter(torch.randn(num_puzzles, embed_dim))
        
        logger.info(
            f"Initialized PuzzleEmbedding: {num_puzzles} puzzles, "
            f"dim={embed_dim}, params={num_puzzles * embed_dim:,}"
        )
    
    def forward(self, puzzle_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for puzzle IDs.
        
        Args:
            puzzle_ids: Tensor of puzzle IDs [batch_size]
            
        Returns:
            Puzzle embeddings [batch_size, embed_dim]
        """
        return self.weights[puzzle_ids]


class HRMModel(nn.Module):
    """
    Hierarchical Recurrent Model for puzzle solving.
    
    Implements dual-level transformer architecture with:
    - Puzzle-specific embeddings
    - Token embeddings
    - H-level (high-level) transformers
    - L-level (low-level) transformers
    - Language modeling head
    - Q-value head for reinforcement learning
    """
    
    def __init__(self, config: HRMConfig):
        """
        Initialize HRM model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        self.H_init = nn.Parameter(torch.randn(config.embed_dim))
        self.L_init = nn.Parameter(torch.randn(config.embed_dim))
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim)
        
        self.puzzle_emb = PuzzleEmbedding(
            num_puzzles=config.num_puzzles,
            embed_dim=config.embed_dim,
        )
        
        self.H_level = TransformerStack(
            num_layers=config.h_num_layers,
            hidden_size=config.h_hidden_size,
            num_attention_heads=config.h_num_attention_heads,
            intermediate_size=config.h_intermediate_size,
            dropout=config.h_dropout,
        )
        
        self.L_level = TransformerStack(
            num_layers=config.l_num_layers,
            hidden_size=config.l_hidden_size,
            num_attention_heads=config.l_num_attention_heads,
            intermediate_size=config.l_intermediate_size,
            dropout=config.l_dropout,
        )
        
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        self.q_head = nn.Linear(config.embed_dim, config.num_actions, bias=True)
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Initialized HRMModel with {total_params:,} parameters "
            f"({total_params * 4 / (1024**2):.2f} MB)"
        )
    
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
        Forward pass of HRM model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            puzzle_ids: Puzzle IDs [batch_size]
            h_state: Optional H-level hidden state
            l_state: Optional L-level hidden state
            attention_mask: Optional attention mask
            return_states: Whether to return hidden states
            
        Returns:
            Dictionary containing:
                - lm_logits: Language modeling logits [batch_size, seq_len, vocab_size]
                - q_values: Q-values [batch_size, num_actions]
                - h_state: H-level state (if return_states=True)
                - l_state: L-level state (if return_states=True)
        """
        batch_size, seq_len = input_ids.shape
        
        token_embeds = self.embed_tokens(input_ids)
        
        puzzle_embeds = self.puzzle_emb(puzzle_ids)
        puzzle_embeds = puzzle_embeds.unsqueeze(1).expand(-1, seq_len, -1)
        
        hidden_states = token_embeds + puzzle_embeds
        
        if h_state is None:
            h_state = self.H_init.unsqueeze(0).unsqueeze(0).expand(
                batch_size, 1, -1
            )
        
        if l_state is None:
            l_state = self.L_init.unsqueeze(0).unsqueeze(0).expand(
                batch_size, 1, -1
            )
        
        h_input = torch.cat([h_state, hidden_states], dim=1)
        h_output = self.H_level(h_input, attention_mask)
        
        l_input = torch.cat([l_state, h_output], dim=1)
        l_output = self.L_level(l_input, attention_mask)
        
        final_hidden = l_output[:, -seq_len:, :]
        
        lm_logits = self.lm_head(final_hidden)
        
        pooled = final_hidden[:, -1, :]
        q_values = self.q_head(pooled)
        
        outputs = {
            "lm_logits": lm_logits,
            "q_values": q_values,
        }
        
        if return_states:
            outputs["h_state"] = h_output[:, :1, :]
            outputs["l_state"] = l_output[:, :1, :]
        
        return outputs
    
    def load_from_checkpoint(self, checkpoint: Dict[str, torch.Tensor]) -> None:
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary with model weights
            
        Raises:
            RuntimeError: If loading fails
        """
        try:
            state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith("model.inner."):
                    new_key = key.replace("model.inner.", "")
                    state_dict[new_key] = value
                else:
                    state_dict[key] = value
            
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            
            if missing:
                logger.warning(f"Missing keys in checkpoint: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
            
            logger.info("Model weights loaded successfully from checkpoint")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    def get_num_params(self) -> Dict[str, int]:
        """
        Get parameter counts by component.
        
        Returns:
            Dictionary mapping component names to parameter counts
        """
        params = {
            "puzzle_emb": self.puzzle_emb.weights.numel(),
            "embed_tokens": self.embed_tokens.weight.numel(),
            "H_level": sum(p.numel() for p in self.H_level.parameters()),
            "L_level": sum(p.numel() for p in self.L_level.parameters()),
            "lm_head": self.lm_head.weight.numel(),
            "q_head": sum(p.numel() for p in self.q_head.parameters()),
            "init_states": self.H_init.numel() + self.L_init.numel(),
        }
        params["total"] = sum(params.values())
        
        return params

