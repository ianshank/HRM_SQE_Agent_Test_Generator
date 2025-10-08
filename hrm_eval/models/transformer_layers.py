"""
Transformer layer components for HRM model.

Implements multi-head attention, MLP blocks, and complete transformer layers
matching the checkpoint architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Implements efficient multi-head attention with combined QKV projection.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            
        Raises:
            ValueError: If hidden_size not divisible by num_attention_heads
        """
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.debug(
            f"Initialized MultiHeadAttention: "
            f"hidden_size={hidden_size}, heads={num_attention_heads}, "
            f"head_dim={self.head_dim}"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(
            batch_size, seq_len, 3, self.num_attention_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        output = self.o_proj(attn_output)
        
        return output


class MLP(nn.Module):
    """
    Multi-layer perceptron with gated activation.
    
    Implements SwiGLU-style MLP with combined gate and up projection.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        """
        Initialize MLP block.
        
        Args:
            hidden_size: Hidden dimension size
            intermediate_size: Intermediate layer size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.gate_up_proj = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.debug(
            f"Initialized MLP: hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}"
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        gate_up = self.gate_up_proj(hidden_states)
        
        gate, up = gate_up.chunk(2, dim=-1)
        
        gate = F.silu(gate)
        
        intermediate = gate * up
        
        output = self.down_proj(intermediate)
        output = self.dropout(output)
        
        return output


class TransformerLayer(nn.Module):
    """
    Complete transformer layer with attention and MLP.
    
    Implements pre-norm transformer architecture with residual connections.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        """
        Initialize transformer layer.
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            intermediate_size: Intermediate MLP size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )
        
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        
        logger.debug(f"Initialized TransformerLayer: hidden_size={hidden_size}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of transformer layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class TransformerStack(nn.Module):
    """Stack of transformer layers."""
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        """
        Initialize transformer stack.
        
        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            intermediate_size: Intermediate MLP size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        logger.info(
            f"Initialized TransformerStack with {num_layers} layers, "
            f"hidden_size={hidden_size}"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all layers.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states

