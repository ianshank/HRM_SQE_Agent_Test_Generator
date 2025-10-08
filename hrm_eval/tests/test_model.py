"""Unit tests for HRM model components."""

import pytest
import torch
from ..models import HRMModel, HRMConfig
from ..models.transformer_layers import MultiHeadAttention, MLP, TransformerLayer


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""
    
    def test_initialization(self):
        """Test attention module initialization."""
        attn = MultiHeadAttention(hidden_size=256, num_attention_heads=8)
        assert attn.hidden_size == 256
        assert attn.num_attention_heads == 8
        assert attn.head_dim == 32
    
    def test_invalid_hidden_size(self):
        """Test error on invalid hidden size."""
        with pytest.raises(ValueError):
            MultiHeadAttention(hidden_size=255, num_attention_heads=8)
    
    def test_forward_pass(self):
        """Test forward pass shape correctness."""
        attn = MultiHeadAttention(hidden_size=256, num_attention_heads=8)
        x = torch.randn(2, 10, 256)
        
        output = attn(x)
        
        assert output.shape == (2, 10, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestMLP:
    """Tests for MLP module."""
    
    def test_initialization(self):
        """Test MLP initialization."""
        mlp = MLP(hidden_size=256, intermediate_size=768)
        assert mlp.hidden_size == 256
        assert mlp.intermediate_size == 768
    
    def test_forward_pass(self):
        """Test forward pass shape correctness."""
        mlp = MLP(hidden_size=256, intermediate_size=768)
        x = torch.randn(2, 10, 256)
        
        output = mlp(x)
        
        assert output.shape == (2, 10, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestTransformerLayer:
    """Tests for TransformerLayer module."""
    
    def test_initialization(self):
        """Test transformer layer initialization."""
        layer = TransformerLayer(
            hidden_size=256,
            num_attention_heads=8,
            intermediate_size=768,
        )
        assert layer.self_attn is not None
        assert layer.mlp is not None
    
    def test_forward_pass(self):
        """Test forward pass shape correctness."""
        layer = TransformerLayer(
            hidden_size=256,
            num_attention_heads=8,
            intermediate_size=768,
        )
        x = torch.randn(2, 10, 256)
        
        output = layer(x)
        
        assert output.shape == (2, 10, 256)
        assert not torch.isnan(output).any()


class TestHRMModel:
    """Tests for HRMModel."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HRMConfig(
            vocab_size=12,
            embed_dim=256,
            num_puzzles=1000,
            h_num_layers=2,
            h_hidden_size=256,
            h_intermediate_size=768,
            h_num_attention_heads=8,
            h_dropout=0.1,
            l_num_layers=2,
            l_hidden_size=256,
            l_intermediate_size=768,
            l_num_attention_heads=8,
            l_dropout=0.1,
            num_actions=2,
        )
    
    def test_model_initialization(self, config):
        """Test model initialization."""
        model = HRMModel(config)
        
        assert model.config == config
        assert model.embed_tokens.num_embeddings == 12
        assert model.puzzle_emb.num_puzzles == 1000
    
    def test_forward_pass(self, config):
        """Test forward pass with correct shapes."""
        model = HRMModel(config)
        model.eval()
        
        input_ids = torch.randint(0, 12, (2, 10))
        puzzle_ids = torch.randint(0, 1000, (2,))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, puzzle_ids=puzzle_ids)
        
        assert "lm_logits" in outputs
        assert "q_values" in outputs
        assert outputs["lm_logits"].shape == (2, 10, 12)
        assert outputs["q_values"].shape == (2, 2)
    
    def test_parameter_count(self, config):
        """Test parameter counting."""
        model = HRMModel(config)
        
        param_counts = model.get_num_params()
        
        assert "total" in param_counts
        assert param_counts["total"] > 0
        assert param_counts["puzzle_emb"] == 1000 * 256

