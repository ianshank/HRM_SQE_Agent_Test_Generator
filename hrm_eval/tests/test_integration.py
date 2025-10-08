"""Integration tests for end-to-end evaluation pipeline."""

import pytest
import torch
from pathlib import Path
from ..models import HRMModel, HRMConfig
from ..data import PuzzleDataset, create_dataloader
from ..evaluation import Evaluator, MetricsCalculator
from ..utils.config_utils import ModelConfig, EvaluationConfig, DataConfig, LoggingConfig, WandBConfig, EnsembleConfig, DeviceConfig, Config, CheckpointConfig


class TestEndToEndEvaluation:
    """Integration tests for complete evaluation pipeline."""
    
    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
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
    
    @pytest.fixture
    def full_config(self):
        """Create full configuration."""
        return Config(
            model=ModelConfig(
                name="test_hrm",
                vocab_size=12,
                embed_dim=256,
                num_puzzles=1000,
                h_level=...,  # Simplified for testing
                l_level=...,
                lm_head=...,
                q_head=...,
            ),
            checkpoint=CheckpointConfig(
                base_dir="./",
                primary="test_checkpoint",
            ),
            device=DeviceConfig(type="cpu"),
            evaluation=EvaluationConfig(
                batch_size=2,
                num_workers=0,
                max_steps_per_puzzle=10,
                timeout_seconds=5,
                metrics=["solve_rate", "accuracy"],
                save_predictions=False,
                save_trajectories=False,
                output_dir="./results",
            ),
            data=DataConfig(
                validation_set="./data/val",
                test_set="./data/test",
                data_format="jsonl",
            ),
            logging=LoggingConfig(
                level="INFO",
                format="json",
                log_dir="./logs",
            ),
            wandb=WandBConfig(enabled=False),
            ensemble=EnsembleConfig(enabled=False),
        )
    
    def test_model_creation_and_forward(self, model_config):
        """Test model creation and forward pass."""
        model = HRMModel(model_config)
        model.eval()
        
        input_ids = torch.randint(0, 12, (2, 10))
        puzzle_ids = torch.randint(0, 1000, (2,))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, puzzle_ids=puzzle_ids)
        
        assert outputs["lm_logits"].shape == (2, 10, 12)
        assert outputs["q_values"].shape == (2, 2)
    
    def test_dataset_loading(self):
        """Test dataset creation and loading."""
        dataset = PuzzleDataset(
            data_path=Path("nonexistent.jsonl"),
            max_seq_len=50,
            vocab_size=12,
        )
        
        assert len(dataset) > 0
        
        sample = dataset[0]
        assert "puzzle_id" in sample
        assert "input_ids" in sample
        assert "target_ids" in sample
    
    def test_dataloader_creation(self):
        """Test DataLoader creation."""
        dataset = PuzzleDataset(
            data_path=Path("nonexistent.jsonl"),
            max_seq_len=50,
            vocab_size=12,
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            num_workers=0,
            shuffle=False,
        )
        
        batch = next(iter(dataloader))
        
        assert "puzzle_ids" in batch
        assert "input_ids" in batch
        assert batch["puzzle_ids"].shape[0] == 4

