"""
Dataset classes for puzzle data loading.

NOTE: This is a template implementation. The actual dataset format needs to be
adapted based on your specific puzzle data structure.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class PuzzleDataset(Dataset):
    """
    Dataset for puzzle solving tasks.
    
    Expected data format (JSONL):
    {
        "puzzle_id": int,
        "input_sequence": List[int],  # Token IDs
        "target_sequence": List[int],  # Target token IDs
        "solution_steps": List[Dict],  # Ground truth solution
        "metadata": Dict  # Additional puzzle metadata
    }
    
    TODO: Adapt this class to match your actual data format.
    """
    
    def __init__(
        self,
        data_path: Path,
        max_seq_len: int = 512,
        vocab_size: int = 12,
    ):
        """
        Initialize puzzle dataset.
        
        Args:
            data_path: Path to dataset file
            max_seq_len: Maximum sequence length
            vocab_size: Vocabulary size
        """
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        self.examples = self._load_data()
        
        logger.info(
            f"Loaded {len(self.examples)} examples from {data_path}"
        )
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from file.
        
        Returns:
            List of puzzle examples
            
        Note:
            If data file doesn't exist, creates mock data for testing.
        """
        if not self.data_path.exists():
            logger.warning(
                f"Data file not found: {self.data_path}. "
                "Creating mock data for testing."
            )
            return self._create_mock_data()
        
        examples = []
        
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r') as f:
                for line in f:
                    example = json.loads(line)
                    examples.append(example)
        
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                examples = json.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        return examples
    
    def _create_mock_data(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Create mock data for testing.
        
        Args:
            num_samples: Number of mock samples to create
            
        Returns:
            List of mock puzzle examples
        """
        logger.info(f"Creating {num_samples} mock puzzle examples")
        
        examples = []
        for i in range(num_samples):
            seq_len = torch.randint(5, 20, (1,)).item()
            
            example = {
                "puzzle_id": i % 1000,  # Cycle through 1000 puzzles
                "input_sequence": torch.randint(
                    0, self.vocab_size, (seq_len,)
                ).tolist(),
                "target_sequence": torch.randint(
                    0, self.vocab_size, (seq_len,)
                ).tolist(),
                "solution_steps": [
                    {"action": j % 2, "state": None}
                    for j in range(seq_len)
                ],
                "metadata": {
                    "difficulty": "easy" if i % 3 == 0 else "medium",
                    "category": f"category_{i % 5}",
                },
            }
            examples.append(example)
        
        return examples
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Args:
            idx: Example index
            
        Returns:
            Dictionary containing:
                - puzzle_id: Puzzle ID tensor
                - input_ids: Input token IDs
                - target_ids: Target token IDs
                - attention_mask: Attention mask
        """
        example = self.examples[idx]
        
        input_ids = torch.tensor(example["input_sequence"], dtype=torch.long)
        target_ids = torch.tensor(example["target_sequence"], dtype=torch.long)
        
        seq_len = len(input_ids)
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            target_ids = target_ids[:self.max_seq_len]
            seq_len = self.max_seq_len
        
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        
        return {
            "puzzle_id": torch.tensor(example["puzzle_id"], dtype=torch.long),
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "metadata": example.get("metadata", {}),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching with padding.
    
    Handles variable-length input and target sequences.
    
    Args:
        batch: List of examples
        
    Returns:
        Batched and padded tensors
    """
    puzzle_ids = torch.stack([item["puzzle_id"] for item in batch])
    
    # Calculate max lengths for input and target separately
    max_input_len = max(item["input_ids"].size(0) for item in batch)
    max_target_len = max(item["target_ids"].size(0) for item in batch)
    
    # Use the maximum of both for padding
    max_len = max(max_input_len, max_target_len)
    
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    target_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, item in enumerate(batch):
        input_seq_len = item["input_ids"].size(0)
        target_seq_len = item["target_ids"].size(0)
        
        input_ids[i, :input_seq_len] = item["input_ids"]
        target_ids[i, :target_seq_len] = item["target_ids"]
        
        # Attention mask based on input length
        attention_mask[i, :input_seq_len] = item["attention_mask"]
    
    return {
        "puzzle_ids": puzzle_ids,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
    }


def create_dataloader(
    dataset: PuzzleDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    """
    Create DataLoader for puzzle dataset.
    
    Args:
        dataset: PuzzleDataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

