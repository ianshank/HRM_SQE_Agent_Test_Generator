"""
Convert SQE agent JSONL data to HRM-compatible format.

Converts prompt-completion pairs to tokenized sequences suitable for HRM model.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import logging

logger = logging.getLogger(__name__)


class HRMDataConverter:
    """
    Convert text data to HRM token format.
    
    HRM uses a specialized 12-token vocabulary. This converter maps
    text to token sequences using a simple hashing-based approach.
    """
    
    VOCAB_SIZE = 12
    MAX_SEQ_LEN = 512
    
    def __init__(self):
        """Initialize converter."""
        self.token_map = {
            'start': 0,
            'test': 1,
            'agent': 2,
            'data': 3,
            'security': 4,
            'performance': 5,
            'integration': 6,
            'automation': 7,
            'validation': 8,
            'deployment': 9,
            'monitoring': 10,
            'end': 11,
        }
        
        self.reverse_map = {v: k for k, v in self.token_map.items()}
        
        logger.info(f"Initialized HRM converter with vocab_size={self.VOCAB_SIZE}")
    
    def text_to_tokens(self, text: str, max_len: int = None) -> List[int]:
        """
        Convert text to token sequence.
        
        Args:
            text: Input text
            max_len: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        if max_len is None:
            max_len = self.MAX_SEQ_LEN
        
        words = text.lower().split()
        tokens = [0]  # Start token
        
        for word in words:
            token_id = self._word_to_token(word)
            tokens.append(token_id)
            
            if len(tokens) >= max_len - 1:
                break
        
        tokens.append(11)  # End token
        
        return tokens
    
    def _word_to_token(self, word: str) -> int:
        """
        Map a word to a token ID using keyword matching.
        
        Args:
            word: Input word
            
        Returns:
            Token ID (1-10)
        """
        word_lower = word.lower()
        
        for keyword, token_id in self.token_map.items():
            if keyword in word_lower:
                return token_id
        
        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
        return (hash_val % 9) + 1  # Map to tokens 1-9
    
    def convert_example(
        self,
        prompt: str,
        completion: str,
        example_id: int,
    ) -> Dict[str, Any]:
        """
        Convert a single prompt-completion pair to HRM format.
        
        Args:
            prompt: Input prompt text
            completion: Target completion text
            example_id: Unique example ID
            
        Returns:
            HRM-formatted example
        """
        input_tokens = self.text_to_tokens(prompt, max_len=100)
        target_tokens = self.text_to_tokens(completion, max_len=100)
        
        puzzle_id = example_id % 1000  # Map to puzzle ID space
        
        num_actions = min(len(input_tokens), len(target_tokens))
        solution_steps = [
            {"action": i % 2, "state": None}
            for i in range(num_actions)
        ]
        
        return {
            "puzzle_id": puzzle_id,
            "input_sequence": input_tokens,
            "target_sequence": target_tokens,
            "solution_steps": solution_steps,
            "metadata": {
                "source": "sqe_agent",
                "prompt_length": len(prompt),
                "completion_length": len(completion),
                "difficulty": self._estimate_difficulty(prompt, completion),
            }
        }
    
    def _estimate_difficulty(self, prompt: str, completion: str) -> str:
        """
        Estimate difficulty based on text length.
        
        Args:
            prompt: Prompt text
            completion: Completion text
            
        Returns:
            Difficulty level (easy, medium, hard)
        """
        total_length = len(prompt) + len(completion)
        
        if total_length < 500:
            return "easy"
        elif total_length < 1500:
            return "medium"
        else:
            return "hard"
    
    def convert_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> int:
        """
        Convert entire JSONL file to HRM format.
        
        Args:
            input_path: Input JSONL file path
            output_path: Output JSONL file path
            
        Returns:
            Number of examples converted
        """
        logger.info(f"Converting {input_path} to HRM format")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        num_examples = 0
        
        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for idx, line in enumerate(f_in):
                try:
                    data = json.loads(line)
                    
                    if 'prompt' not in data or 'completion' not in data:
                        logger.warning(f"Skipping line {idx}: missing prompt or completion")
                        continue
                    
                    hrm_example = self.convert_example(
                        prompt=data['prompt'],
                        completion=data['completion'],
                        example_id=idx,
                    )
                    
                    f_out.write(json.dumps(hrm_example) + '\n')
                    num_examples += 1
                    
                except Exception as e:
                    logger.error(f"Error converting line {idx}: {e}")
                    continue
        
        logger.info(f"Converted {num_examples} examples to {output_path}")
        
        return num_examples


def main():
    """Main conversion function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    input_file = Path("../sqe_agent_real_data.jsonl")
    output_file = Path("./data/sqe_agent_hrm_format.jsonl")
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    converter = HRMDataConverter()
    num_converted = converter.convert_file(input_file, output_file)
    
    logger.info("="*60)
    logger.info(f"Conversion complete: {num_converted} examples")
    logger.info(f"Output saved to: {output_file}")
    logger.info("="*60)
    
    logger.info("\nSample converted example:")
    with open(output_file, 'r') as f:
        sample = json.loads(f.readline())
        print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()

