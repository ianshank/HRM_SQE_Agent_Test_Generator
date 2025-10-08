# HRM v9 Optimized - Evaluation & Deployment Framework

Production-ready evaluation framework for the Hierarchical Recurrent Model (HRM) v9 Optimized for puzzle solving tasks.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for models, data, evaluation, and utilities
- **Checkpoint Management**: Robust loading and validation of PyTorch checkpoints
- **Comprehensive Metrics**: Solve rate, accuracy, step efficiency, and time-based measurements
- **Ensemble Support**: Weighted averaging, voting, and stacking strategies for combining multiple checkpoints
- **Extensive Testing**: Unit tests, integration tests, and contract tests with >80% coverage
- **Production Logging**: Structured JSON logging with multiple output streams
- **Type Safety**: Pydantic-based configuration validation
- **GPU Support**: CUDA, MPS, and CPU device support with mixed precision

## Project Structure

```
hrm_eval/
├── models/               # Model architecture definitions
│   ├── hrm_model.py     # Main HRM model
│   └── transformer_layers.py  # Transformer components
├── data/                # Data loading and preprocessing
│   ├── dataset.py       # Dataset classes
│   └── puzzle_env.py    # Puzzle environment simulator
├── evaluation/          # Evaluation framework
│   ├── metrics.py       # Metrics computation
│   └── evaluator.py     # Main evaluator
├── ensemble/            # Ensemble mechanisms
│   └── ensemble_model.py  # Ensemble model implementation
├── utils/               # Utility functions
│   ├── logging_utils.py   # Logging setup
│   ├── config_utils.py    # Configuration management
│   └── checkpoint_utils.py  # Checkpoint loading
├── tests/               # Test suite
│   ├── test_model.py
│   ├── test_metrics.py
│   ├── test_utils.py
│   └── test_integration.py
├── configs/             # Configuration files
│   ├── model_config.yaml
│   └── eval_config.yaml
├── deploy.py            # Main deployment script
├── requirements.txt     # Python dependencies
└── setup.py            # Package setup

```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (optional, for GPU support)

### Setup

```bash
# Navigate to the project directory
cd hrm_eval

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

## Quick Start

### 1. Evaluate Single Checkpoint

Evaluate the best checkpoint (step_7566):

```bash
python deploy.py \
    --mode evaluate \
    --checkpoint step_7566 \
    --data-path ./data/validation \
    --output-dir ./results \
    --device cuda
```

### 2. Evaluate Ensemble

Evaluate ensemble of two checkpoints:

```bash
python deploy.py \
    --mode ensemble \
    --checkpoints step_7566 step_3607 \
    --data-path ./data/validation \
    --output-dir ./results \
    --device cuda
```

### 3. Run Tests

Run the complete test suite:

```bash
python deploy.py --mode test
```

Or use pytest directly:

```bash
pytest tests/ -v --cov=hrm_eval --cov-report=html
```

## Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  name: "hrm_v9_optimized"
  vocab_size: 12
  embed_dim: 256
  num_puzzles: 95996
  
  h_level:  # High-level transformer
    num_layers: 2
    hidden_size: 256
    intermediate_size: 768
    num_attention_heads: 8
    dropout: 0.1
  
  l_level:  # Low-level transformer
    num_layers: 2
    hidden_size: 256
    intermediate_size: 768
    num_attention_heads: 8
    dropout: 0.1
```

### Evaluation Configuration (`configs/eval_config.yaml`)

```yaml
evaluation:
  batch_size: 32
  num_workers: 4
  max_steps_per_puzzle: 1000
  timeout_seconds: 60
  
  metrics:
    - "solve_rate"
    - "accuracy"
    - "average_steps"
    - "step_efficiency"
  
  save_predictions: true
  save_trajectories: false
  output_dir: "./results"
```

## Usage Examples

### Programmatic Usage

```python
import torch
from pathlib import Path
from hrm_eval import HRMModel, HRMConfig, Evaluator, PuzzleDataset
from hrm_eval.utils import load_checkpoint, load_config

# Load configuration
config = load_config(
    model_config_path=Path("configs/model_config.yaml"),
    eval_config_path=Path("configs/eval_config.yaml"),
)

# Create model
hrm_config = HRMConfig.from_yaml_config(config)
model = HRMModel(hrm_config)

# Load checkpoint
checkpoint = load_checkpoint("../checkpoints_hrm_v9_optimized_step_7566")
model.load_from_checkpoint(checkpoint)

# Setup evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load dataset
dataset = PuzzleDataset(
    data_path=Path("data/validation"),
    vocab_size=12,
)

# Evaluate
evaluator = Evaluator(model=model, device=device, config=config)
results = evaluator.evaluate(dataset=dataset)

# Save results
results.save(Path("results/evaluation_results.json"))

# Print metrics
print(f"Solve Rate: {results.aggregate_metrics['solve_rate']:.2%}")
print(f"Average Steps: {results.aggregate_metrics['average_steps']:.2f}")
```

### Ensemble Usage

```python
from hrm_eval.ensemble import EnsembleModel, EnsembleStrategy

# Create ensemble from checkpoints
ensemble = EnsembleModel.create_from_checkpoints(
    checkpoint_paths=[
        "../checkpoints_hrm_v9_optimized_step_7566",
        "../checkpoints_hrm_v9_optimized_step_3607",
    ],
    config=config,
    device=device,
    strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
    weights=[0.6, 0.4],  # Give more weight to later checkpoint
)

# Evaluate ensemble
evaluator = Evaluator(model=ensemble, device=device, config=config)
results = evaluator.evaluate(dataset=dataset)
```

## Data Format

The framework expects puzzle data in JSONL format:

```json
{
  "puzzle_id": 42,
  "input_sequence": [0, 1, 2, 3, 4],
  "target_sequence": [4, 3, 2, 1, 0],
  "solution_steps": [
    {"action": 0, "state": [...]},
    {"action": 1, "state": [...]}
  ],
  "metadata": {
    "difficulty": "medium",
    "category": "sorting"
  }
}
```

**Note**: If no data file exists, the system will create mock data for testing purposes.

## Metrics

### Aggregate Metrics

- **solve_rate**: Percentage of puzzles solved successfully
- **average_steps**: Mean number of steps across all puzzles
- **median_steps**: Median number of steps
- **average_accuracy**: Mean token prediction accuracy
- **step_efficiency**: Ratio of optimal steps to actual steps taken
- **average_time**: Mean time per puzzle
- **error_rate**: Percentage of puzzles with errors

### Per-Puzzle Metrics

- **solved**: Boolean indicating if puzzle was solved
- **num_steps**: Number of steps taken
- **time_elapsed**: Time taken in seconds
- **accuracy**: Token prediction accuracy for this puzzle
- **correct_actions**: Number of correct actions
- **final_q_value**: Final Q-value from the RL head
- **trajectory**: Complete trajectory (if enabled)

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suite

```bash
# Model tests
pytest tests/test_model.py -v

# Metrics tests
pytest tests/test_metrics.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Coverage Report

```bash
pytest tests/ --cov=hrm_eval --cov-report=html
```

View the HTML report in `htmlcov/index.html`.

## Checkpoints

### Available Checkpoints

| Checkpoint | Training Steps | Status | Recommendation |
|------------|---------------|---------|----------------|
| step_180   | 180           | Early   | Baseline only  |
| step_904   | 904           | Early   | -              |
| step_1083  | 1,083         | Early   | -              |
| step_1261  | 1,261         | Early   | -              |
| step_3607  | 3,607         | Mid     | Ensemble use   |
| step_4503  | 4,503         | Mid     | -              |
| step_5767  | 5,767         | Late    | -              |
| step_6306  | 6,306         | Late    | -              |
| **step_7566** | **7,566** | **Final** | **Primary** |

### Checkpoint Details

- **Total Parameters**: 27,990,018 (~28M)
- **Model Size**: 106.77 MB (float32)
- **Architecture**: Hierarchical Dual-Level Transformer
- **Puzzle Embeddings**: 95,996 unique puzzles
- **Vocabulary Size**: 12 tokens

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size in eval_config.yaml
   evaluation:
     batch_size: 8  # Reduce from 32
   ```

2. **Checkpoint Loading Fails**
   ```python
   # Check checkpoint path and structure
   from hrm_eval.utils import get_checkpoint_info
   info = get_checkpoint_info(checkpoint)
   print(info)
   ```

3. **Data Format Issues**
   - Ensure JSONL format is correct
   - Check vocabulary size matches (12 tokens)
   - Verify puzzle IDs are within range [0, 95995]

### Debug Mode

Enable debug logging for detailed information:

```bash
python deploy.py \
    --mode evaluate \
    --checkpoint step_7566 \
    --log-level DEBUG
```

## Performance Optimization

### GPU Optimization

```yaml
device:
  type: "cuda"
  device_id: 0
  mixed_precision: true  # Enable for faster inference
```

### Batch Size Tuning

- **GPU (16GB)**: batch_size=32-64
- **GPU (8GB)**: batch_size=16-32
- **CPU**: batch_size=4-8

### Multi-GPU

```python
# Wrap model with DataParallel
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

## Contributing

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all functions/classes
- Keep functions under 50 lines
- Maintain test coverage >80%

### Adding New Features

1. Create feature branch
2. Implement feature with tests
3. Run full test suite
4. Update documentation
5. Submit pull request

## License

Copyright © 2025 Ian Cruickshank. All rights reserved.

## References

- **W&B Sweep**: https://wandb.ai/ianshank-none/hrm-sweep-optimization/sweeps/8wsnage1
- **Checkpoint Analysis**: See `CHECKPOINT_EVALUATION_REPORT.md`

## Support

For issues, questions, or contributions, please contact the development team or open an issue in the repository.

---

**Status**: Production Ready [DONE]  
**Version**: 1.0.0  
**Last Updated**: October 7, 2025

