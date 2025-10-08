# HRM v9 Optimized - Quick Start Guide

Get started with evaluating the HRM v9 Optimized model in 5 minutes.

## Step 1: Installation (1 minute)

```bash
cd hrm_eval
pip install -r requirements.txt
```

## Step 2: Evaluate Best Checkpoint (2 minutes)

### Option A: Using Mock Data (for testing setup)

```bash
python deploy.py \
    --mode evaluate \
    --checkpoint step_7566 \
    --device cuda
```

This will:
- Load checkpoint `step_7566` (the best performing checkpoint)
- Generate mock data (since no validation data exists yet)
- Evaluate the model
- Save results to `./results/results_step_7566.json`

### Option B: Using Your Own Data

```bash
python deploy.py \
    --mode evaluate \
    --checkpoint step_7566 \
    --data-path /path/to/your/validation/data.jsonl \
    --device cuda
```

## Step 3: View Results (1 minute)

```bash
# View aggregate metrics
cat results/results_step_7566.json | python -m json.tool | head -30
```

Expected output:
```json
{
  "aggregate_metrics": {
    "total_puzzles": 100,
    "solved_puzzles": 75,
    "solve_rate": 0.75,
    "average_steps": 125.5,
    "average_accuracy": 0.82,
    ...
  },
  ...
}
```

## Step 4 (Optional): Evaluate Ensemble (2 minutes)

```bash
python deploy.py \
    --mode ensemble \
    --checkpoints step_7566 step_3607 \
    --device cuda
```

This combines predictions from two checkpoints (weighted 60/40 by default).

## Step 5 (Optional): Run Tests (1 minute)

```bash
python deploy.py --mode test
```

Or manually:
```bash
pytest tests/ -v
```

## Next Steps

1. **Integrate Your Data**: Replace mock data with actual puzzle data
   - Format: JSONL with fields `puzzle_id`, `input_sequence`, `target_sequence`
   - See `README.md` for complete data format specification

2. **Customize Configuration**: Edit `configs/*.yaml` to adjust:
   - Batch size (for memory management)
   - Evaluation metrics
   - Output settings

3. **Analyze Results**: Use the generated JSON files to:
   - Identify difficult puzzles
   - Analyze failure modes
   - Compare checkpoint performance

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
# Edit configs/eval_config.yaml:
evaluation:
  batch_size: 8  # Default is 32
```

### Checkpoint Not Found
```bash
# Verify checkpoint exists
ls -l ../checkpoints_hrm_v9_optimized_step_7566
```

### CUDA Not Available
```bash
# Use CPU instead
python deploy.py --mode evaluate --checkpoint step_7566 --device cpu
```

## Common Commands

```bash
# Evaluate with custom settings
python deploy.py \
    --mode evaluate \
    --checkpoint step_7566 \
    --model-config configs/model_config.yaml \
    --eval-config configs/eval_config.yaml \
    --output-dir results/run_001 \
    --log-level DEBUG

# Run specific tests
pytest tests/test_model.py -v

# Check test coverage
pytest tests/ --cov=hrm_eval --cov-report=term-missing
```

## Support

- **Full Documentation**: See `README.md`
- **Checkpoint Analysis**: See `../CHECKPOINT_EVALUATION_REPORT.md`
- **Architecture Details**: See `../model_architecture.json`

---

**Need Help?** Check the full README.md for detailed documentation.

