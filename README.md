# HRM Evaluation & Test Generation System

A production-ready system for evaluating Hierarchical Recurrent Models (HRM) and automatically generating test cases from structured requirements using transformer-based neural networks.

## Overview

This repository contains two integrated systems:

1. **HRM Model Evaluation Framework**: Tools for training, evaluating, and deploying HRM v9 Optimized models for puzzle-solving tasks
2. **Requirements-to-Test-Cases System**: AI-powered test case generation from user stories and acceptance criteria using agent-based workflows

## Key Features

- **No Hardcoded Logic**: All test generation uses actual HRM model inference
- **Modular Architecture**: Clean separation between parsing, generation, evaluation, and deployment
- **REST API**: FastAPI service with authentication, rate limiting, and monitoring
- **Agent System Integration**: Multi-agent mesh coordination for collaborative testing
- **Fine-Tuning Pipeline**: Collect feedback and improve model performance over time
- **Comprehensive Testing**: Unit, integration, and end-to-end tests with >95% coverage target

## Project Structure

```
hrm_train_us_central1/
├── hrm_eval/                  # Main evaluation and generation framework
│   ├── models/                # HRM model architecture
│   ├── data/                  # Data loading and preprocessing
│   ├── evaluation/            # Evaluation metrics and framework
│   ├── test_generator/        # Test case generation from requirements
│   ├── requirements_parser/   # Parse and validate requirements
│   ├── api_service/           # REST API for test generation
│   ├── integration/           # Multi-agent system integration
│   ├── fine_tuning/           # Model improvement pipeline
│   ├── orchestration/         # Workflow management
│   ├── rag_vector_store/      # RAG-based retrieval system
│   ├── configs/               # Configuration files
│   └── tests/                 # Comprehensive test suite
├── checkpoints_*_step_7566/   # Latest trained model checkpoint
├── docs/                      # Documentation and summaries
├── analysis/                  # Checkpoint analysis and reports
└── sqe_agent_real_data.jsonl # Training data for test generation

```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd hrm_train_us_central1

# Install dependencies
cd hrm_eval
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Evaluate Model Checkpoint

```bash
cd hrm_eval
python deploy.py \
    --mode evaluate \
    --checkpoint step_7566 \
    --data-path ./data/validation \
    --output-dir ./results
```

### Generate Test Cases from Requirements

```python
from hrm_eval.test_generator import TestCaseGenerator
from hrm_eval.requirements_parser import RequirementParser

# Parse requirements
parser = RequirementParser()
epic = parser.parse_epic(epic_json)

# Generate test cases
generator = TestCaseGenerator(
    model_path="checkpoints_hrm_v9_optimized_step_7566",
    device="cuda"
)
test_cases = generator.generate_test_cases_from_epic(epic)
```

### Run API Server

```bash
cd hrm_eval
uvicorn api_service.main:app --host 0.0.0.0 --port 8000
```

## System Components

### 1. Requirements Parser

Validates and extracts testable contexts from structured requirements:

- **Pydantic Schemas**: Epic, UserStory, AcceptanceCriteria, TestCase
- **Validation**: Testability scoring and quality checks
- **Context Extraction**: Positive, negative, and edge case scenarios

### 2. Test Generator

Generates comprehensive test cases using HRM model inference:

- **No Hardcoding**: Uses actual neural network predictions
- **Post-Processing**: Converts model outputs to structured test cases
- **Coverage Analysis**: Ensures comprehensive test coverage
- **Template Engine**: Formats tests in various styles (Gherkin, pytest, etc.)

### 3. API Service

Production-ready FastAPI service:

- **Authentication**: API key and JWT token support
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Monitoring**: Prometheus metrics and structured logging
- **Documentation**: OpenAPI/Swagger UI at `/docs`

### 4. Multi-Agent Integration

Coordinate multiple AI agents for comprehensive testing:

- **Agent Mesh**: Register as SQE agent in multi-agent systems
- **Workflow Orchestration**: LangGraph-based workflows
- **Consensus Building**: Multi-agent review and approval

### 5. Fine-Tuning Pipeline

Continuously improve model performance:

- **Data Collection**: Gather training examples from user feedback
- **Model Training**: Fine-tune HRM on domain-specific data
- **Evaluation**: Compare base vs fine-tuned performance

## Model Architecture

The HRM v9 Optimized model features:

- **Parameters**: 27,990,018 (~28M)
- **Architecture**: Hierarchical Dual-Level Transformer
- **Layers**: 23 total (2 high-level + 2 low-level transformer layers)
- **Embeddings**: Puzzle-specific (95,996 puzzles) + Token (vocab=12)
- **Best Checkpoint**: `step_7566` (converged, stable weights)

## Configuration

Configuration files are located in `hrm_eval/configs/`:

- `model_config.yaml`: Model architecture and training parameters
- `eval_config.yaml`: Evaluation metrics and thresholds
- `test_generation_config.yaml`: Test generation settings
- `rag_sqe_config.yaml`: RAG retrieval configuration

## Testing

Run the comprehensive test suite:

```bash
cd hrm_eval
pytest tests/ -v --cov=. --cov-report=html
```

Test categories:
- Unit tests: Individual component testing
- Integration tests: End-to-end workflows
- Contract tests: API and interface validation

## Documentation

Detailed documentation available in `docs/`:

- **REQUIREMENTS_TO_TEST_CASES_SUMMARY.md**: Complete implementation details
- **DEPLOYMENT_SUMMARY.md**: Deployment guide and checklist
- **SQE_DATA_EVALUATION_SUMMARY.md**: Training data evaluation

Additional guides in `hrm_eval/`:
- **QUICK_START_GUIDE.md**: Step-by-step tutorial
- **IMPLEMENTATION_GUIDE.md**: Architecture and design decisions
- **API_USAGE_GUIDE.md**: API reference and examples

## Analysis Tools

Checkpoint analysis tools in `analysis/scripts/`:

- `evaluate_checkpoints.py`: Compare multiple checkpoints
- `detailed_checkpoint_analysis.py`: Deep dive into model weights
- `inspect_model_architecture.py`: Visualize architecture

Results stored in `analysis/`:
- Checkpoint comparison tables (CSV/JSON)
- Model architecture visualization
- Training convergence reports

## Deployment

The system is ready for production deployment:

- [x] Modular, reusable architecture
- [x] Comprehensive error handling and logging
- [x] API authentication and rate limiting
- [x] Monitoring and metrics collection
- [x] Docker-ready (see `hrm_eval/deploy.py`)
- [x] CI/CD compatible

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Apache

## Contact
Ian Cruickshank
ianshank@gmai.com