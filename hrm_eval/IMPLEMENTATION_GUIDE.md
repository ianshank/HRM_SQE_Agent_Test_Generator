# Requirements to Test Cases - Implementation Guide

## Overview

This system converts structured requirements (Epics and User Stories) into comprehensive test cases using the HRM v9 Optimized model. The implementation follows a modular architecture with **NO hardcoded test generation** - all test cases are generated through the actual HRM model workflow.

## Architecture Components

### 1. Requirements Parser (`requirements_parser/`)
- **Purpose**: Parse and validate structured requirements
- **Key Classes**:
  - `RequirementParser`: Extracts test contexts from requirements
  - `RequirementValidator`: Validates requirement quality
  - `Epic`, `UserStory`, `TestCase`: Pydantic schemas

### 2. Test Generator (`test_generator/`)
- **Purpose**: Generate test cases using HRM model inference
- **Key Classes**:
  - `TestCaseGenerator`: Orchestrates model inference (NO DUMMY LOGIC)
  - `TestCasePostProcessor`: Converts model output to structured test cases
  - `TestCaseTemplate`: Provides formatting templates
  - `CoverageAnalyzer`: Analyzes test coverage

### 3. API Service (`api_service/`)
- **Purpose**: REST API for test case generation
- **Endpoints**:
  - `POST /api/v1/generate-tests`: Generate test cases for an epic
  - `POST /api/v1/batch-generate`: Batch generation
  - `GET /api/v1/health`: Health check

### 4. Fine-Tuning Pipeline (`fine_tuning/`)
- **Purpose**: Improve model on requirements data
- **Key Classes**:
  - `TrainingDataCollector`: Collect training examples
  - `HRMFineTuner`: Fine-tune model
  - `FineTuningEvaluator`: Evaluate improvements

### 5. Agent System Integration (`integration/`)
- **Purpose**: Integrate with multi-agent systems
- **Key Classes**:
  - `AgentSystemAdapter`: Agent mesh adapter
  - `WorkflowConnector`: LangGraph workflow connector

## Usage Examples

### Basic Usage: Generate Test Cases from Epic

```python
import torch
from pathlib import Path
from hrm_eval.models import HRMModel
from hrm_eval.test_generator import TestCaseGenerator
from hrm_eval.requirements_parser import RequirementParser, Epic, UserStory, AcceptanceCriteria
from hrm_eval.utils import load_config, load_checkpoint

# 1. Load HRM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints_hrm_v9_optimized_step_7566"

config = load_config("hrm_eval/configs/model_config.yaml")
model = HRMModel(
    vocab_size=config["model"]["vocab_size"],
    embed_dim=config["model"]["embed_dim"],
    num_h_layers=config["model"]["num_h_layers"],
    num_l_layers=config["model"]["num_l_layers"],
    num_heads=config["model"]["num_heads"],
    mlp_ratio=config["model"]["mlp_ratio"],
    dropout=config["model"]["dropout"],
    puzzle_vocab_size=config["model"]["puzzle_vocab_size"],
    q_head_actions=config["model"]["q_head_actions"],
)

checkpoint = load_checkpoint(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# 2. Initialize generator
gen_config = load_config("hrm_eval/configs/test_generation_config.yaml")
generator = TestCaseGenerator(model, device, gen_config)

# 3. Define requirements
epic = Epic(
    epic_id="EPIC-001",
    title="User Authentication System",
    user_stories=[
        UserStory(
            id="US-001",
            summary="User login functionality",
            description="As a user, I want to log in to the system securely",
            acceptance_criteria=[
                AcceptanceCriteria(criteria="User can login with valid credentials"),
                AcceptanceCriteria(criteria="System rejects invalid credentials"),
                AcceptanceCriteria(criteria="Session is created on successful login"),
            ],
            tech_stack=["FastAPI", "PostgreSQL", "JWT"],
        )
    ],
    tech_stack=["Python", "Docker"],
    architecture="Microservices",
)

# 4. Generate test cases (uses ACTUAL HRM model)
parser = RequirementParser()
test_contexts = parser.extract_test_contexts(epic)
test_cases = generator.generate_test_cases(test_contexts)

# 5. Output results
print(f"Generated {len(test_cases)} test cases:")
for tc in test_cases:
    print(f"  {tc.id}: {tc.description} [{tc.type.value}, {tc.priority.value}]")
```

### API Service Usage

```bash
# Start the API server
uvicorn hrm_eval.api_service.main:app --host 0.0.0.0 --port 8000

# Generate test cases via API
curl -X POST "http://localhost:8000/api/v1/generate-tests" \
  -H "Content-Type: application/json" \
  -d @epic_request.json
```

Example `epic_request.json`:
```json
{
  "epic": {
    "epic_id": "EPIC-001",
    "title": "Media Asset Management",
    "user_stories": [
      {
        "id": "US-001",
        "summary": "Asset ingestion",
        "description": "System ingests media assets via upload",
        "acceptance_criteria": [
          {"criteria": "Assets are verified and assigned unique IDs"},
          {"criteria": "Checksums are computed"}
        ],
        "tech_stack": ["FastAPI", "S3", "FFmpeg"]
      }
    ],
    "tech_stack": ["Python", "RabbitMQ", "PostgreSQL"]
  },
  "options": {
    "test_types": ["positive", "negative", "edge"],
    "min_coverage": 0.8
  }
}
```

### Fine-Tuning the Model

```python
from hrm_eval.fine_tuning import TrainingDataCollector, HRMFineTuner, FineTuningConfig

# 1. Collect training data
collector = TrainingDataCollector(output_dir="training_data")

# Collect from generation results with feedback
collector.collect_from_generation(
    requirements=[epic],
    generated_tests=test_cases,
    feedback=user_feedback,
)

# Augment with existing SQE data
collector.augment_with_sqe_data("sqe_agent_real_data.jsonl")

# Save training data
training_file = collector.save_training_data("fine_tuning_data.jsonl")

# 2. Fine-tune model
config = FineTuningConfig(
    learning_rate=1e-5,
    epochs=3,
    batch_size=16,
)

fine_tuner = HRMFineTuner(model, device, config)
metrics = fine_tuner.fine_tune(training_file)

print(f"Fine-tuning complete: best_val_loss={metrics['best_val_loss']:.4f}")
```

### Agent System Integration

```python
from hrm_eval.integration import AgentSystemAdapter, WorkflowConnector

# 1. Register as agent
adapter = AgentSystemAdapter(generator, agent_id="test_generator")
await adapter.register_as_agent()

# 2. Handle requests from other agents
request = {
    "type": "generate_test_cases",
    "request_id": "req-001",
    "epic": epic.dict(),
}

response = await adapter.handle_agent_request(request)
print(f"Generated {len(response['test_cases'])} test cases")

# 3. Create LangGraph workflow
connector = WorkflowConnector(generator, enable_refinement=True)
workflow_graph = connector.create_workflow_graph()
```

## Testing

Run tests with pytest:

```bash
# All tests
pytest hrm_eval/tests/ -v

# Specific test file
pytest hrm_eval/tests/test_requirement_parser.py -v

# With coverage
pytest hrm_eval/tests/ --cov=hrm_eval --cov-report=html
```

## Configuration

Edit `hrm_eval/configs/test_generation_config.yaml`:

```yaml
generator:
  model_checkpoint: "step_7566"
  batch_size: 16
  max_test_cases_per_story: 5
  test_type_distribution:
    positive: 0.4
    negative: 0.35
    edge: 0.25

api:
  host: "0.0.0.0"
  port: 8000
  rate_limit: "100/minute"

fine_tuning:
  learning_rate: 0.00001
  epochs: 3
  validation_split: 0.2
```

## Key Principles

1. **NO Hardcoded Test Generation**: All test cases are generated through the actual HRM model workflow
2. **Modular Design**: Each component has a single responsibility
3. **Comprehensive Testing**: Unit, integration, and contract tests
4. **Production-Ready**: Logging, monitoring, error handling, rate limiting
5. **Extensible**: Easy to add new test types, priorities, or integrations

## Monitoring

Monitor generation events:

```python
from hrm_eval.utils.monitoring import TestGenerationMonitor, GenerationEvent

monitor = TestGenerationMonitor(log_dir="logs/monitoring")

event = GenerationEvent(
    event_type="generation",
    epic_id=epic.epic_id,
    num_test_cases=len(test_cases),
    generation_time=generation_time,
    coverage=coverage_percentage,
)

monitor.log_generation_event(event)
```

## Troubleshooting

### Model Not Loading
- Verify checkpoint path exists
- Check device compatibility (CUDA vs CPU)
- Ensure model config matches checkpoint

### Low Coverage
- Add more acceptance criteria to user stories
- Adjust `test_type_distribution` in config
- Review generated test contexts

### API Errors
- Check logs in `logs/test_generation.log`
- Verify epic JSON structure matches schema
- Ensure model is initialized (`/api/v1/initialize`)

## Support

For issues, see:
- Implementation logs: `logs/test_generation.log`
- Monitoring events: `logs/monitoring/events.jsonl`
- API documentation: `http://localhost:8000/docs`

