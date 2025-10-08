# Quick Start: Requirements to Test Cases

## Installation

```bash
cd /Users/iancruickshank/Downloads/hrm_train_us_central1/hrm_eval

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Test

```bash
# Run example usage script
python example_usage.py
```

## API Server

```bash
# Start the API server
uvicorn api_service.main:app --host 0.0.0.0 --port 8000

# In another terminal, test it
curl http://localhost:8000/api/v1/health
```

## Create Your First Test Cases

```python
from hrm_eval.requirements_parser import RequirementParser, Epic, UserStory, AcceptanceCriteria

# 1. Define your requirements
epic = Epic(
    epic_id="EPIC-001",
    title="Your Epic Title",
    user_stories=[
        UserStory(
            id="US-001",
            summary="Your user story summary",
            description="Detailed description of the requirement",
            acceptance_criteria=[
                AcceptanceCriteria(criteria="First acceptance criterion"),
                AcceptanceCriteria(criteria="Second acceptance criterion"),
            ],
            tech_stack=["Python", "FastAPI"],  # Optional
        )
    ],
    tech_stack=["Docker", "PostgreSQL"],  # Optional
    architecture="Microservices",  # Optional
)

# 2. Generate test cases (uses ACTUAL HRM model)
from hrm_eval.test_generator import TestCaseGenerator
from hrm_eval.models import HRMModel
import torch

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HRMModel(...)  # Configure from model_config.yaml
model.load_state_dict(torch.load("../checkpoints_hrm_v9_optimized_step_7566"))

# Generate
generator = TestCaseGenerator(model, device, config)
parser = RequirementParser()

test_contexts = parser.extract_test_contexts(epic)
test_cases = generator.generate_test_cases(test_contexts)

# 3. View results
for tc in test_cases:
    print(f"{tc.id}: {tc.description} [{tc.type.value}, {tc.priority.value}]")
```

## Using the API

```bash
# Create a JSON file with your epic
cat > my_epic.json << 'EOF'
{
  "epic": {
    "epic_id": "EPIC-TEST-001",
    "title": "Test Epic",
    "user_stories": [
      {
        "id": "US-001",
        "summary": "User login",
        "description": "As a user, I want to log in securely",
        "acceptance_criteria": [
          {"criteria": "User can login with valid credentials"},
          {"criteria": "System rejects invalid credentials"}
        ]
      }
    ]
  }
}
EOF

# Generate test cases
curl -X POST http://localhost:8000/api/v1/generate-tests \
  -H "Content-Type: application/json" \
  -d @my_epic.json | jq .
```

## Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Configuration

Edit `configs/test_generation_config.yaml` to customize:

- Model checkpoint
- Test type distribution
- Coverage thresholds
- API settings
- Fine-tuning parameters

## Example Input Format

```json
{
  "epic_id": "EPIC-001",
  "title": "Epic Title",
  "user_stories": [
    {
      "id": "US-001",
      "summary": "Story summary",
      "description": "Detailed description",
      "acceptance_criteria": [
        {"criteria": "Criterion 1"},
        {"criteria": "Criterion 2"}
      ],
      "tech_stack": ["FastAPI", "PostgreSQL"]
    }
  ],
  "tech_stack": ["Python", "Docker"],
  "architecture": "Microservices"
}
```

## Example Output

```json
{
  "test_cases": [
    {
      "id": "TC-001",
      "type": "positive",
      "priority": "P1",
      "description": "Verify successful user authentication",
      "preconditions": ["System operational", "Valid credentials available"],
      "test_steps": [
        {"step_number": 1, "action": "Navigate to login page"},
        {"step_number": 2, "action": "Enter valid credentials"},
        {"step_number": 3, "action": "Click login button"}
      ],
      "expected_results": [
        {"result": "User is authenticated"},
        {"result": "Dashboard is displayed"}
      ],
      "test_data": "username: testuser, password: Test123!",
      "labels": ["authentication", "positive", "P1"]
    }
  ],
  "metadata": {
    "model_checkpoint": "step_7566",
    "generation_time_seconds": 2.45,
    "num_test_cases": 1,
    "coverage_score": 0.85
  },
  "coverage_report": {
    "positive_tests": 1,
    "negative_tests": 1,
    "edge_tests": 1,
    "coverage_percentage": 85.0
  }
}
```

## Troubleshooting

### Model not loading
- Verify checkpoint path: `../checkpoints_hrm_v9_optimized_step_7566`
- Check CUDA availability: `torch.cuda.is_available()`

### API not responding
- Check if service is running: `curl http://localhost:8000/api/v1/health`
- Review logs: `tail -f logs/test_generation.log`

### Low coverage
- Add more acceptance criteria to user stories
- Adjust `test_type_distribution` in config
- Check testability score: `parser.validator.check_testability(epic)`

## Next Steps

1. **Fine-tune the model** on your domain-specific data
2. **Integrate with agent systems** for agent-based workflows
3. **Deploy to production** with proper authentication
4. **Monitor performance** using the monitoring utilities
5. **Collect feedback** to improve model over time

## Support

- Documentation: `IMPLEMENTATION_GUIDE.md`
- Full Summary: `REQUIREMENTS_TO_TEST_CASES_SUMMARY.md`
- API Docs: http://localhost:8000/docs (when server running)

