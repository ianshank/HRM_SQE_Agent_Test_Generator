# Drop Folder User Guide

## Overview

The Drop Folder System provides automated test case generation from natural language requirements. Simply drop a requirement file into the input folder, and the system automatically generates comprehensive test cases using the RAG+HRM hybrid workflow.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Setup](#setup)
3. [Usage Modes](#usage-modes)
4. [Requirement File Format](#requirement-file-format)
5. [Output Format](#output-format)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Features](#advanced-features)

---

## Quick Start

### 1. Setup

Run the setup script to create the folder structure:

```bash
./setup_drop_folder.sh
```

Or manually:

```bash
python -m hrm_eval.drop_folder setup
```

### 2. Create a Requirement File

Create a `.txt` or `.md` file in `drop_folder/input/`:

```bash
cat > drop_folder/input/my_feature.txt << 'EOF'
Epic: User Authentication

As a user, I want to log in with my email and password
So that I can access my account

AC: Email must be in valid format
AC: Password must be at least 8 characters
AC: Failed login shows error message
EOF
```

### 3. Generate Tests

**Option A: Process Once**
```bash
python -m hrm_eval.drop_folder process
```

**Option B: Watch Mode (Continuous)**
```bash
python -m hrm_eval.drop_folder watch
```

### 4. View Results

```bash
ls drop_folder/output/
```

Output structure:
```
output/20251008_143022_my_feature/
├── test_cases.json          # Structured test cases
├── test_cases.md            # Readable documentation
├── generation_report.md     # Processing metadata
└── metadata.json            # Technical details
```

---

## Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch
- HRM model checkpoint
- (Optional) watchdog library for file monitoring

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd hrm_train_us_central1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r hrm_eval/requirements.txt
   ```

3. **Run setup:**
   ```bash
   ./setup_drop_folder.sh
   ```

### Directory Structure

After setup, you'll have:

```
drop_folder/
├── input/          # Drop requirement files here
├── processing/     # Files being processed (temporary)
├── output/         # Generated test cases
├── archive/        # Successfully processed files
├── errors/         # Failed files with error logs
└── README.md       # Quick reference
```

---

## Usage Modes

### 1. Manual Processing (Batch Mode)

Process all pending files once:

```bash
python -m hrm_eval.drop_folder process
```

**Use case:** Manual review workflow, scheduled batch jobs

**Output:**
```
============================================================
Processing Complete
============================================================
Total Files: 3
Successful: 3
Failed: 0
============================================================

✓ auth_requirements.txt
   Output: drop_folder/output/20251008_143022_auth_requirements
   Tests: 12
   Time: 3.45s
```

### 2. Watch Mode (Continuous)

Monitor folder and process files automatically:

```bash
python -m hrm_eval.drop_folder watch
```

**Use case:** Real-time processing, integration with CI/CD

**Features:**
- Automatic file detection
- Debouncing (waits for complete upload)
- Graceful shutdown (Ctrl+C)

### 3. Single File Processing

Process a specific file:

```bash
python -m hrm_eval.drop_folder process-file drop_folder/input/feature.txt
```

**Use case:** Testing, debugging, specific file processing

---

## Requirement File Format

### Supported Formats

The NL parser supports multiple formats:

#### 1. User Story Format (Recommended)

```text
Epic: Feature Name

User Story 1: Story Title
As a [role], I want [feature]
So that [benefit]

AC: Acceptance criterion 1
AC: Acceptance criterion 2
AC: Acceptance criterion 3

User Story 2: Another Story
As a [role], I want [feature]

AC: Criterion 1
AC: Criterion 2
```

#### 2. Gherkin Style

```text
Epic: Feature Name

Scenario: Login with valid credentials
Given I am on the login page
When I enter valid email and password
Then I should be logged in
And dashboard should be displayed

Scenario: Login with invalid credentials
Given I am on the login page
When I enter invalid credentials
Then I should see an error message
```

#### 3. Numbered Requirements

```text
# Payment Processing System

1. Process credit card payments
   - Validate card number using Luhn algorithm
   - Check expiration date is future date
   - Verify CVV is 3-4 digits

2. Handle payment failures
   - Log error details for debugging
   - Display user-friendly error message
   - Retry failed transactions up to 3 times
```

#### 4. Markdown Format

```markdown
# Epic: Shopping Cart

## User Story: Add Item to Cart
As a shopper, I want to add items to my cart

**Acceptance Criteria:**
- Item shows correct price
- Quantity can be adjusted
- Cart total updates automatically

## User Story: Remove Item
As a shopper, I want to remove items

**Acceptance Criteria:**
- Remove button available for each item
- Confirmation prompt before removal
```

#### 5. Free Text (Fallback)

If no structure is detected, the entire text becomes a single user story:

```text
The system should allow users to log in with email and password.
Users must provide a valid email address.
Password must be at least 8 characters long.
Failed login attempts should be logged.
```

### Best Practices

1. **Be Specific:** Provide clear acceptance criteria
2. **One Epic Per File:** Keep requirements focused
3. **Use Consistent Format:** Stick to one format per file
4. **Include Context:** Explain the "why" not just the "what"
5. **List Edge Cases:** Include boundary conditions and error scenarios

---

## Output Format

### Generated Files

Each processed requirement generates 4 files:

#### 1. `test_cases.json`

Structured JSON for programmatic access:

```json
[
  {
    "id": "TC-001",
    "description": "Verify successful login with valid credentials",
    "type": "functional",
    "priority": "high",
    "preconditions": [
      "User account exists in system",
      "User is on login page"
    ],
    "test_steps": [
      {
        "step_number": 1,
        "action": "Enter valid email address"
      },
      {
        "step_number": 2,
        "action": "Enter correct password"
      },
      {
        "step_number": 3,
        "action": "Click 'Login' button"
      }
    ],
    "expected_results": [
      {
        "result": "User is successfully authenticated"
      },
      {
        "result": "User is redirected to dashboard"
      }
    ]
  }
]
```

#### 2. `test_cases.md`

Human-readable Markdown:

```markdown
# Test Cases: User Authentication

**Generated:** 2025-10-08 14:30:22
**Total Test Cases:** 12

---

## Functional Tests (8)

### 1. TC-001: Verify successful login with valid credentials

**Priority:** high
**Type:** functional

**Preconditions:**
- User account exists in system
- User is on login page

**Test Steps:**
1. Enter valid email address
2. Enter correct password
3. Click 'Login' button

**Expected Results:**
- User is successfully authenticated
- User is redirected to dashboard

---
```

#### 3. `generation_report.md`

Processing metadata and statistics:

```markdown
# Test Generation Report

**Generated:** 2025-10-08 14:30:22
**Processing Time:** 3.45 seconds

---

## Epic Summary

**Title:** User Authentication
**User Stories:** 3
**Total Test Cases Generated:** 12

## Test Case Breakdown

### By Type
- **functional:** 8
- **negative:** 3
- **edge_case:** 1

### By Priority
- **high:** 5
- **medium:** 6
- **low:** 1

## RAG Enhancement

**Status:** Enabled
**Total Similar Tests Retrieved:** 15
**Retrieval Operations:** 3
**Average per Story:** 5.0

## Quality Metrics

- **Average Steps per Test:** 4.2
- **Average Preconditions per Test:** 2.1
- **Average Expected Results per Test:** 2.3
- **Generation Speed:** 3.48 tests/second
```

#### 4. `metadata.json`

Technical details:

```json
{
  "input_file": "auth_requirements.txt",
  "timestamp": "2025-10-08T14:30:22.123456",
  "epic_title": "User Authentication",
  "user_stories_count": 3,
  "test_cases_count": 12,
  "processing_time_seconds": 3.45,
  "rag_enabled": true,
  "retrieved_examples": 15,
  "model_checkpoint": "fine_tuned_checkpoints/media_fulfillment/checkpoint_epoch_3_best.pt"
}
```

---

## Configuration

### Configuration File

Located at: `hrm_eval/configs/drop_folder_config.yaml`

```yaml
drop_folder:
  # Paths
  base_path: "./drop_folder"
  input_dir: "input"
  output_dir: "output"
  archive_dir: "archive"
  processing_dir: "processing"
  errors_dir: "errors"
  
  # File monitoring
  watch_interval: 5              # seconds between scans
  file_extensions: [".txt", ".md"]
  debounce_delay: 2              # wait time after file write
  max_file_size_mb: 10           # maximum file size
  
  # Processing
  use_rag: true                  # enable RAG enhancement
  top_k_similar: 5               # number of similar tests to retrieve
  min_similarity: 0.5            # minimum similarity threshold
  
  # Model
  model_checkpoint: "fine_tuned_checkpoints/media_fulfillment/checkpoint_epoch_3_best.pt"
  fallback_checkpoint: "checkpoints_hrm_v9_optimized_step_7566"
  use_fine_tuned: true
  
  # Output
  save_json: true
  save_markdown: true
  generate_report: true
  include_metadata: true
  
  # Security
  rate_limit_per_minute: 10
  validate_file_content: true
  
  # Logging
  log_level: "INFO"
  log_to_file: true
  log_file: "drop_folder/drop_folder.log"
```

### CLI Options

```bash
# Set custom config
python -m hrm_eval.drop_folder --config /path/to/config.yaml process

# Set log level
python -m hrm_eval.drop_folder --log-level DEBUG process

# Watch with custom interval
python -m hrm_eval.drop_folder watch --interval 10
```

---

## Troubleshooting

### Common Issues

#### 1. No tests generated

**Problem:** File processed but no test cases created

**Solutions:**
- Check that requirements are clearly written
- Ensure acceptance criteria are present
- Review logs: `drop_folder/drop_folder.log`
- Try more explicit user story format

#### 2. File moved to errors/

**Problem:** Processing failed

**Solutions:**
- Check error log: `drop_folder/errors/filename.txt.error.log`
- Verify file format (`.txt` or `.md`)
- Check file size (< 10MB)
- Validate file content is UTF-8 encoded

#### 3. watchdog not found warning

**Problem:** "watchdog library not available" message

**Solutions:**
```bash
pip install watchdog>=3.0.0
```
Or: System will use polling mode (slightly slower but functional)

#### 4. Model not loading

**Problem:** "Model checkpoint not found"

**Solutions:**
- Verify checkpoint path in config
- Check that fine-tuned model exists
- System will fallback to base model automatically

#### 5. Rate limit exceeded

**Problem:** "Rate limit per minute exceeded"

**Solutions:**
- Increase `rate_limit_per_minute` in config
- Process files in smaller batches
- Use batch mode instead of watch mode

### Debug Mode

Enable debug logging:

```bash
python -m hrm_eval.drop_folder --log-level DEBUG process
```

View detailed logs:

```bash
tail -f drop_folder/drop_folder.log
```

---

## Advanced Features

### 1. RAG-Enhanced Generation

RAG (Retrieval-Augmented Generation) retrieves similar test cases from the vector store to improve quality:

**Enable:**
```yaml
use_rag: true
top_k_similar: 5
min_similarity: 0.5
```

**Benefits:**
- Higher quality test cases
- Better coverage
- Consistent patterns
- Learns from existing tests

### 2. Custom Configuration

Create project-specific config:

```bash
cp hrm_eval/configs/drop_folder_config.yaml my_project_config.yaml
# Edit my_project_config.yaml
python -m hrm_eval.drop_folder --config my_project_config.yaml process
```

### 3. Scheduled Processing

Setup cron job for regular processing:

```bash
# Edit crontab
crontab -e

# Add line to process every hour
0 * * * * cd /path/to/project && python -m hrm_eval.drop_folder process
```

### 4. Integration with CI/CD

**GitHub Actions:**

```yaml
name: Generate Tests
on:
  push:
    paths:
      - 'requirements/**'

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r hrm_eval/requirements.txt
      - name: Generate tests
        run: python -m hrm_eval.drop_folder process
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: test-cases
          path: drop_folder/output/
```

### 5. Webhook Notifications

Add webhook support (future enhancement):

```python
# In processor.py
def notify_webhook(result: ProcessingResult):
    requests.post(
        WEBHOOK_URL,
        json={
            'file': result.input_file,
            'success': result.success,
            'test_count': result.test_count
        }
    )
```

---

## API Reference

For programmatic usage:

```python
from hrm_eval.drop_folder import DropFolderProcessor

# Initialize processor
processor = DropFolderProcessor(config_path="config.yaml")

# Process single file
result = processor.process_file(Path("requirements.txt"))

if result.success:
    print(f"Generated {result.test_count} tests")
    print(f"Output: {result.output_dir}")

# Process all pending
results = processor.process_all_pending()
```

---

## Support

For issues or questions:
- Check logs: `drop_folder/drop_folder.log`
- Review error files: `drop_folder/errors/`
- See examples: `drop_folder/README.md`

---

## Change Log

### Version 1.0.0 (2025-10-08)
- Initial release
- Natural language parsing
- RAG-enhanced generation
- Multiple output formats
- Watch mode support
- Comprehensive error handling
