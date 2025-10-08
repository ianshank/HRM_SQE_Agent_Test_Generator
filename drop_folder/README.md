# Drop Folder for Automated Test Generation

## Usage

1. Drop requirement files (.txt or .md) into the `input/` folder
2. Run processing:
   - **Manual**: `python -m hrm_eval.drop_folder process`
   - **Watch Mode**: `python -m hrm_eval.drop_folder watch`

## Directory Structure

- `input/` - Drop requirement files here
- `processing/` - Files being processed (temporary)
- `output/` - Generated test cases (organized by timestamp)
- `archive/` - Successfully processed files
- `errors/` - Failed files with error logs

## Output Format

Each processed file creates a timestamped directory in `output/`:
```
output/20251008_143022_auth_req/
├── test_cases.json          # Structured JSON
├── test_cases.md            # Readable Markdown
├── generation_report.md     # Processing metadata
└── metadata.json            # Processing info
```

## Requirement File Format

Requirements can be written in natural language. The system will detect:
- Epic titles (first line or "Epic:" prefix)
- User stories ("As a...", "US:", numbered items)
- Acceptance criteria ("AC:", "Given/When/Then", bullet points)

See `sample_requirement.txt` for an example.
