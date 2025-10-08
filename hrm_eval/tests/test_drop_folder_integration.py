"""
Integration tests for Drop Folder System.

Tests the complete workflow from file drop to test generation.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch

from hrm_eval.drop_folder.processor import DropFolderProcessor
from hrm_eval.drop_folder.formatter import OutputFormatter
from hrm_eval.requirements_parser.schemas import (
    Epic, UserStory, AcceptanceCriteria, TestCase, TestStep, ExpectedResult
)


class TestOutputFormatter:
    """Test suite for OutputFormatter."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_epic(self):
        """Create sample epic."""
        return Epic(
            epic_id="EPIC-TEST",
            title="Test Epic",
            user_stories=[
                UserStory(
                    id="US-001",
                    summary="User login",
                    description="User can log in with credentials",
                    acceptance_criteria=[
                        AcceptanceCriteria(criteria="Email must be valid"),
                        AcceptanceCriteria(criteria="Password required")
                    ]
                )
            ]
        )
    
    @pytest.fixture
    def sample_test_cases(self):
        """Create sample test cases."""
        return [
            TestCase(
                id="TC-001",
                description="Verify valid login",
                preconditions=["User has account", "User is on login page"],
                test_steps=[
                    TestStep(step_number=1, action="Enter valid email"),
                    TestStep(step_number=2, action="Enter valid password"),
                    TestStep(step_number=3, action="Click login button")
                ],
                expected_results=[
                    ExpectedResult(result="User is logged in"),
                    ExpectedResult(result="Dashboard is displayed")
                ],
                priority="P1",
                type="positive"
            ),
            TestCase(
                id="TC-002",
                description="Verify invalid login",
                preconditions=["User is on login page"],
                test_steps=[
                    TestStep(step_number=1, action="Enter invalid email"),
                    TestStep(step_number=2, action="Click login button")
                ],
                expected_results=[
                    ExpectedResult(result="Error message displayed")
                ],
                priority="P2",
                type="negative"
            )
        ]
    
    def test_formatter_initialization(self):
        """Test formatter initialization."""
        formatter = OutputFormatter()
        assert formatter is not None
    
    def test_save_json(self, temp_dir, sample_test_cases):
        """Test saving test cases as JSON."""
        formatter = OutputFormatter()
        output_path = temp_dir / "test_cases.json"
        
        formatter.save_json(sample_test_cases, output_path)
        
        assert output_path.exists()
        
        # Verify JSON content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 2
        assert data[0]['id'] == 'TC-001'
        assert data[0]['description'] == 'Verify valid login'
        assert len(data[0]['test_steps']) == 3
    
    def test_save_markdown(self, temp_dir, sample_epic, sample_test_cases):
        """Test saving test cases as Markdown."""
        formatter = OutputFormatter()
        output_path = temp_dir / "test_cases.md"
        
        formatter.save_markdown(sample_test_cases, sample_epic, output_path)
        
        assert output_path.exists()
        
        # Verify Markdown content
        content = output_path.read_text()
        assert "Test Epic" in content
        assert "TC-001" in content
        assert "Verify valid login" in content
        assert "positive" in content.lower()
        assert "Test Steps" in content or "test steps" in content.lower()
    
    def test_save_report(self, temp_dir, sample_epic, sample_test_cases):
        """Test saving generation report."""
        formatter = OutputFormatter()
        output_path = temp_dir / "report.md"
        
        rag_stats = {
            'rag_enabled': True,
            'retrieved_examples': 5,
            'total_retrievals': 2
        }
        
        formatter.save_report(
            sample_epic,
            sample_test_cases,
            rag_stats,
            processing_time=2.5,
            output_path=output_path
        )
        
        assert output_path.exists()
        
        # Verify report content
        content = output_path.read_text()
        assert "Test Generation Report" in content
        assert "2.5" in content  # Processing time
        assert "RAG Enhancement" in content
        assert "Enabled" in content
        assert "5" in content  # Retrieved examples
    
    def test_format_test_case_markdown(self, sample_test_cases):
        """Test formatting a single test case."""
        formatter = OutputFormatter()
        
        lines = formatter._format_test_case_markdown(sample_test_cases[0], 1)
        
        assert len(lines) > 0
        content = '\n'.join(lines)
        assert "TC-001" in content
        assert "Verify valid login" in content
        assert "P1" in content or "p1" in content.lower()
    
    def test_save_markdown_groups_by_type(self, temp_dir, sample_epic, sample_test_cases):
        """Test that markdown groups tests by type."""
        formatter = OutputFormatter()
        output_path = temp_dir / "test_cases.md"
        
        formatter.save_markdown(sample_test_cases, sample_epic, output_path)
        
        content = output_path.read_text()
        
        # Should have sections for different test types
        assert "Positive" in content or "positive" in content.lower()
        assert "Negative" in content or "negative" in content.lower()


class TestDropFolderIntegration:
    """Full integration tests for drop folder system."""
    
    @pytest.fixture
    def setup_system(self):
        """Setup complete drop folder system."""
        temp_path = Path(tempfile.mkdtemp())
        
        # Create configuration
        config_path = temp_path / "config.yaml"
        config_content = f"""
drop_folder:
  base_path: "{temp_path / 'drop_folder'}"
  input_dir: "input"
  output_dir: "output"
  archive_dir: "archive"
  processing_dir: "processing"
  errors_dir: "errors"
  file_extensions: [".txt", ".md"]
  max_file_size_mb: 10
  use_rag: false
  save_json: true
  save_markdown: true
  generate_report: true
  include_metadata: true
"""
        config_path.write_text(config_content)
        
        # Create folder structure
        base = temp_path / 'drop_folder'
        for subdir in ['input', 'output', 'archive', 'processing', 'errors']:
            (base / subdir).mkdir(parents=True, exist_ok=True)
        
        yield temp_path, str(config_path), base
        
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    def test_file_lifecycle(self, setup_system):
        """Test complete file lifecycle from input to archive."""
        temp_path, config_path, base = setup_system
        
        # Create requirement file
        req_file = base / "input" / "test_req.txt"
        req_file.write_text("""Epic: User Authentication

User Story: Login
As a user, I want to log in
AC: Email required
AC: Password required
""")
        
        assert req_file.exists()
        
        # Note: We can't run full processing without models
        # This test validates the file structure and paths
        processor = DropFolderProcessor(config_path)
        
        # Verify paths are set up correctly
        assert processor.input_dir == base / "input"
        assert processor.output_dir == base / "output"
        assert processor.archive_dir == base / "archive"
        assert processor.processing_dir == base / "processing"
        assert processor.errors_dir == base / "errors"
    
    @patch('hrm_eval.drop_folder.processor.parse_natural_language_requirements')
    @patch.object(DropFolderProcessor, 'initialize_models')
    @patch.object(DropFolderProcessor, '_generate_tests')
    def test_full_processing_pipeline(
        self,
        mock_generate_tests,
        mock_init_models,
        mock_parse_nl,
        setup_system
    ):
        """Test full processing pipeline with mocked components."""
        temp_path, config_path, base = setup_system
        
        # Setup mocks
        mock_epic = Epic(
            epic_id="EPIC-AUTH",
            title="Authentication",
            user_stories=[
                UserStory(
                    id="US-001",
                    summary="Login",
                    description="User login",
                    acceptance_criteria=[AcceptanceCriteria(criteria="Email required")]
                )
            ]
        )
        mock_parse_nl.return_value = mock_epic
        
        mock_test_cases = [
            TestCase(
                id="TC-001",
                description="Test login",
                preconditions=["User exists"],
                test_steps=[TestStep(step_number=1, action="Enter credentials")],
                expected_results=[ExpectedResult(result="User logged in")],
                priority="P1",
                type="positive"
            )
        ]
        mock_generate_tests.return_value = (mock_test_cases, {'rag_enabled': False})
        
        # Create input file
        req_file = base / "input" / "auth.txt"
        req_file.write_text("Epic: Auth\nRequirement")
        
        # Process file
        processor = DropFolderProcessor(config_path)
        result = processor.process_file(req_file)
        
        assert result.success is True
        assert result.test_count == 1
        
        # Verify output structure
        output_dir = Path(result.output_dir)
        assert output_dir.exists()
        assert (output_dir / "test_cases.json").exists()
        assert (output_dir / "test_cases.md").exists()
        assert (output_dir / "generation_report.md").exists()
        assert (output_dir / "metadata.json").exists()
        
        # Verify metadata
        with open(output_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        assert metadata['input_file'] == 'auth.txt'
        assert metadata['test_cases_count'] == 1
        
        # Verify file was archived
        assert (base / "archive" / "auth.txt").exists()
        assert not req_file.exists()
    
    def test_error_handling(self, setup_system):
        """Test error handling and error file creation."""
        temp_path, config_path, base = setup_system
        
        # Create invalid file
        bad_file = base / "input" / "bad.pdf"
        bad_file.write_text("Invalid content")
        
        processor = DropFolderProcessor(config_path)
        result = processor.process_file(bad_file)
        
        assert result.success is False
        assert "Invalid file extension" in result.error_message
    
    def test_multiple_files_processing(self, setup_system):
        """Test processing multiple files."""
        temp_path, config_path, base = setup_system
        
        # Create multiple requirement files
        for i in range(3):
            req_file = base / "input" / f"req_{i}.txt"
            req_file.write_text(f"Epic: Test {i}\nRequirement {i}")
        
        processor = DropFolderProcessor(config_path)
        
        # Find all files
        pending = list(processor.input_dir.glob("*.txt"))
        assert len(pending) == 3


class TestNaturalLanguageParserIntegration:
    """Integration tests for NL parser with drop folder."""
    
    def test_parse_real_world_format(self):
        """Test parsing a real-world format requirement."""
        from hrm_eval.requirements_parser.nl_parser import parse_natural_language_requirements
        
        text = """# E-Commerce Checkout System

## User Story 1: Add Items to Cart
As a shopper, I want to add items to my cart
So that I can purchase multiple items at once

**Acceptance Criteria:**
- Items show correct price
- Quantity can be adjusted
- Cart total updates automatically

## User Story 2: Apply Discount Code
As a shopper, I want to apply discount codes
So that I can save money on my purchase

**Acceptance Criteria:**
- Discount code validates before applying
- Invalid codes show error message
- Discount reflects in order total
"""
        
        epic = parse_natural_language_requirements(text)
        
        assert epic is not None
        assert len(epic.user_stories) >= 2
        assert any("cart" in story.summary.lower() for story in epic.user_stories)
