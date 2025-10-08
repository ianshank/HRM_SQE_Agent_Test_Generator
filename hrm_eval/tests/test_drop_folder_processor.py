"""
Unit tests for Drop Folder Processor.

Tests file validation, processing pipeline, and error handling.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from hrm_eval.drop_folder.processor import DropFolderProcessor, ProcessingResult
from hrm_eval.requirements_parser.schemas import Epic, UserStory, TestCase, TestStep, ExpectedResult


class TestProcessingResult:
    """Test ProcessingResult dataclass."""
    
    def test_processing_result_creation(self):
        """Test creating a processing result."""
        result = ProcessingResult(
            success=True,
            input_file="test.txt",
            output_dir="/output/test",
            test_count=5,
            processing_time=1.5,
            rag_enabled=True,
            retrieved_examples=3
        )
        
        assert result.success is True
        assert result.test_count == 5
        assert result.rag_enabled is True
    
    def test_processing_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ProcessingResult(
            success=True,
            input_file="test.txt",
            output_dir="/output/test",
            test_count=5,
            processing_time=1.5
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['success'] is True
        assert result_dict['test_count'] == 5


class TestDropFolderProcessor:
    """Test suite for DropFolderProcessor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    @pytest.fixture
    def config_file(self, temp_dir):
        """Create test configuration file."""
        config_path = temp_dir / "test_config.yaml"
        config_content = f"""
drop_folder:
  base_path: "{temp_dir / 'drop_folder'}"
  input_dir: "input"
  output_dir: "output"
  archive_dir: "archive"
  processing_dir: "processing"
  errors_dir: "errors"
  watch_interval: 5
  file_extensions: [".txt", ".md"]
  debounce_delay: 2
  max_file_size_mb: 10
  use_rag: false
  top_k_similar: 5
  model_checkpoint: "checkpoints_hrm_v9_optimized_step_7566"
  use_fine_tuned: false
  save_json: true
  save_markdown: true
  generate_report: true
  rate_limit_per_minute: 10
"""
        config_path.write_text(config_content)
        
        # Create folder structure
        base = temp_dir / 'drop_folder'
        for subdir in ['input', 'output', 'archive', 'processing', 'errors']:
            (base / subdir).mkdir(parents=True, exist_ok=True)
        
        return str(config_path)
    
    def test_processor_initialization(self, config_file):
        """Test processor initialization."""
        processor = DropFolderProcessor(config_file)
        
        assert processor.config is not None
        assert processor.base_path.exists()
        assert processor.input_dir.exists()
        assert processor.device is not None
    
    def test_load_config(self, config_file):
        """Test configuration loading."""
        processor = DropFolderProcessor(config_file)
        config = processor.config
        
        assert 'base_path' in config
        assert 'input_dir' in config
        assert config['use_rag'] is False
    
    def test_validate_file_valid(self, config_file, temp_dir):
        """Test file validation with valid file."""
        processor = DropFolderProcessor(config_file)
        
        # Create valid test file
        test_file = processor.input_dir / "test.txt"
        test_file.write_text("Test content")
        
        # Should not raise
        processor._validate_file(test_file)
    
    def test_validate_file_not_found(self, config_file):
        """Test file validation with non-existent file."""
        processor = DropFolderProcessor(config_file)
        
        non_existent = processor.input_dir / "does_not_exist.txt"
        
        with pytest.raises(FileNotFoundError):
            processor._validate_file(non_existent)
    
    def test_validate_file_invalid_extension(self, config_file, temp_dir):
        """Test file validation with invalid extension."""
        processor = DropFolderProcessor(config_file)
        
        # Create file with invalid extension
        test_file = processor.input_dir / "test.pdf"
        test_file.write_text("Test content")
        
        with pytest.raises(ValueError, match="Invalid file extension"):
            processor._validate_file(test_file)
    
    def test_validate_file_too_large(self, config_file):
        """Test file validation with oversized file."""
        processor = DropFolderProcessor(config_file)
        
        # Create large file (> 10MB)
        large_file = processor.input_dir / "large.txt"
        large_file.write_text("x" * (11 * 1024 * 1024))  # 11 MB
        
        with pytest.raises(ValueError, match="File too large"):
            processor._validate_file(large_file)
        
        # Cleanup
        large_file.unlink()
    
    @patch('hrm_eval.drop_folder.processor.parse_natural_language_requirements')
    @patch.object(DropFolderProcessor, 'initialize_models')
    @patch.object(DropFolderProcessor, '_generate_tests')
    @patch.object(DropFolderProcessor, '_save_outputs')
    def test_process_file_success(
        self,
        mock_save_outputs,
        mock_generate_tests,
        mock_init_models,
        mock_parse_nl,
        config_file
    ):
        """Test successful file processing."""
        # Setup mocks
        mock_epic = Mock(spec=Epic)
        mock_epic.title = "Test Epic"
        mock_epic.user_stories = [Mock(spec=UserStory)]
        mock_parse_nl.return_value = mock_epic
        
        mock_test_case = Mock(spec=TestCase)
        mock_generate_tests.return_value = ([mock_test_case], {'rag_enabled': False})
        mock_save_outputs.return_value = Path("/output/test")
        
        # Create processor and test file
        processor = DropFolderProcessor(config_file)
        test_file = processor.input_dir / "test.txt"
        test_file.write_text("Epic: Test\nRequirement 1")
        
        # Process file
        result = processor.process_file(test_file)
        
        assert result.success is True
        assert result.test_count == 1
        assert result.processing_time > 0
        
        # Verify file was archived
        archived = processor.archive_dir / "test.txt"
        assert archived.exists()
    
    @patch('hrm_eval.drop_folder.processor.parse_natural_language_requirements')
    def test_process_file_parse_error(self, mock_parse_nl, config_file):
        """Test file processing with parsing error."""
        # Setup mock to raise error
        mock_parse_nl.side_effect = ValueError("Parse error")
        
        # Create processor and test file
        processor = DropFolderProcessor(config_file)
        test_file = processor.input_dir / "bad.txt"
        test_file.write_text("Invalid content")
        
        # Process file
        result = processor.process_file(test_file)
        
        assert result.success is False
        assert "Parse error" in result.error_message
        
        # Verify file was moved to errors
        error_file = processor.errors_dir / "bad.txt"
        assert error_file.exists()
    
    @patch.object(DropFolderProcessor, 'initialize_models')
    @patch.object(DropFolderProcessor, 'process_file')
    def test_process_all_pending(self, mock_process_file, mock_init_models, config_file):
        """Test processing all pending files."""
        # Create mock result
        mock_result = ProcessingResult(
            success=True,
            input_file="test.txt",
            output_dir="/output",
            test_count=3,
            processing_time=1.0
        )
        mock_process_file.return_value = mock_result
        
        # Create processor and test files
        processor = DropFolderProcessor(config_file)
        (processor.input_dir / "test1.txt").write_text("Test 1")
        (processor.input_dir / "test2.txt").write_text("Test 2")
        
        # Process all
        results = processor.process_all_pending()
        
        assert len(results) == 2
        assert all(r.success for r in results)
        assert mock_process_file.call_count == 2
    
    def test_process_all_pending_no_files(self, config_file):
        """Test processing with no pending files."""
        processor = DropFolderProcessor(config_file)
        
        results = processor.process_all_pending()
        
        assert len(results) == 0
    
    def test_save_outputs_structure(self, config_file, temp_dir):
        """Test output directory structure."""
        from datetime import datetime
        from hrm_eval.requirements_parser.schemas import AcceptanceCriteria
        
        processor = DropFolderProcessor(config_file)
        
        # Create test data
        epic = Epic(
            epic_id="EPIC-TEST",
            title="Test Epic",
            user_stories=[
                UserStory(
                    id="US-001",
                    summary="Test story",
                    description="Description",
                    acceptance_criteria=[AcceptanceCriteria(criteria="Test AC")]
                )
            ]
        )
        
        test_case = TestCase(
            id="TC-001",
            description="Test case",
            preconditions=["Precondition 1"],
            test_steps=[TestStep(step_number=1, action="Do something")],
            expected_results=[ExpectedResult(result="Expected result")],
            priority="P1",
            type="positive"
        )
        
        rag_stats = {'rag_enabled': False}
        
        # Save outputs
        output_path = processor._save_outputs(
            "test.txt",
            epic,
            [test_case],
            rag_stats,
            datetime.now()
        )
        
        assert output_path.exists()
        assert (output_path / "test_cases.json").exists()
        assert (output_path / "test_cases.md").exists()
        assert (output_path / "generation_report.md").exists()
        assert (output_path / "metadata.json").exists()


class TestProcessorIntegration:
    """Integration tests for processor (without actual model loading)."""
    
    @pytest.fixture
    def integration_setup(self):
        """Setup for integration tests."""
        temp_path = Path(tempfile.mkdtemp())
        
        # Create config
        config_path = temp_path / "config.yaml"
        config_content = f"""
drop_folder:
  base_path: "{temp_path / 'drop_folder'}"
  input_dir: "input"
  output_dir: "output"
  archive_dir: "archive"
  processing_dir: "processing"
  errors_dir: "errors"
  file_extensions: [".txt"]
  max_file_size_mb: 5
  use_rag: false
  save_json: true
  save_markdown: true
  generate_report: true
"""
        config_path.write_text(config_content)
        
        # Create folders
        base = temp_path / 'drop_folder'
        for subdir in ['input', 'output', 'archive', 'processing', 'errors']:
            (base / subdir).mkdir(parents=True, exist_ok=True)
        
        yield temp_path, str(config_path)
        
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    def test_end_to_end_validation_only(self, integration_setup):
        """Test end-to-end validation without model."""
        temp_path, config_path = integration_setup
        
        processor = DropFolderProcessor(config_path)
        
        # Create test file
        test_file = processor.input_dir / "valid.txt"
        test_file.write_text("Epic: Test\nRequirement")
        
        # Validate (should not raise)
        processor._validate_file(test_file)
