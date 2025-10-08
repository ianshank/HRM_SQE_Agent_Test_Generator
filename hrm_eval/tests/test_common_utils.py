"""
Unit tests for common utilities module.

Tests helper functions for query creation, formatting, and data processing.
"""

import pytest
from pathlib import Path
import tempfile

from hrm_eval.core.common_utils import (
    create_query_from_story,
    create_rag_query_from_context,
    format_context_from_tests,
    create_test_text_representation,
    slice_list_with_config,
    create_timestamped_directory,
    format_separator_line,
    format_section_header,
    extract_checkpoint_step,
    safe_dict_get,
    merge_dictionaries,
    truncate_string,
    batch_list,
    calculate_statistics,
)
from hrm_eval.requirements_parser.schemas import Epic, UserStory, AcceptanceCriteria, TestContext, TestType
from hrm_eval.utils.unified_config import load_system_config


class TestQueryCreation:
    """Test query creation functions."""
    
    def test_create_query_from_story_basic(self):
        """Test basic query creation from story."""
        config = load_system_config()
        
        story = UserStory(
            id="US-001",
            summary="Test Story",
            description="Test description",
            acceptance_criteria=[]
        )
        
        epic = Epic(
            epic_id="EPIC-001",
            title="Test Epic",
            user_stories=[story]
        )
        
        query = create_query_from_story(epic, story, config)
        
        assert "Epic: Test Epic" in query
        assert "Story: Test Story" in query
        assert "Description: Test description" in query
    
    def test_create_query_from_story_with_criteria(self):
        """Test query creation with acceptance criteria."""
        config = load_system_config()
        
        story = UserStory(
            id="US-001",
            summary="Test Story",
            description="Test description",
            acceptance_criteria=[
                AcceptanceCriteria(criteria="Criteria 1"),
                AcceptanceCriteria(criteria="Criteria 2"),
                AcceptanceCriteria(criteria="Criteria 3"),
                AcceptanceCriteria(criteria="Criteria 4"),  # Should be truncated
            ]
        )
        
        epic = Epic(
            epic_id="EPIC-001",
            title="Test Epic",
            user_stories=[story]
        )
        
        query = create_query_from_story(epic, story, config)
        
        assert "Criteria: Criteria 1 Criteria 2 Criteria 3" in query
        assert "Criteria 4" not in query  # Truncated by config
    
    def test_create_rag_query_from_context(self):
        """Test RAG query creation from test context."""
        config = load_system_config()
        
        context = TestContext(
            story_id="US-001",
            requirement_text="Test requirement",
            test_type=TestType.POSITIVE
        )
        
        query = create_rag_query_from_context(context, config)
        
        assert "Requirement: Test requirement" in query
        assert "Type:" in query and "POSITIVE" in query


class TestContextFormatting:
    """Test context formatting functions."""
    
    def test_format_context_from_tests_basic(self):
        """Test basic context formatting."""
        config = load_system_config()
        
        tests = [
            {"description": "Test 1", "type": "positive"},
            {"description": "Test 2", "type": "negative"},
        ]
        
        context = format_context_from_tests(tests, config)
        
        assert "Test 1:" in context
        assert "Test 2:" in context
        assert "Description: Test 1" in context
    
    def test_format_context_with_steps(self):
        """Test context formatting with test steps."""
        config = load_system_config()
        
        tests = [{
            "description": "Test 1",
            "steps": [
                {"action": "Step 1"},
                {"action": "Step 2"},
                {"action": "Step 3"},
                {"action": "Step 4"},  # Should be truncated
            ]
        }]
        
        context = format_context_from_tests(tests, config)
        
        assert "Step 1" in context
        assert "Step 2" in context
        assert "Step 3" in context
        assert "Step 4" not in context  # Truncated by config


class TestTextRepresentation:
    """Test text representation creation."""
    
    def test_create_test_text_representation(self):
        """Test creating text representation of test."""
        config = load_system_config()
        
        test_data = {
            "description": "Test description",
            "type": "positive",
            "test_steps": [
                {"action": "Step 1"},
                {"action": "Step 2"},
            ],
            "expected_results": [
                {"result": "Result 1"},
            ]
        }
        
        text = create_test_text_representation(test_data, config)
        
        assert "Test: Test description" in text
        assert "Type: positive" in text
        assert "Steps:" in text
        assert "Expected:" in text


class TestListSlicing:
    """Test list slicing with configuration."""
    
    def test_slice_list_with_config(self):
        """Test slicing list using config value."""
        config = load_system_config()
        
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        sliced = slice_list_with_config(items, "test_steps_max", config)
        
        assert len(sliced) == 3  # Default value from config
        assert sliced == [1, 2, 3]
    
    def test_slice_list_with_default(self):
        """Test slicing with default value."""
        config = load_system_config()
        
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        sliced = slice_list_with_config(items, "nonexistent_key", config, default=5)
        
        assert len(sliced) == 5
        assert sliced == [1, 2, 3, 4, 5]


class TestDirectoryCreation:
    """Test directory creation utilities."""
    
    def test_create_timestamped_directory(self):
        """Test creating timestamped directory."""
        config = load_system_config()
        config.output.use_timestamps = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_timestamped_directory(
                Path(tmpdir),
                "test_prefix",
                config
            )
            
            assert output_dir.exists()
            assert output_dir.is_dir()
            assert "test_prefix" in output_dir.name
    
    def test_create_directory_without_timestamp(self):
        """Test creating directory without timestamp."""
        config = load_system_config()
        config.output.use_timestamps = False
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = create_timestamped_directory(
                Path(tmpdir),
                "test_prefix",
                config
            )
            
            assert output_dir.exists()
            assert output_dir.name == "test_prefix"


class TestFormatting:
    """Test formatting utilities."""
    
    def test_format_separator_line(self):
        """Test separator line formatting."""
        config = load_system_config()
        
        sep = format_separator_line(config)
        
        assert len(sep) == config.output.formatting_width
        assert sep == config.output.separator_char * config.output.formatting_width
    
    def test_format_section_header(self):
        """Test section header formatting."""
        config = load_system_config()
        
        header = format_section_header("Test Section", config)
        
        assert "Test Section" in header
        assert config.output.separator_char * config.output.formatting_width in header
    
    def test_format_section_header_without_separator(self):
        """Test section header without separator."""
        config = load_system_config()
        
        header = format_section_header("Test Section", config, include_separator=False)
        
        assert "Test Section" in header
        assert header.count(config.output.separator_char) == 0


class TestCheckpointExtraction:
    """Test checkpoint step extraction."""
    
    def test_extract_checkpoint_step_standard(self):
        """Test extracting step from standard name."""
        step = extract_checkpoint_step("checkpoints_hrm_v9_optimized_step_7566")
        
        assert step == 7566
    
    def test_extract_checkpoint_step_simple(self):
        """Test extracting step from simple name."""
        step = extract_checkpoint_step("step_1000")
        
        assert step == 1000
    
    def test_extract_checkpoint_step_none(self):
        """Test extracting step when not present."""
        step = extract_checkpoint_step("checkpoint_best")
        
        assert step is None


class TestDictionaryOperations:
    """Test dictionary operation utilities."""
    
    def test_safe_dict_get_simple(self):
        """Test safe dictionary access."""
        data = {"key1": "value1"}
        
        value = safe_dict_get(data, "key1")
        
        assert value == "value1"
    
    def test_safe_dict_get_nested(self):
        """Test safe nested dictionary access."""
        data = {"level1": {"level2": {"level3": "value"}}}
        
        value = safe_dict_get(data, "level1", "level2", "level3")
        
        assert value == "value"
    
    def test_safe_dict_get_missing(self):
        """Test safe access to missing key."""
        data = {"key1": "value1"}
        
        value = safe_dict_get(data, "missing", default="default")
        
        assert value == "default"
    
    def test_merge_dictionaries_shallow(self):
        """Test shallow dictionary merge."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        
        merged = merge_dictionaries(dict1, dict2)
        
        assert merged == {"a": 1, "b": 3, "c": 4}
    
    def test_merge_dictionaries_deep(self):
        """Test deep dictionary merge."""
        dict1 = {"a": {"x": 1, "y": 2}}
        dict2 = {"a": {"y": 3, "z": 4}}
        
        merged = merge_dictionaries(dict1, dict2, deep=True)
        
        assert merged == {"a": {"x": 1, "y": 3, "z": 4}}


class TestStringOperations:
    """Test string operation utilities."""
    
    def test_truncate_string_no_truncation(self):
        """Test string that doesn't need truncation."""
        text = "Short text"
        
        truncated = truncate_string(text, 20)
        
        assert truncated == "Short text"
    
    def test_truncate_string_with_truncation(self):
        """Test string truncation."""
        text = "This is a very long text that needs truncation"
        
        truncated = truncate_string(text, 20)
        
        assert len(truncated) == 20
        assert truncated.endswith("...")
    
    def test_truncate_string_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "This is a very long text"
        
        truncated = truncate_string(text, 15, suffix=">>")
        
        assert len(truncated) == 15
        assert truncated.endswith(">>")


class TestListOperations:
    """Test list operation utilities."""
    
    def test_batch_list_even_batches(self):
        """Test batching with even division."""
        items = list(range(10))
        
        batches = batch_list(items, batch_size=2)
        
        assert len(batches) == 5
        assert batches[0] == [0, 1]
        assert batches[-1] == [8, 9]
    
    def test_batch_list_uneven_batches(self):
        """Test batching with uneven division."""
        items = list(range(10))
        
        batches = batch_list(items, batch_size=3)
        
        assert len(batches) == 4
        assert batches[-1] == [9]  # Last batch is smaller


class TestStatisticsCalculation:
    """Test statistics calculation."""
    
    def test_calculate_statistics_basic(self):
        """Test basic statistics calculation."""
        results = {
            "test_cases": [1, 2, 3, 4, 5],
            "validation": {
                "total_tests": 10,
                "valid_tests": 8,
            }
        }
        
        stats = calculate_statistics(results)
        
        assert stats["total_tests"] == 5
        assert stats["success_rate"] == 80.0
    
    def test_calculate_statistics_with_rag(self):
        """Test statistics with RAG examples."""
        results = {
            "test_cases": [1, 2, 3],
            "rag_examples": {
                "story1": [1, 2, 3],
                "story2": [4, 5],
            }
        }
        
        stats = calculate_statistics(results)
        
        assert stats["rag_examples_retrieved"] == 5
    
    def test_calculate_statistics_empty(self):
        """Test statistics with empty results."""
        results = {}
        
        stats = calculate_statistics(results)
        
        assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

