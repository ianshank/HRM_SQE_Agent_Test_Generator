"""
Unit tests for test generation quality improvements.

Tests cover:
- Story-scoped ID generation
- Repetitive text cleaning
- Deduplication
- Quality validation
"""

import pytest
from unittest.mock import Mock, MagicMock
import torch

from hrm_eval.test_generator.generator import TestCaseGenerator
from hrm_eval.test_generator.post_processor import TestCasePostProcessor
from hrm_eval.requirements_parser.schemas import (
    TestCase,
    TestStep,
    ExpectedResult,
    TestContext,
    TestType,
    Priority,
)


class TestStoryScopedIDGeneration:
    """Test story-scoped ID generation."""
    
    def test_story_scoped_id_format(self):
        """Test that IDs follow TC-US001-001 format."""
        # Create mock model and config
        model = MagicMock()
        device = torch.device('cpu')
        config = {'generation': {'max_length': 100}}
        
        generator = TestCaseGenerator(model, device, config)
        
        # Create test cases
        test_cases = [
            TestCase(
                id="TC-000",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Test 1",
                preconditions=["Pre 1"],
                test_steps=[TestStep(step_number=1, action="Action 1")],
                expected_results=[ExpectedResult(result="Result 1")],
                test_data="Data 1",
                labels=["label1"],
                source_story_id="US-001"
            ),
            TestCase(
                id="TC-000",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Test 2",
                preconditions=["Pre 2"],
                test_steps=[TestStep(step_number=1, action="Action 2")],
                expected_results=[ExpectedResult(result="Result 2")],
                test_data="Data 2",
                labels=["label2"],
                source_story_id="US-001"
            ),
        ]
        
        result = generator._assign_ids_and_priorities(test_cases, "US-001")
        
        assert result[0].id == "TC-US001-001"
        assert result[1].id == "TC-US001-002"
    
    def test_story_scoped_id_edge_cases(self):
        """Test ID generation handles edge cases correctly."""
        model = MagicMock()
        device = torch.device('cpu')
        config = {'generation': {'max_length': 100}}
        
        generator = TestCaseGenerator(model, device, config)
        
        # Test various story ID formats
        test_cases_empty = [
            TestCase(
                id="TC-000",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Test",
                preconditions=["Pre"],
                test_steps=[TestStep(step_number=1, action="Action")],
                expected_results=[ExpectedResult(result="Result")],
                test_data="Data",
                labels=["label"],
                source_story_id=""
            ),
        ]
        
        # Empty story ID should use default US000
        result = generator._assign_ids_and_priorities(test_cases_empty, "")
        assert result[0].id == "TC-US000-001"
        
        # Non-numeric story ID should use fallback
        test_cases_custom = [
            TestCase(
                id="TC-000",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Test",
                preconditions=["Pre"],
                test_steps=[TestStep(step_number=1, action="Action")],
                expected_results=[ExpectedResult(result="Result")],
                test_data="Data",
                labels=["label"],
                source_story_id="CUSTOM"
            ),
        ]
        
        result = generator._assign_ids_and_priorities(test_cases_custom, "CUSTOM")
        assert result[0].id == "TC-CUSTOM-001"
    
    def test_story_scoped_id_different_stories(self):
        """Test that different stories get different ID scopes."""
        model = MagicMock()
        device = torch.device('cpu')
        config = {'generation': {'max_length': 100}}
        
        generator = TestCaseGenerator(model, device, config)
        
        # Create test cases for US-001
        test_cases_us1 = [
            TestCase(
                id="TC-000",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Test 1",
                preconditions=["Pre 1"],
                test_steps=[TestStep(step_number=1, action="Action 1")],
                expected_results=[ExpectedResult(result="Result 1")],
                test_data="Data 1",
                labels=["label1"],
                source_story_id="US-001"
            ),
        ]
        
        # Create test cases for US-002
        test_cases_us2 = [
            TestCase(
                id="TC-000",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Test 2",
                preconditions=["Pre 2"],
                test_steps=[TestStep(step_number=1, action="Action 2")],
                expected_results=[ExpectedResult(result="Result 2")],
                test_data="Data 2",
                labels=["label2"],
                source_story_id="US-002"
            ),
        ]
        
        result1 = generator._assign_ids_and_priorities(test_cases_us1, "US-001")
        result2 = generator._assign_ids_and_priorities(test_cases_us2, "US-002")
        
        assert result1[0].id == "TC-US001-001"
        assert result2[0].id == "TC-US002-001"
        assert result1[0].id != result2[0].id


class TestRepetitiveTextCleaning:
    """Test removal of repetitive words."""
    
    def test_consecutive_duplicate_removal(self):
        """Test removal of consecutive duplicate words."""
        processor = TestCasePostProcessor()
        
        text = "integration integration test test for: requirement"
        cleaned = processor._clean_repetitive_text(text)
        
        # Consecutive duplicates should be removed
        assert "integration integration" not in cleaned
        assert "test test" not in cleaned
        # But if duplicates are not consecutive, they remain
        assert "integration" in cleaned
        assert "test" in cleaned
    
    def test_truncated_word_removal(self):
        """Test removal of very short truncated words."""
        processor = TestCasePostProcessor()
        
        # Test with very short truncated word (< 3 chars)
        text = "Verify successful ab cd test for requirement"
        cleaned = processor._clean_repetitive_text(text)
        
        # Short words "ab" and "cd" should be removed (less than 3 chars and not in common_short set)
        assert "ab" not in cleaned
        assert "cd" not in cleaned
        # But normal words should remain
        assert "Verify" in cleaned
        assert "successful" in cleaned
        assert "test" in cleaned
    
    def test_common_short_words_preserved(self):
        """Test that common short words are preserved."""
        processor = TestCasePostProcessor()
        
        text = "Test for the requirement in a system"
        cleaned = processor._clean_repetitive_text(text)
        
        # Common short words should remain
        assert "for" in cleaned
        assert "the" in cleaned
        assert "in" in cleaned
        assert "a" in cleaned


class TestDeduplication:
    """Test deduplication functionality."""
    
    def test_duplicate_removal(self):
        """Test that duplicate tests are removed."""
        processor = TestCasePostProcessor()
        
        # Create duplicate test cases
        test_cases = [
            TestCase(
                id="TC-001",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Verify successful operation",
                preconditions=["Pre 1"],
                test_steps=[TestStep(step_number=1, action="Execute operation")],
                expected_results=[ExpectedResult(result="Success")],
                test_data="Data 1",
                labels=["label1"],
                source_story_id="US-001"
            ),
            TestCase(
                id="TC-002",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Verify successful operation",
                preconditions=["Pre 1"],
                test_steps=[TestStep(step_number=1, action="Execute operation")],
                expected_results=[ExpectedResult(result="Success")],
                test_data="Data 1",
                labels=["label1"],
                source_story_id="US-001"
            ),
            TestCase(
                id="TC-003",
                type=TestType.NEGATIVE,
                priority=Priority.P2,
                description="Verify error handling",
                preconditions=["Pre 2"],
                test_steps=[TestStep(step_number=1, action="Trigger error")],
                expected_results=[ExpectedResult(result="Error handled")],
                test_data="Data 2",
                labels=["label2"],
                source_story_id="US-001"
            ),
        ]
        
        result = processor.deduplicate_tests(test_cases)
        
        # Should remove one duplicate
        assert len(result) == 2
        assert result[0].id == "TC-001"
        assert result[1].id == "TC-003"
    
    def test_no_duplicates(self):
        """Test that unique tests are preserved."""
        processor = TestCasePostProcessor()
        
        test_cases = [
            TestCase(
                id="TC-001",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Test A",
                preconditions=["Pre A"],
                test_steps=[TestStep(step_number=1, action="Action A")],
                expected_results=[ExpectedResult(result="Result A")],
                test_data="Data A",
                labels=["labelA"],
                source_story_id="US-001"
            ),
            TestCase(
                id="TC-002",
                type=TestType.NEGATIVE,
                priority=Priority.P2,
                description="Test B",
                preconditions=["Pre B"],
                test_steps=[TestStep(step_number=1, action="Action B")],
                expected_results=[ExpectedResult(result="Result B")],
                test_data="Data B",
                labels=["labelB"],
                source_story_id="US-001"
            ),
        ]
        
        result = processor.deduplicate_tests(test_cases)
        
        # All tests should be preserved
        assert len(result) == 2


class TestQualityValidation:
    """Test quality validation functionality."""
    
    def test_consecutive_duplicate_words_detection(self):
        """Test detection of consecutive duplicate words."""
        processor = TestCasePostProcessor()
        
        # Test with consecutive duplicates
        test_case = TestCase(
            id="TC-001",
            type=TestType.POSITIVE,
            priority=Priority.P1,
            description="integration integration test test",
            preconditions=["Pre 1"],
            test_steps=[TestStep(step_number=1, action="Action 1")],
            expected_results=[ExpectedResult(result="Result 1")],
            test_data="Data 1",
            labels=["label1"],
            source_story_id="US-001"
        )
        
        is_valid, issues = processor.validate_test_quality(test_case)
        
        assert not is_valid
        assert "Consecutive duplicate words in description" in issues
    
    def test_non_consecutive_duplicates_allowed(self):
        """Test that non-consecutive duplicate words are allowed."""
        processor = TestCasePostProcessor()
        
        # Test with non-consecutive duplicates (this is valid!)
        test_case = TestCase(
            id="TC-001",
            type=TestType.POSITIVE,
            priority=Priority.P1,
            description="test user authentication and user permissions",
            preconditions=["System is running", "User account exists"],
            test_steps=[
                TestStep(step_number=1, action="Navigate to login page"),
                TestStep(step_number=2, action="Enter credentials")
            ],
            expected_results=[ExpectedResult(result="Success")],
            test_data="Valid data",
            labels=["auth"],
            source_story_id="US-001"
        )
        
        is_valid, issues = processor.validate_test_quality(test_case)
        
        # Should NOT be flagged for having "user" and "test" repeated non-consecutively
        assert "Consecutive duplicate words" not in str(issues)
    
    def test_truncated_text_detection(self):
        """Test detection of truncated text."""
        processor = TestCasePostProcessor()
        
        test_case = TestCase(
            id="TC-001",
            type=TestType.POSITIVE,
            priority=Priority.P1,
            description="Verify ceptance Criteria",
            preconditions=["Pre 1"],
            test_steps=[TestStep(step_number=1, action="Action 1")],
            expected_results=[ExpectedResult(result="Result 1")],
            test_data="Data 1",
            labels=["label1"],
            source_story_id="US-001"
        )
        
        is_valid, issues = processor.validate_test_quality(test_case)
        
        assert not is_valid
        assert "Truncated text detected" in issues
    
    def test_generic_description_detection(self):
        """Test detection of generic descriptions."""
        processor = TestCasePostProcessor()
        
        test_case = TestCase(
            id="TC-001",
            type=TestType.POSITIVE,
            priority=Priority.P1,
            description="Verify for: requirement",
            preconditions=["Pre 1"],
            test_steps=[TestStep(step_number=1, action="Action 1")],
            expected_results=[ExpectedResult(result="Result 1")],
            test_data="Data 1",
            labels=["label1"],
            source_story_id="US-001"
        )
        
        is_valid, issues = processor.validate_test_quality(test_case)
        
        assert not is_valid
        assert "Too generic - lacks specificity" in issues
    
    def test_valid_test_case(self):
        """Test that valid test cases pass validation."""
        processor = TestCasePostProcessor()
        
        test_case = TestCase(
            id="TC-001",
            type=TestType.POSITIVE,
            priority=Priority.P1,
            description="Verify successful user login with valid credentials",
            preconditions=["System is running", "User account exists"],
            test_steps=[
                TestStep(step_number=1, action="Navigate to login page"),
                TestStep(step_number=2, action="Enter valid credentials"),
            ],
            expected_results=[ExpectedResult(result="User successfully logged in")],
            test_data="Valid user credentials",
            labels=["authentication"],
            source_story_id="US-001"
        )
        
        is_valid, issues = processor.validate_test_quality(test_case)
        
        assert is_valid
        assert len(issues) == 0

