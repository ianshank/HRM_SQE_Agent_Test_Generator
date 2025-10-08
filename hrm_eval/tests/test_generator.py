"""
Unit tests for test case generator.

Tests generation, post-processing, and coverage analysis.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from ..test_generator import (
    TestCaseGenerator,
    TestCasePostProcessor,
    TestCaseTemplate,
    CoverageAnalyzer,
)
from ..requirements_parser.schemas import (
    TestContext,
    TestType,
    TestCase,
    TestStep,
    ExpectedResult,
    Priority,
)
from ..models import HRMModel


class TestTestCasePostProcessor:
    """Test TestCasePostProcessor class."""
    
    def test_tokens_to_text(self):
        """Test token to text conversion."""
        processor = TestCasePostProcessor()
        
        tokens = [0, 1, 2, 3, 11]
        text = processor._tokens_to_text(tokens)
        
        assert "test" in text
        assert "agent" in text
        assert "data" in text
    
    def test_generate_description(self):
        """Test description generation."""
        processor = TestCasePostProcessor()
        
        context = TestContext(
            story_id="US-001",
            test_type=TestType.POSITIVE,
            requirement_text="Test requirement",
            acceptance_criterion="User can login",
        )
        
        description = processor._generate_description(context, "test security validation")
        
        assert "Verify successful" in description
        assert "User can login" in description
    
    def test_generate_test_steps(self):
        """Test step generation."""
        processor = TestCasePostProcessor()
        
        context = TestContext(
            story_id="US-002",
            test_type=TestType.POSITIVE,
            requirement_text="Test requirement",
            acceptance_criterion="System processes data",
        )
        
        steps = processor._generate_test_steps(context, "data validation integration")
        
        assert len(steps) > 0
        assert all(isinstance(step, TestStep) for step in steps)
        assert steps[0].step_number == 1
    
    def test_generate_expected_results(self):
        """Test expected results generation."""
        processor = TestCasePostProcessor()
        
        context = TestContext(
            story_id="US-003",
            test_type=TestType.NEGATIVE,
            requirement_text="Test requirement",
        )
        
        results = processor._generate_expected_results(context, "security")
        
        assert len(results) > 0
        assert all(isinstance(result, ExpectedResult) for result in results)
        assert any("error" in result.result.lower() for result in results)
    
    def test_determine_priority(self):
        """Test priority determination."""
        processor = TestCasePostProcessor()
        
        security_test = TestCase(
            id="TC-001",
            type=TestType.POSITIVE,
            priority=Priority.P2,
            description="Security validation test",
            preconditions=[],
            test_steps=[TestStep(step_number=1, action="Test")],
            expected_results=[ExpectedResult(result="Pass")],
            labels=["security"],
        )
        
        priority = processor.determine_priority(security_test)
        
        assert priority == Priority.P1


class TestTestCaseTemplate:
    """Test TestCaseTemplate class."""
    
    def test_format_test_case(self):
        """Test test case formatting."""
        template = TestCaseTemplate()
        
        test_case = TestCase(
            id="TC-001",
            type=TestType.POSITIVE,
            priority=Priority.P1,
            description="Test description",
            preconditions=["Precondition 1"],
            test_steps=[
                TestStep(step_number=1, action="Step 1"),
                TestStep(step_number=2, action="Step 2"),
            ],
            expected_results=[
                ExpectedResult(result="Result 1"),
            ],
            labels=["test", "positive"],
        )
        
        formatted = template.format_test_case(test_case, counter=5)
        
        assert formatted["id"] == "TC-005"
        assert formatted["type"] == "positive"
        assert formatted["priority"] == "P1"
        assert len(formatted["test_steps"]) == 2
    
    def test_generate_description_positive(self):
        """Test positive test description generation."""
        template = TestCaseTemplate()
        
        description = template.generate_description(
            TestType.POSITIVE,
            "user authentication",
            "login flow"
        )
        
        assert "successful" in description.lower()
        assert "user authentication" in description
    
    def test_determine_priority_by_rules(self):
        """Test rule-based priority determination."""
        template = TestCaseTemplate()
        
        edge_test = TestCase(
            id="TC-002",
            type=TestType.EDGE,
            priority=Priority.P2,
            description="Edge case test",
            preconditions=[],
            test_steps=[TestStep(step_number=1, action="Test")],
            expected_results=[ExpectedResult(result="Pass")],
            labels=["edge", "boundary"],
        )
        
        priority = template.determine_priority_by_rules(edge_test)
        
        assert priority == Priority.P3


class TestCoverageAnalyzer:
    """Test CoverageAnalyzer class."""
    
    def test_analyze_coverage_basic(self):
        """Test basic coverage analysis."""
        analyzer = CoverageAnalyzer()
        
        test_cases = [
            TestCase(
                id="TC-001",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Test 1",
                preconditions=[],
                test_steps=[TestStep(step_number=1, action="Action")],
                expected_results=[ExpectedResult(result="Result")],
                source_story_id="US-001",
            ),
            TestCase(
                id="TC-002",
                type=TestType.NEGATIVE,
                priority=Priority.P2,
                description="Test 2",
                preconditions=[],
                test_steps=[TestStep(step_number=1, action="Action")],
                expected_results=[ExpectedResult(result="Result")],
                source_story_id="US-001",
            ),
        ]
        
        contexts = [
            TestContext(
                story_id="US-001",
                test_type=TestType.POSITIVE,
                requirement_text="Requirement",
            ),
            TestContext(
                story_id="US-001",
                test_type=TestType.NEGATIVE,
                requirement_text="Requirement",
            ),
        ]
        
        report = analyzer.analyze_coverage(test_cases, contexts)
        
        assert report["total_test_cases"] == 2
        assert report["positive_tests"] == 1
        assert report["negative_tests"] == 1
        assert report["coverage_percentage"] > 0
    
    def test_identify_gaps(self):
        """Test gap identification."""
        analyzer = CoverageAnalyzer()
        
        test_cases = [
            TestCase(
                id="TC-001",
                type=TestType.POSITIVE,
                priority=Priority.P1,
                description="Test",
                preconditions=[],
                test_steps=[TestStep(step_number=1, action="Action")],
                expected_results=[ExpectedResult(result="Result")],
                source_story_id="US-001",
            ),
        ]
        
        contexts = [
            TestContext(story_id="US-001", test_type=TestType.POSITIVE, requirement_text="Req"),
            TestContext(story_id="US-001", test_type=TestType.NEGATIVE, requirement_text="Req"),
            TestContext(story_id="US-002", test_type=TestType.POSITIVE, requirement_text="Req"),
        ]
        
        gaps = analyzer._identify_gaps(test_cases, contexts)
        
        assert len(gaps) > 0
    
    def test_get_recommendations(self):
        """Test recommendations generation."""
        analyzer = CoverageAnalyzer(min_coverage=0.8)
        
        report = {
            "total_test_cases": 5,
            "coverage_percentage": 60.0,
            "missing_test_types": ["edge"],
            "positive_tests": 1,
            "negative_tests": 3,
            "edge_tests": 0,
            "priority_distribution": {"P1": 1, "P2": 3, "P3": 1},
        }
        
        recommendations = analyzer.get_recommendations(report)
        
        assert len(recommendations) > 0
        assert any("coverage" in rec.lower() for rec in recommendations)
        assert any("edge" in rec.lower() for rec in recommendations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestTestCaseGenerator:
    """Test TestCaseGenerator class (requires model)."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock HRM model."""
        model = Mock(spec=HRMModel)
        model.eval = Mock()
        model.to = Mock(return_value=model)
        
        mock_output = {
            "lm_logits": torch.randn(1, 10, 12),
            "q_values": torch.randn(1, 10, 2),
        }
        model.return_value = mock_output
        
        return model
    
    def test_generator_initialization(self, mock_model):
        """Test generator initialization."""
        device = torch.device("cpu")
        config = {"batch_size": 16}
        
        generator = TestCaseGenerator(mock_model, device, config)
        
        assert generator.model == mock_model
        assert generator.device == device
        mock_model.to.assert_called_once_with(device)
        mock_model.eval.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

