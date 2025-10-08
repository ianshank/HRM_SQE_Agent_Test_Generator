"""
Unit tests for requirement parser.

Tests parsing, validation, and test context extraction.
"""

import pytest
from pydantic import ValidationError

from ..requirements_parser import (
    RequirementParser,
    RequirementValidator,
    Epic,
    UserStory,
    AcceptanceCriteria,
    TestContext,
    TestType,
)
from ..requirements_parser.requirement_validator import RequirementValidationError


class TestRequirementValidator:
    """Test RequirementValidator class."""
    
    def test_validate_epic_success(self):
        """Test successful epic validation."""
        validator = RequirementValidator()
        
        epic = Epic(
            epic_id="EPIC-001",
            title="Test Epic Title",
            user_stories=[
                UserStory(
                    id="US-001",
                    summary="Test user story",
                    description="This is a test user story description",
                    acceptance_criteria=[
                        AcceptanceCriteria(criteria="Criterion 1"),
                        AcceptanceCriteria(criteria="Criterion 2"),
                    ],
                )
            ],
        )
        
        is_valid, issues = validator.validate_epic(epic)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_epic_missing_criteria(self):
        """Test epic validation with missing acceptance criteria."""
        validator = RequirementValidator(strict_mode=False)
        
        epic = Epic(
            epic_id="EPIC-002",
            title="Test Epic",
            user_stories=[
                UserStory(
                    id="US-002",
                    summary="Test story",
                    description="Test description",
                    acceptance_criteria=[],
                )
            ],
        )
        
        is_valid, issues = validator.validate_epic(epic)
        
        assert not is_valid
        assert any("acceptance criteria" in issue.lower() for issue in issues)
    
    def test_validate_json_missing_required_fields(self):
        """Test JSON validation with missing required fields."""
        validator = RequirementValidator()
        
        invalid_json = {
            "epic_id": "EPIC-003",
        }
        
        is_valid, issues = validator.validate_json(invalid_json)
        
        assert not is_valid
        assert any("title" in issue for issue in issues)
        assert any("user_stories" in issue for issue in issues)
    
    def test_check_testability(self):
        """Test testability scoring."""
        validator = RequirementValidator()
        
        epic = Epic(
            epic_id="EPIC-004",
            title="Testable Epic",
            user_stories=[
                UserStory(
                    id="US-004",
                    summary="Well-defined user story with criteria",
                    description="This is a detailed description of the user story requirements",
                    acceptance_criteria=[
                        AcceptanceCriteria(criteria="Criterion 1"),
                        AcceptanceCriteria(criteria="Criterion 2"),
                    ],
                    tech_stack=["Python", "FastAPI"],
                )
            ],
            tech_stack=["Docker", "PostgreSQL"],
        )
        
        score, report = validator.check_testability(epic)
        
        assert score > 0.6
        assert report["total_stories"] == 1
        assert report["stories_with_criteria"] == 1


class TestRequirementParser:
    """Test RequirementParser class."""
    
    def test_parse_epic_valid_json(self):
        """Test parsing valid JSON into Epic."""
        parser = RequirementParser()
        
        epic_data = {
            "epic_id": "EPIC-005",
            "title": "Test Epic",
            "user_stories": [
                {
                    "id": "US-005",
                    "summary": "Test summary",
                    "description": "Test description",
                    "acceptance_criteria": [
                        {"criteria": "Criterion 1"}
                    ],
                }
            ],
        }
        
        epic = parser.parse_epic(epic_data)
        
        assert epic.epic_id == "EPIC-005"
        assert len(epic.user_stories) == 1
        assert epic.user_stories[0].id == "US-005"
    
    def test_parse_epic_invalid_json(self):
        """Test parsing invalid JSON raises error."""
        parser = RequirementParser()
        
        invalid_data = {
            "epic_id": "EPIC-006",
        }
        
        with pytest.raises(RequirementValidationError):
            parser.parse_epic(invalid_data)
    
    def test_extract_test_contexts(self):
        """Test extraction of test contexts from epic."""
        parser = RequirementParser()
        
        epic = Epic(
            epic_id="EPIC-007",
            title="Test Epic",
            user_stories=[
                UserStory(
                    id="US-007",
                    summary="User story 1",
                    description="Description 1",
                    acceptance_criteria=[
                        AcceptanceCriteria(criteria="Criterion 1"),
                    ],
                )
            ],
        )
        
        contexts = parser.extract_test_contexts(epic)
        
        assert len(contexts) > 0
        assert isinstance(contexts[0], TestContext)
        assert contexts[0].story_id == "US-007"
    
    def test_extract_test_contexts_multiple_types(self):
        """Test that multiple test types are generated."""
        parser = RequirementParser()
        
        epic = Epic(
            epic_id="EPIC-008",
            title="Multi-type Epic",
            user_stories=[
                UserStory(
                    id="US-008",
                    summary="Story with criteria",
                    description="Description",
                    acceptance_criteria=[
                        AcceptanceCriteria(criteria="Criterion A"),
                    ],
                )
            ],
        )
        
        contexts = parser.extract_test_contexts(epic)
        
        test_types = set(ctx.test_type for ctx in contexts)
        
        assert TestType.POSITIVE in test_types
        assert TestType.NEGATIVE in test_types
        assert TestType.EDGE in test_types
    
    def test_format_requirement_for_model(self):
        """Test requirement text formatting."""
        parser = RequirementParser()
        
        epic = Epic(
            epic_id="EPIC-009",
            title="Format Test Epic",
            user_stories=[
                UserStory(
                    id="US-009",
                    summary="Format test story",
                    description="Test description",
                    acceptance_criteria=[
                        AcceptanceCriteria(criteria="Test criterion"),
                    ],
                )
            ],
            tech_stack=["Python"],
        )
        
        story = epic.user_stories[0]
        criterion = story.acceptance_criteria[0]
        
        formatted = parser._format_requirement_for_model(
            story, criterion, TestType.POSITIVE, epic
        )
        
        assert "Epic: Format Test Epic" in formatted
        assert "Story: Format test story" in formatted
        assert "Acceptance Criterion: Test criterion" in formatted
        assert "Tech Stack: Python" in formatted
    
    def test_get_coverage_analysis(self):
        """Test coverage analysis."""
        parser = RequirementParser()
        
        epic = Epic(
            epic_id="EPIC-010",
            title="Coverage Test Epic",
            user_stories=[
                UserStory(
                    id="US-010",
                    summary="Story 1",
                    description="Description 1",
                    acceptance_criteria=[
                        AcceptanceCriteria(criteria="Criterion 1"),
                        AcceptanceCriteria(criteria="Criterion 2"),
                    ],
                ),
                UserStory(
                    id="US-011",
                    summary="Story 2",
                    description="Description 2",
                    acceptance_criteria=[
                        AcceptanceCriteria(criteria="Criterion 3"),
                    ],
                ),
            ],
        )
        
        analysis = parser.get_coverage_analysis(epic)
        
        assert "testability_score" in analysis
        assert "total_stories" in analysis
        assert analysis["total_stories"] == 2
        assert "potential_test_contexts" in analysis
        assert analysis["potential_test_contexts"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

