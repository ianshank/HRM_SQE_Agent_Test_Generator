"""
Unit tests for Natural Language Parser.

Tests parsing of various requirement formats into structured Epics.
"""

import pytest
from hrm_eval.requirements_parser.nl_parser import (
    NaturalLanguageParser,
    parse_natural_language_requirements
)
from hrm_eval.requirements_parser.schemas import Epic, UserStory, AcceptanceCriteria


class TestNaturalLanguageParser:
    """Test suite for NaturalLanguageParser."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return NaturalLanguageParser()
    
    def test_parse_basic_epic(self, parser):
        """Test parsing a basic epic with user stories."""
        text = """Epic: User Authentication

User Story 1: Login
As a user, I want to log in with email
AC: Email must be valid
AC: Password required

User Story 2: Logout
As a user, I want to log out
AC: Session cleared
"""
        
        epic = parser.parse(text)
        
        assert isinstance(epic, Epic)
        assert "Authentication" in epic.title
        # Parser may split into multiple stories, verify at least key stories exist
        assert len(epic.user_stories) >= 2
        assert any("login" in story.summary.lower() or "login" in story.description.lower() 
                   for story in epic.user_stories)
        assert any("logout" in story.summary.lower() or "logout" in story.description.lower() 
                   for story in epic.user_stories)
    
    def test_parse_numbered_stories(self, parser):
        """Test parsing numbered user stories."""
        text = """Requirements: Payment System

1. Process credit card payments
   - Validate card number
   - Check expiration date

2. Handle payment failures
   - Log error messages
   - Notify user
"""
        
        epic = parser.parse(text)
        
        assert isinstance(epic, Epic)
        assert len(epic.user_stories) >= 1
    
    def test_parse_gherkin_format(self, parser):
        """Test parsing Gherkin-style acceptance criteria."""
        text = """Epic: Shopping Cart

US: Add to Cart
Given I am viewing a product
When I click "Add to Cart"
Then the product is added to my cart

US: Remove from Cart
Given I have items in my cart
When I click "Remove"
Then the item is removed
"""
        
        epic = parser.parse(text)
        
        assert isinstance(epic, Epic)
        assert len(epic.user_stories) >= 1
        
        # Check for Gherkin-style criteria
        for story in epic.user_stories:
            if story.acceptance_criteria:
                for ac in story.acceptance_criteria:
                    # Should have criteria text
                    assert len(ac.criteria) > 0
    
    def test_parse_markdown_headers(self, parser):
        """Test parsing with markdown headers."""
        text = """# Order Management System

## User Story: Create Order
As a customer, I want to create orders

**Acceptance Criteria:**
- Order must have line items
- Total calculated automatically
- Order confirmation sent

## User Story: Cancel Order
As a customer, I want to cancel orders

**Acceptance Criteria:**
- Only pending orders can be cancelled
- Refund processed automatically
"""
        
        epic = parser.parse(text)
        
        assert isinstance(epic, Epic)
        assert "Order" in epic.title
        assert len(epic.user_stories) >= 1
    
    def test_parse_free_text_fallback(self, parser):
        """Test fallback parsing for unstructured text."""
        text = """This is a free-form requirement.
The system should allow users to log in.
Users must have valid credentials.
"""
        
        epic = parser.parse(text, "test_req.txt")
        
        assert isinstance(epic, Epic)
        assert len(epic.user_stories) == 1  # Fallback creates single story
        assert epic.user_stories[0].id == "US-001"
    
    def test_parse_empty_text_raises_error(self, parser):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Empty text"):
            parser.parse("")
        
        with pytest.raises(ValueError, match="Empty text"):
            parser.parse("   \n\n   ")
    
    def test_extract_epic_title_from_patterns(self, parser):
        """Test epic title extraction from various patterns."""
        # Test "Epic:" prefix
        title = parser._extract_epic_title("Epic: My Epic Title\nContent...", None)
        assert title == "My Epic Title"
        
        # Test markdown header
        title = parser._extract_epic_title("# Payment System\nContent...", None)
        assert title == "Payment System"
        
        # Test first line fallback
        title = parser._extract_epic_title("This is the title\nMore content", None)
        assert title == "This is the title"
    
    def test_extract_acceptance_criteria(self, parser):
        """Test acceptance criteria extraction."""
        lines = [
            "AC: Password must be strong",
            "- Email must be unique",
            "* Username required",
            "Given user has account When they login Then dashboard shown"
        ]
        
        criteria = parser._extract_acceptance_criteria(lines)
        
        assert len(criteria) >= 3  # At least the AC and bullet points
        assert all(isinstance(ac, AcceptanceCriteria) for ac in criteria)
    
    def test_generate_epic_id(self, parser):
        """Test epic ID generation."""
        # From title
        epic_id = parser._generate_epic_id("User Authentication System", None)
        assert epic_id.startswith("EPIC-")
        assert "USER" in epic_id or "AUTHENTICATION" in epic_id
        
        # From filename
        epic_id = parser._generate_epic_id("Title", "test_requirements.txt")
        assert epic_id == "EPIC-TEST_REQUIREMENTS"
    
    def test_create_fallback_story(self, parser):
        """Test fallback story creation."""
        text = """This is a requirement.
- First bullet
- Second bullet
- Third bullet
More text here."""
        
        story = parser._create_fallback_story(text)
        
        assert isinstance(story, UserStory)
        assert story.id == "US-001"
        assert len(story.summary) > 0
        assert len(story.acceptance_criteria) >= 3  # Three bullets
    
    def test_parse_with_filename(self, parser):
        """Test parsing with filename for ID generation."""
        text = "Simple requirement text"
        epic = parser.parse(text, "auth_requirements.txt")
        
        assert "AUTH_REQUIREMENTS" in epic.epic_id
    
    def test_parse_multiple_formats_mixed(self, parser):
        """Test parsing with mixed formats."""
        text = """# Main Epic

As a user, I want feature A

User Story: Feature B
AC: Requirement 1
AC: Requirement 2

1. Numbered feature C
   - Sub-requirement
   - Another sub-requirement
"""
        
        epic = parser.parse(text)
        
        assert isinstance(epic, Epic)
        assert len(epic.user_stories) >= 1


class TestConvenienceFunction:
    """Test the convenience function."""
    
    def test_parse_natural_language_requirements(self):
        """Test the convenience function."""
        text = "Epic: Test\nRequirement 1\nRequirement 2"
        
        epic = parse_natural_language_requirements(text)
        
        assert isinstance(epic, Epic)
        assert len(epic.user_stories) >= 1
