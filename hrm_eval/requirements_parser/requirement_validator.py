"""
Requirement validator for input validation and quality checks.

Ensures requirements meet minimum quality standards before processing.
"""

from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
import logging

from .schemas import Epic, UserStory, AcceptanceCriteria

logger = logging.getLogger(__name__)


class RequirementValidationError(Exception):
    """Exception raised when requirement validation fails."""
    pass


class RequirementValidator:
    """
    Validates requirement structures for completeness and quality.
    
    Performs validation checks to ensure requirements contain sufficient
    information for test case generation.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, enforce stricter validation rules
        """
        self.strict_mode = strict_mode
        logger.info(f"RequirementValidator initialized (strict_mode={strict_mode})")
    
    def validate_epic(self, epic: Epic) -> Tuple[bool, List[str]]:
        """
        Validate an epic for test case generation readiness.
        
        Args:
            epic: Epic to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not epic.user_stories:
            issues.append("Epic contains no user stories")
            return False, issues
        
        if len(epic.title) < 10:
            issues.append(f"Epic title too short: '{epic.title}'")
        
        if self.strict_mode and not epic.tech_stack:
            issues.append("Epic missing technology stack information")
        
        for idx, story in enumerate(epic.user_stories):
            story_issues = self._validate_user_story(story)
            if story_issues:
                issues.extend([f"Story {idx+1} ({story.id}): {issue}" for issue in story_issues])
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Epic validation failed with {len(issues)} issues")
        else:
            logger.info(f"Epic '{epic.epic_id}' validated successfully")
        
        return is_valid, issues
    
    def _validate_user_story(self, story: UserStory) -> List[str]:
        """
        Validate a single user story.
        
        Args:
            story: User story to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        if len(story.summary) < 10:
            issues.append(f"Summary too short: '{story.summary}'")
        
        if len(story.description) < 20:
            issues.append(f"Description too short (min 20 chars)")
        
        if not story.acceptance_criteria:
            issues.append("No acceptance criteria defined")
        elif self.strict_mode:
            if len(story.acceptance_criteria) < 2:
                issues.append("Fewer than 2 acceptance criteria (recommended: 2+)")
        
        for idx, criterion in enumerate(story.acceptance_criteria):
            if len(criterion.criteria) < 10:
                issues.append(f"Acceptance criterion {idx+1} too short")
        
        return issues
    
    def validate_json(self, epic_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate raw JSON input before parsing.
        
        Args:
            epic_data: Raw JSON dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        required_fields = ['epic_id', 'title', 'user_stories']
        for field in required_fields:
            if field not in epic_data:
                issues.append(f"Missing required field: '{field}'")
        
        if 'user_stories' in epic_data:
            if not isinstance(epic_data['user_stories'], list):
                issues.append("'user_stories' must be a list")
            elif len(epic_data['user_stories']) == 0:
                issues.append("'user_stories' list is empty")
            else:
                for idx, story in enumerate(epic_data['user_stories']):
                    if not isinstance(story, dict):
                        issues.append(f"User story {idx+1} is not a dictionary")
                        continue
                    
                    story_required = ['id', 'summary', 'description']
                    for field in story_required:
                        if field not in story:
                            issues.append(f"User story {idx+1} missing '{field}'")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.error(f"JSON validation failed with {len(issues)} issues")
        
        return is_valid, issues
    
    def check_testability(self, epic: Epic) -> Tuple[float, Dict[str, Any]]:
        """
        Assess how testable the requirements are.
        
        Args:
            epic: Epic to assess
            
        Returns:
            Tuple of (testability_score, detailed_report)
        """
        score = 0.0
        max_score = 0.0
        report = {
            "total_stories": len(epic.user_stories),
            "stories_with_criteria": 0,
            "total_criteria": 0,
            "has_tech_stack": bool(epic.tech_stack),
            "has_architecture": bool(epic.architecture),
            "testable_stories": [],
            "weak_stories": [],
        }
        
        for story in epic.user_stories:
            max_score += 100
            story_score = 0
            
            if story.acceptance_criteria:
                story_score += 40
                report["stories_with_criteria"] += 1
                report["total_criteria"] += len(story.acceptance_criteria)
            
            if len(story.description) > 50:
                story_score += 20
            
            if story.tech_stack:
                story_score += 20
            
            if len(story.acceptance_criteria) >= 2:
                story_score += 20
            
            score += story_score
            
            if story_score >= 60:
                report["testable_stories"].append(story.id)
            else:
                report["weak_stories"].append({
                    "id": story.id,
                    "score": story_score,
                    "issues": self._validate_user_story(story)
                })
        
        testability_score = (score / max_score) if max_score > 0 else 0.0
        report["testability_score"] = testability_score
        
        logger.info(f"Testability score: {testability_score:.2%}")
        
        return testability_score, report

