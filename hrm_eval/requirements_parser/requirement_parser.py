"""
Requirement parser for converting structured requirements to test contexts.

Parses Epic/UserStory structures and extracts test contexts for HRM model processing.
NO dummy test generation - all outputs are prepared for actual HRM model inference.
"""

from typing import List, Dict, Any, Optional
from pydantic import ValidationError
import logging

from .schemas import (
    Epic,
    UserStory,
    AcceptanceCriteria,
    TestContext,
    TestType,
)
from .requirement_validator import RequirementValidator, RequirementValidationError

logger = logging.getLogger(__name__)


class RequirementParser:
    """
    Parses requirements and extracts test contexts for HRM processing.
    
    Prepares requirement data for the HRM model workflow - does NOT generate
    test cases directly (that's done by the HRM model).
    """
    
    def __init__(self, validator: Optional[RequirementValidator] = None):
        """
        Initialize parser.
        
        Args:
            validator: Optional validator instance (creates default if None)
        """
        self.validator = validator or RequirementValidator()
        logger.info("RequirementParser initialized")
    
    def parse_epic(self, epic_data: Dict[str, Any]) -> Epic:
        """
        Parse and validate epic from JSON data.
        
        Args:
            epic_data: Raw JSON dictionary containing epic data
            
        Returns:
            Validated Epic object
            
        Raises:
            RequirementValidationError: If validation fails
            ValidationError: If Pydantic validation fails
        """
        logger.info(f"Parsing epic data with {len(epic_data.get('user_stories', []))} stories")
        
        is_valid, issues = self.validator.validate_json(epic_data)
        if not is_valid:
            error_msg = f"JSON validation failed: {'; '.join(issues)}"
            logger.error(error_msg)
            raise RequirementValidationError(error_msg)
        
        try:
            epic = Epic(**epic_data)
        except ValidationError as e:
            logger.error(f"Pydantic validation failed: {e}")
            raise
        
        is_valid, issues = self.validator.validate_epic(epic)
        if not is_valid:
            logger.warning(f"Epic has {len(issues)} validation issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        logger.info(f"Successfully parsed epic '{epic.epic_id}' with {len(epic.user_stories)} stories")
        
        return epic
    
    def extract_test_contexts(self, epic: Epic) -> List[TestContext]:
        """
        Extract test contexts from epic for HRM model processing.
        
        Each test context represents a scenario that will be sent to the HRM model
        for test case generation. This method PREPARES the inputs for the model,
        it does NOT generate test cases itself.
        
        Args:
            epic: Validated Epic object
            
        Returns:
            List of TestContext objects ready for HRM processing
        """
        logger.info(f"Extracting test contexts from epic '{epic.epic_id}'")
        
        test_contexts = []
        
        for story in epic.user_stories:
            contexts = self._extract_contexts_from_story(story, epic)
            test_contexts.extend(contexts)
        
        logger.info(f"Extracted {len(test_contexts)} test contexts for HRM processing")
        
        return test_contexts
    
    def _extract_contexts_from_story(
        self,
        story: UserStory,
        epic: Epic
    ) -> List[TestContext]:
        """
        Extract test contexts from a single user story.
        
        Creates multiple test contexts per story to cover:
        - Positive scenarios (happy path)
        - Negative scenarios (error cases)
        - Edge cases (boundary conditions)
        
        Args:
            story: User story to process
            epic: Parent epic for context
            
        Returns:
            List of test contexts
        """
        contexts = []
        
        tech_context = list(set(story.tech_stack + epic.tech_stack))
        
        if story.acceptance_criteria:
            for criterion in story.acceptance_criteria:
                contexts.append(TestContext(
                    story_id=story.id,
                    test_type=TestType.POSITIVE,
                    requirement_text=self._format_requirement_for_model(
                        story, criterion, TestType.POSITIVE, epic
                    ),
                    acceptance_criterion=criterion.criteria,
                    tech_context=tech_context,
                ))
                
                contexts.append(TestContext(
                    story_id=story.id,
                    test_type=TestType.NEGATIVE,
                    requirement_text=self._format_requirement_for_model(
                        story, criterion, TestType.NEGATIVE, epic
                    ),
                    acceptance_criterion=criterion.criteria,
                    tech_context=tech_context,
                ))
        else:
            contexts.append(TestContext(
                story_id=story.id,
                test_type=TestType.POSITIVE,
                requirement_text=self._format_requirement_for_model(
                    story, None, TestType.POSITIVE, epic
                ),
                acceptance_criterion=None,
                tech_context=tech_context,
            ))
        
        edge_context = TestContext(
            story_id=story.id,
            test_type=TestType.EDGE,
            requirement_text=self._format_requirement_for_model(
                story, None, TestType.EDGE, epic
            ),
            acceptance_criterion="Edge cases and boundary conditions",
            tech_context=tech_context,
        )
        contexts.append(edge_context)
        
        logger.debug(f"Extracted {len(contexts)} contexts from story '{story.id}'")
        
        return contexts
    
    def _format_requirement_for_model(
        self,
        story: UserStory,
        criterion: Optional[AcceptanceCriteria],
        test_type: TestType,
        epic: Epic
    ) -> str:
        """
        Format requirement text for HRM model input.
        
        Creates a structured prompt that the HRM model can process to generate
        test cases. This is the input that gets tokenized and sent to the model.
        
        Args:
            story: User story
            criterion: Specific acceptance criterion (if any)
            test_type: Type of test to generate
            epic: Parent epic for context
            
        Returns:
            Formatted requirement text for model
        """
        parts = []
        
        parts.append(f"Epic: {epic.title}")
        parts.append(f"Story: {story.summary}")
        parts.append(f"Description: {story.description}")
        
        if criterion:
            parts.append(f"Acceptance Criterion: {criterion.criteria}")
        elif story.acceptance_criteria:
            criteria_text = "; ".join([c.criteria for c in story.acceptance_criteria])
            parts.append(f"Acceptance Criteria: {criteria_text}")
        
        if epic.tech_stack or story.tech_stack:
            tech = list(set(epic.tech_stack + story.tech_stack))
            parts.append(f"Tech Stack: {', '.join(tech)}")
        
        if epic.architecture:
            parts.append(f"Architecture: {epic.architecture}")
        
        test_type_instruction = self._get_test_type_instruction(test_type)
        parts.append(test_type_instruction)
        
        formatted_text = "\n".join(parts)
        
        logger.debug(f"Formatted requirement for {test_type.value} test (length: {len(formatted_text)} chars)")
        
        return formatted_text
    
    def _get_test_type_instruction(self, test_type: TestType) -> str:
        """
        Get instruction for the model based on test type.
        
        Args:
            test_type: Type of test case desired
            
        Returns:
            Instruction text for the model
        """
        instructions = {
            TestType.POSITIVE: "Generate test case for successful scenario (happy path)",
            TestType.NEGATIVE: "Generate test case for error/failure scenario",
            TestType.EDGE: "Generate test case for edge cases and boundary conditions",
            TestType.PERFORMANCE: "Generate performance test case",
            TestType.SECURITY: "Generate security test case",
        }
        
        return f"Test Type: {instructions.get(test_type, 'Generate test case')}"
    
    def get_coverage_analysis(self, epic: Epic) -> Dict[str, Any]:
        """
        Analyze potential test coverage from requirements.
        
        Args:
            epic: Epic to analyze
            
        Returns:
            Coverage analysis report
        """
        testability_score, report = self.validator.check_testability(epic)
        
        test_contexts = self.extract_test_contexts(epic)
        
        coverage_analysis = {
            "testability_score": testability_score,
            "total_stories": len(epic.user_stories),
            "total_acceptance_criteria": report["total_criteria"],
            "potential_test_contexts": len(test_contexts),
            "test_type_distribution": {
                "positive": sum(1 for ctx in test_contexts if ctx.test_type == TestType.POSITIVE),
                "negative": sum(1 for ctx in test_contexts if ctx.test_type == TestType.NEGATIVE),
                "edge": sum(1 for ctx in test_contexts if ctx.test_type == TestType.EDGE),
            },
            "testability_report": report,
        }
        
        logger.info(f"Coverage analysis: {len(test_contexts)} potential tests, "
                   f"testability score: {testability_score:.2%}")
        
        return coverage_analysis

