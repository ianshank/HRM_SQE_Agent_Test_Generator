"""
Post-processor for converting HRM model outputs to structured test cases.

Processes raw model token outputs into the structured TestCase format.
"""

from typing import List, Dict, Any, Optional
import logging
import re

from ..requirements_parser.schemas import (
    TestCase,
    TestStep,
    ExpectedResult,
    TestContext,
    TestType,
    Priority,
)

logger = logging.getLogger(__name__)


class TestCasePostProcessor:
    """
    Post-processes HRM model outputs into structured test cases.
    
    Converts token sequences from the model into properly formatted
    TestCase objects with steps, expected results, and metadata.
    """
    
    def __init__(self):
        """Initialize post-processor."""
        self.token_to_keyword = {
            0: "start",
            1: "test",
            2: "agent",
            3: "data",
            4: "security",
            5: "performance",
            6: "integration",
            7: "automation",
            8: "validation",
            9: "deployment",
            10: "monitoring",
            11: "end",
        }
        
        logger.info("TestCasePostProcessor initialized")
    
    def process_model_output(
        self,
        input_tokens: List[int],
        output_tokens: List[int],
        q_values: List[float],
        context: TestContext,
    ) -> List[TestCase]:
        """
        Process model output tokens into structured test cases.
        
        Args:
            input_tokens: Input token sequence
            output_tokens: Model-generated output tokens
            q_values: Q-values from RL head
            context: Original test context
            
        Returns:
            List of structured TestCase objects
        """
        logger.debug(f"Post-processing model output: {len(output_tokens)} tokens")
        
        output_text = self._tokens_to_text(output_tokens)
        
        logger.debug(f"Decoded output text: {output_text[:100]}...")
        
        test_case = self._construct_test_case(
            output_text=output_text,
            context=context,
            q_values=q_values,
        )
        
        return [test_case]
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """
        Convert token sequence to text representation.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Text representation
        """
        keywords = []
        for token_id in tokens:
            if token_id in self.token_to_keyword:
                keyword = self.token_to_keyword[token_id]
                if keyword not in ["start", "end"]:
                    keywords.append(keyword)
        
        return " ".join(keywords)
    
    def _construct_test_case(
        self,
        output_text: str,
        context: TestContext,
        q_values: List[float],
    ) -> TestCase:
        """
        Construct TestCase from model output and context.
        
        Args:
            output_text: Decoded output text
            context: Original test context
            q_values: Q-values from model
            
        Returns:
            Structured TestCase object
        """
        description = self._generate_description(context, output_text)
        preconditions = self._extract_preconditions(context)
        test_steps = self._generate_test_steps(context, output_text)
        expected_results = self._generate_expected_results(context, output_text)
        test_data = self._generate_test_data(context)
        labels = self._generate_labels(context, output_text)
        
        test_case = TestCase(
            id="TC-000",  # Will be assigned by generator
            type=context.test_type,
            priority=Priority.P2,  # Will be refined by determine_priority
            description=description,
            preconditions=preconditions,
            test_steps=test_steps,
            expected_results=expected_results,
            test_data=test_data,
            labels=labels,
            source_story_id=context.story_id,
        )
        
        return test_case
    
    def _generate_description(self, context: TestContext, output_text: str) -> str:
        """Generate test case description."""
        if context.test_type == TestType.POSITIVE:
            prefix = "Verify successful"
        elif context.test_type == TestType.NEGATIVE:
            prefix = "Verify proper error handling when"
        else:
            prefix = "Verify system behavior under"
        
        criterion = context.acceptance_criterion or "requirement"
        
        keywords = output_text.split()[:5]
        context_hint = " ".join(keywords) if keywords else "scenario"
        
        description = f"{prefix} {context_hint} for: {criterion}"
        
        return description
    
    def _extract_preconditions(self, context: TestContext) -> List[str]:
        """Extract preconditions from context."""
        preconditions = []
        
        if context.test_type != TestType.NEGATIVE:
            preconditions.append("System is operational and accessible")
            preconditions.append("User has appropriate permissions")
        
        if context.tech_context:
            preconditions.append(f"Required services available: {', '.join(context.tech_context[:3])}")
        
        if context.test_type == TestType.NEGATIVE:
            preconditions.append("Invalid or error condition present")
        
        return preconditions
    
    def _generate_test_steps(self, context: TestContext, output_text: str) -> List[TestStep]:
        """Generate test execution steps."""
        steps = []
        
        keywords = output_text.split()
        
        step_num = 1
        
        if "data" in keywords or "validation" in keywords:
            steps.append(TestStep(
                step_number=step_num,
                action="Prepare test data and prerequisites"
            ))
            step_num += 1
        
        if context.acceptance_criterion:
            steps.append(TestStep(
                step_number=step_num,
                action=f"Execute: {context.acceptance_criterion[:100]}"
            ))
            step_num += 1
        
        if "integration" in keywords or "agent" in keywords:
            steps.append(TestStep(
                step_number=step_num,
                action="Verify integration points and dependencies"
            ))
            step_num += 1
        
        if "monitoring" in keywords or "validation" in keywords:
            steps.append(TestStep(
                step_number=step_num,
                action="Monitor execution and capture results"
            ))
            step_num += 1
        
        steps.append(TestStep(
            step_number=step_num,
            action="Verify outcome and cleanup"
        ))
        
        return steps
    
    def _generate_expected_results(self, context: TestContext, output_text: str) -> List[ExpectedResult]:
        """Generate expected results."""
        results = []
        
        if context.test_type == TestType.POSITIVE:
            results.append(ExpectedResult(result="Operation completes successfully"))
            results.append(ExpectedResult(result="All acceptance criteria met"))
        elif context.test_type == TestType.NEGATIVE:
            results.append(ExpectedResult(result="Appropriate error message displayed"))
            results.append(ExpectedResult(result="System remains stable"))
        else:
            results.append(ExpectedResult(result="System handles edge case correctly"))
            results.append(ExpectedResult(result="No data corruption or loss"))
        
        keywords = output_text.split()
        
        if "security" in keywords:
            results.append(ExpectedResult(result="Security controls validated"))
        
        if "performance" in keywords:
            results.append(ExpectedResult(result="Performance within acceptable limits"))
        
        if "data" in keywords or "validation" in keywords:
            results.append(ExpectedResult(result="Data integrity maintained"))
        
        return results
    
    def _generate_test_data(self, context: TestContext) -> str:
        """Generate test data description."""
        if context.test_type == TestType.POSITIVE:
            return "Valid test data meeting all requirements"
        elif context.test_type == TestType.NEGATIVE:
            return "Invalid or malformed test data"
        else:
            return "Boundary condition test data"
    
    def _generate_labels(self, context: TestContext, output_text: str) -> List[str]:
        """Generate test case labels."""
        labels = [context.test_type.value]
        
        if context.story_id:
            labels.append(context.story_id)
        
        keywords = set(output_text.split())
        
        for keyword in ["security", "performance", "integration", "automation", "monitoring"]:
            if keyword in keywords:
                labels.append(keyword)
        
        priority = self.determine_priority_from_context(context)
        labels.append(priority.value)
        
        return list(set(labels))
    
    def determine_priority(self, test_case: TestCase) -> Priority:
        """
        Determine priority based on test case characteristics.
        
        Args:
            test_case: Test case to prioritize
            
        Returns:
            Priority level
        """
        if "security" in test_case.labels or "security" in test_case.description.lower():
            return Priority.P1
        
        if "data" in test_case.labels and test_case.type == TestType.NEGATIVE:
            return Priority.P1
        
        if test_case.type == TestType.POSITIVE and "integration" not in test_case.labels:
            return Priority.P1
        
        if test_case.type == TestType.EDGE or test_case.type == TestType.PERFORMANCE:
            return Priority.P3
        
        return Priority.P2
    
    def determine_priority_from_context(self, context: TestContext) -> Priority:
        """
        Determine priority from test context.
        
        Args:
            context: Test context
            
        Returns:
            Priority level
        """
        if context.test_type == TestType.POSITIVE:
            return Priority.P1
        elif context.test_type == TestType.NEGATIVE:
            return Priority.P2
        else:
            return Priority.P3

