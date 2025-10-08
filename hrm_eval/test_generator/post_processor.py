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
    
    # Constants for text cleaning
    COMMON_SHORT_WORDS = {'for', 'the', 'and', 'but', 'or', 'is', 'in', 'on', 'at', 'to', 'a', 'an'}
    
    # Constants for quality validation
    GENERIC_TERMS = ['integration test', 'agent agent', 'test test', 'for: requirement']
    TRUNCATED_INDICATORS = ['ceptance', 'equirement', 'rchitecture']
    
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
        
        # Clean repetitive text
        description = self._clean_repetitive_text(description)
        
        return description
    
    def _clean_repetitive_text(self, text: str) -> str:
        """
        Remove repetitive words and clean up text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text without consecutive duplicates
        """
        words = text.split()
        
        # Remove consecutive duplicates
        cleaned = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() != words[i-1].lower():
                cleaned.append(word)
        
        result = " ".join(cleaned)
        
        # Remove truncated words (less than 3 chars unless common)
        words = result.split()
        cleaned_words = [w for w in words if len(w) >= 3 or w.lower() in self.COMMON_SHORT_WORDS]
        
        return " ".join(cleaned_words)
    
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
    
    def deduplicate_tests(self, test_cases: List[TestCase]) -> List[TestCase]:
        """
        Remove duplicate or near-duplicate test cases.
        
        Args:
            test_cases: List of test cases to deduplicate
            
        Returns:
            Deduplicated list of test cases
        """
        unique_tests = []
        seen_signatures = set()
        
        for test in test_cases:
            # Create signature from description + type + first step
            first_step = ""
            if test.test_steps:
                first_step = test.test_steps[0].action[:50]
            
            signature = f"{test.description[:50]}|{test.type}|{first_step}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_tests.append(test)
            else:
                logger.debug(f"Removing duplicate test: {test.id}")
        
        removed_count = len(test_cases) - len(unique_tests)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate test cases")
        
        return unique_tests
    
    def enhance_test_specificity(self, test_case: TestCase, context: TestContext) -> TestCase:
        """
        Make test more specific based on context.
        
        Args:
            test_case: Test case to enhance
            context: Original test context
            
        Returns:
            Enhanced test case
        """
        # Replace generic terms with specific ones from context
        if "requirement" in test_case.description and context.acceptance_criterion:
            specific_desc = test_case.description.replace(
                "requirement",
                context.acceptance_criterion[:50]
            )
            test_case.description = specific_desc
        
        # Add context-specific details to steps
        for step in test_case.test_steps:
            if "Execute:" in step.action and context.acceptance_criterion:
                step.action = f"Execute: {context.acceptance_criterion[:80]}"
        
        return test_case
    
    def validate_test_quality(self, test_case: TestCase) -> tuple[bool, List[str]]:
        """
        Validate test case quality and return issues.
        
        Args:
            test_case: Test case to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for consecutive duplicate words (not all repetitions)
        words = test_case.description.split()
        word_list = [w.lower() for w in words]
        has_consecutive_duplicates = any(
            word_list[i] == word_list[i-1] 
            for i in range(1, len(word_list))
        )
        if has_consecutive_duplicates:
            issues.append("Consecutive duplicate words in description")
        
        # Check for truncated text using class constant
        if any(word in test_case.description.lower() for word in self.TRUNCATED_INDICATORS):
            issues.append("Truncated text detected")
        
        # Check for generic descriptions using class constant
        if any(term in test_case.description.lower() for term in self.GENERIC_TERMS):
            issues.append("Too generic - lacks specificity")
        
        # Check minimum steps
        if len(test_case.test_steps) < 2:
            issues.append("Insufficient test steps")
        
        # Check preconditions
        if not test_case.preconditions:
            issues.append("Missing preconditions")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.debug(f"Quality issues in test {test_case.id}: {', '.join(issues)}")
        
        return is_valid, issues

