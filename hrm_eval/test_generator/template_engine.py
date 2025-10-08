"""
Template engine for test case formatting.

Provides templates and formatting utilities for generating consistent
test case structures.
"""

from typing import Dict, Any, List
import logging

from ..requirements_parser.schemas import TestCase, TestType, Priority

logger = logging.getLogger(__name__)


class TestCaseTemplate:
    """
    Template engine for test case formatting.
    
    Provides standardized templates matching the output format from
    the Media Asset Management example.
    """
    
    TEMPLATE_BASE = {
        "id": "TC-{counter:03d}",
        "type": "{test_type}",
        "priority": "{priority}",
        "description": "{description}",
        "preconditions": [],
        "test_steps": [],
        "expected_results": [],
        "test_data": "{test_data}",
        "labels": [],
    }
    
    TYPE_DESCRIPTIONS = {
        TestType.POSITIVE: "Verify successful {scenario} when all conditions are met",
        TestType.NEGATIVE: "Verify proper error handling when {scenario} fails",
        TestType.EDGE: "Verify system behavior under edge case: {scenario}",
        TestType.PERFORMANCE: "Verify performance requirements for {scenario}",
        TestType.SECURITY: "Verify security controls for {scenario}",
    }
    
    PRIORITY_RULES = {
        Priority.P1: {
            "keywords": ["security", "data integrity", "critical", "authentication", "authorization"],
            "test_types": [TestType.POSITIVE, TestType.SECURITY],
            "description": "Critical functionality, security, or data integrity",
        },
        Priority.P2: {
            "keywords": ["integration", "api", "workflow", "standard"],
            "test_types": [TestType.NEGATIVE],
            "description": "Standard functionality and integrations",
        },
        Priority.P3: {
            "keywords": ["edge", "boundary", "performance", "ui", "ux"],
            "test_types": [TestType.EDGE, TestType.PERFORMANCE],
            "description": "Edge cases, performance, UX improvements",
        },
    }
    
    def __init__(self):
        """Initialize template engine."""
        logger.info("TestCaseTemplate initialized")
    
    def format_test_case(self, test_case: TestCase, counter: int) -> Dict[str, Any]:
        """
        Format test case using template.
        
        Args:
            test_case: TestCase object
            counter: Test case counter for ID
            
        Returns:
            Formatted test case dictionary
        """
        formatted = {
            "id": f"TC-{counter:03d}",
            "type": test_case.type.value,
            "priority": test_case.priority.value,
            "description": test_case.description,
            "preconditions": test_case.preconditions,
            "test_steps": [
                {
                    "step_number": step.step_number,
                    "action": step.action,
                }
                for step in test_case.test_steps
            ],
            "expected_results": [
                result.result for result in test_case.expected_results
            ],
            "test_data": test_case.test_data or "Standard test dataset",
            "labels": test_case.labels,
        }
        
        return formatted
    
    def generate_description(
        self,
        test_type: TestType,
        scenario: str,
        context: str = "",
    ) -> str:
        """
        Generate test case description from template.
        
        Args:
            test_type: Type of test
            scenario: Scenario being tested
            context: Additional context
            
        Returns:
            Formatted description
        """
        template = self.TYPE_DESCRIPTIONS.get(
            test_type,
            "Verify {scenario}"
        )
        
        description = template.format(scenario=scenario)
        
        if context:
            description += f" ({context})"
        
        return description
    
    def determine_priority_by_rules(
        self,
        test_case: TestCase,
    ) -> Priority:
        """
        Determine priority using rule-based logic.
        
        Args:
            test_case: Test case to prioritize
            
        Returns:
            Priority level
        """
        text_to_check = (
            test_case.description.lower() + " " +
            " ".join(test_case.labels).lower()
        )
        
        for priority, rules in self.PRIORITY_RULES.items():
            if test_case.type in rules["test_types"]:
                return priority
            
            for keyword in rules["keywords"]:
                if keyword in text_to_check:
                    return priority
        
        return Priority.P2
    
    def apply_naming_conventions(self, test_cases: List[TestCase]) -> List[TestCase]:
        """
        Apply consistent naming conventions to test cases.
        
        Args:
            test_cases: List of test cases
            
        Returns:
            Test cases with consistent naming
        """
        for idx, test_case in enumerate(test_cases, start=1):
            if not test_case.id.startswith("TC-"):
                test_case.id = f"TC-{idx:03d}"
        
        return test_cases
    
    def generate_markdown_output(self, test_cases: List[TestCase]) -> str:
        """
        Generate markdown-formatted output matching example format.
        
        Args:
            test_cases: List of test cases
            
        Returns:
            Markdown-formatted string
        """
        lines = []
        lines.append("# Generated Test Cases")
        lines.append("")
        lines.append(f"Total: {len(test_cases)} test cases")
        lines.append("")
        
        for test_case in test_cases:
            lines.append("─" * 80)
            lines.append(f"## {test_case.id}: {test_case.description}")
            lines.append("─" * 80)
            lines.append(f"**Type:** {test_case.type.value}")
            lines.append(f"**Priority:** {test_case.priority.value}")
            lines.append("")
            
            if test_case.preconditions:
                lines.append("**Preconditions:**")
                for precond in test_case.preconditions:
                    lines.append(f"  - {precond}")
                lines.append("")
            
            lines.append("**Test Steps:**")
            for step in test_case.test_steps:
                lines.append(f"  {step.step_number}. {step.action}")
            lines.append("")
            
            lines.append("**Expected Results:**")
            for result in test_case.expected_results:
                lines.append(f"  [x] {result.result}")
            lines.append("")
            
            if test_case.test_data:
                lines.append(f"**Test Data:** {test_case.test_data}")
                lines.append("")
            
            if test_case.labels:
                lines.append(f"**Labels:** {', '.join(test_case.labels)}")
                lines.append("")
        
        return "\n".join(lines)

