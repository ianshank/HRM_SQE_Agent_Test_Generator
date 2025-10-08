"""
Output Formatter for Drop Folder System.

Formats test cases into multiple output formats:
- JSON (structured data)
- Markdown (human-readable)
- Report (processing metadata)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from hrm_eval.requirements_parser.schemas import Epic, TestCase

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Format test cases into various output formats.
    
    Generates:
    - Structured JSON for programmatic access
    - Readable Markdown for documentation
    - Processing reports with metadata
    """
    
    def __init__(self):
        """Initialize formatter."""
        logger.debug("OutputFormatter initialized")
    
    def save_json(self, test_cases: List[TestCase], output_path: Path):
        """
        Save test cases as structured JSON.
        
        Args:
            test_cases: List of generated test cases
            output_path: Path to save JSON file
        """
        logger.info(f"Saving {len(test_cases)} test cases to JSON: {output_path}")
        
        try:
            # Convert test cases to dictionaries
            test_data = [tc.dict() for tc in test_cases]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"JSON saved successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}", exc_info=True)
            raise
    
    def save_markdown(
        self,
        test_cases: List[TestCase],
        epic: Epic,
        output_path: Path
    ):
        """
        Save test cases as readable Markdown.
        
        Args:
            test_cases: List of generated test cases
            epic: Source epic
            output_path: Path to save Markdown file
        """
        logger.info(f"Saving {len(test_cases)} test cases to Markdown: {output_path}")
        
        try:
            lines = []
            
            # Header
            lines.append(f"# Test Cases: {epic.title}")
            lines.append("")
            lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"**Total Test Cases:** {len(test_cases)}")
            lines.append("")
            lines.append("---")
            lines.append("")
            
            # Group by type
            test_by_type = {}
            for tc in test_cases:
                tc_type = tc.type if hasattr(tc, 'type') else 'unknown'
                if tc_type not in test_by_type:
                    test_by_type[tc_type] = []
                test_by_type[tc_type].append(tc)
            
            # Write each type section
            for test_type, tests in sorted(test_by_type.items()):
                lines.append(f"## {test_type.title()} Tests ({len(tests)})")
                lines.append("")
                
                for idx, tc in enumerate(tests, 1):
                    lines.extend(self._format_test_case_markdown(tc, idx))
                    lines.append("")
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.debug(f"Markdown saved successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save Markdown: {e}", exc_info=True)
            raise
    
    def _format_test_case_markdown(self, tc: TestCase, number: int) -> List[str]:
        """Format a single test case as Markdown lines."""
        lines = []
        
        # Test case header
        tc_id = tc.id if hasattr(tc, 'id') else f"TC-{number:03d}"
        lines.append(f"### {number}. {tc_id}: {tc.description}")
        lines.append("")
        
        # Priority and type
        if hasattr(tc, 'priority'):
            lines.append(f"**Priority:** {tc.priority}")
        if hasattr(tc, 'type'):
            lines.append(f"**Type:** {tc.type}")
        lines.append("")
        
        # Preconditions
        if tc.preconditions:
            lines.append("**Preconditions:**")
            for precond in tc.preconditions:
                lines.append(f"- {precond}")
            lines.append("")
        
        # Test steps
        lines.append("**Test Steps:**")
        for step in tc.test_steps:
            step_num = step.step_number if hasattr(step, 'step_number') else '?'
            lines.append(f"{step_num}. {step.action}")
        lines.append("")
        
        # Expected results
        lines.append("**Expected Results:**")
        for result in tc.expected_results:
            result_text = result.result if hasattr(result, 'result') else str(result)
            lines.append(f"- {result_text}")
        lines.append("")
        
        # Metadata if available
        if hasattr(tc, 'metadata') and tc.metadata:
            if tc.metadata.get('rag_enabled'):
                lines.append(f"*RAG-Enhanced: {tc.metadata.get('retrieved_examples', 0)} examples used*")
                lines.append("")
        
        lines.append("---")
        
        return lines
    
    def save_report(
        self,
        epic: Epic,
        test_cases: List[TestCase],
        rag_stats: Dict[str, Any],
        processing_time: float,
        output_path: Path
    ):
        """
        Save processing report with metadata.
        
        Args:
            epic: Source epic
            test_cases: Generated test cases
            rag_stats: RAG retrieval statistics
            processing_time: Time taken to process (seconds)
            output_path: Path to save report
        """
        logger.info(f"Saving processing report to: {output_path}")
        
        try:
            lines = []
            
            # Header
            lines.append("# Test Generation Report")
            lines.append("")
            lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"**Processing Time:** {processing_time:.2f} seconds")
            lines.append("")
            lines.append("---")
            lines.append("")
            
            # Epic summary
            lines.append("## Epic Summary")
            lines.append("")
            lines.append(f"**Title:** {epic.title}")
            lines.append(f"**User Stories:** {len(epic.user_stories)}")
            lines.append(f"**Total Test Cases Generated:** {len(test_cases)}")
            lines.append("")
            
            # Test case breakdown
            lines.append("## Test Case Breakdown")
            lines.append("")
            
            # By type
            type_counts = {}
            priority_counts = {}
            for tc in test_cases:
                tc_type = getattr(tc, 'type', 'unknown')
                type_counts[tc_type] = type_counts.get(tc_type, 0) + 1
                
                priority = getattr(tc, 'priority', 'unknown')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            lines.append("### By Type")
            for test_type, count in sorted(type_counts.items()):
                lines.append(f"- **{test_type}:** {count}")
            lines.append("")
            
            lines.append("### By Priority")
            for priority, count in sorted(priority_counts.items()):
                lines.append(f"- **{priority}:** {count}")
            lines.append("")
            
            # RAG statistics
            if rag_stats.get('rag_enabled', False):
                lines.append("## RAG Enhancement")
                lines.append("")
                lines.append(f"**Status:** Enabled")
                lines.append(f"**Total Similar Tests Retrieved:** {rag_stats.get('retrieved_examples', 0)}")
                lines.append(f"**Retrieval Operations:** {rag_stats.get('total_retrievals', 0)}")
                avg_per_story = rag_stats.get('retrieved_examples', 0) / max(rag_stats.get('total_retrievals', 1), 1)
                lines.append(f"**Average per Story:** {avg_per_story:.1f}")
                lines.append("")
            else:
                lines.append("## RAG Enhancement")
                lines.append("")
                lines.append(f"**Status:** Disabled")
                lines.append("")
            
            # User story details
            lines.append("## User Story Details")
            lines.append("")
            for idx, story in enumerate(epic.user_stories, 1):
                lines.append(f"### {idx}. {story.summary}")
                lines.append("")
                lines.append(f"**Description:** {story.description[:200]}...")
                if story.acceptance_criteria:
                    lines.append(f"**Acceptance Criteria:** {len(story.acceptance_criteria)}")
                
                # Count tests for this story
                story_tests = [tc for tc in test_cases if hasattr(tc, 'user_story_id') and tc.user_story_id == story.id]
                lines.append(f"**Tests Generated:** {len(story_tests) if story_tests else 'N/A'}")
                lines.append("")
            
            # Quality metrics
            lines.append("## Quality Metrics")
            lines.append("")
            
            avg_steps = sum(len(tc.test_steps) for tc in test_cases) / len(test_cases) if test_cases else 0
            avg_preconditions = sum(len(tc.preconditions) for tc in test_cases) / len(test_cases) if test_cases else 0
            avg_results = sum(len(tc.expected_results) for tc in test_cases) / len(test_cases) if test_cases else 0
            
            lines.append(f"- **Average Steps per Test:** {avg_steps:.1f}")
            lines.append(f"- **Average Preconditions per Test:** {avg_preconditions:.1f}")
            lines.append(f"- **Average Expected Results per Test:** {avg_results:.1f}")
            lines.append(f"- **Generation Speed:** {len(test_cases) / processing_time:.2f} tests/second")
            lines.append("")
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.debug(f"Report saved successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}", exc_info=True)
            raise
