"""
Coverage analyzer for ensuring comprehensive test coverage.

Analyzes generated test cases to identify coverage gaps and ensure
all requirements and test types are adequately covered.
"""

from typing import List, Dict, Any, Set
from collections import defaultdict
import logging

from ..requirements_parser.schemas import TestCase, TestContext, TestType

logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """
    Analyzes test coverage to ensure comprehensive testing.
    
    Tracks coverage across:
    - Test types (positive, negative, edge)
    - User stories
    - Acceptance criteria
    - Priority distribution
    """
    
    def __init__(self, min_coverage: float = 0.8):
        """
        Initialize coverage analyzer.
        
        Args:
            min_coverage: Minimum acceptable coverage percentage
        """
        self.min_coverage = min_coverage
        logger.info(f"CoverageAnalyzer initialized (min_coverage={min_coverage})")
    
    def analyze_coverage(
        self,
        test_cases: List[TestCase],
        contexts: List[TestContext],
    ) -> Dict[str, Any]:
        """
        Analyze test coverage comprehensively.
        
        Args:
            test_cases: Generated test cases
            contexts: Original test contexts
            
        Returns:
            Coverage analysis report
        """
        logger.info(f"Analyzing coverage for {len(test_cases)} test cases")
        
        report = {
            "total_test_cases": len(test_cases),
            "total_contexts": len(contexts),
            "positive_tests": 0,
            "negative_tests": 0,
            "edge_tests": 0,
            "performance_tests": 0,
            "security_tests": 0,
            "stories_covered": set(),
            "missing_test_types": [],
            "priority_distribution": defaultdict(int),
            "coverage_percentage": 0.0,
            "gaps": [],
        }
        
        for test_case in test_cases:
            if test_case.type == TestType.POSITIVE:
                report["positive_tests"] += 1
            elif test_case.type == TestType.NEGATIVE:
                report["negative_tests"] += 1
            elif test_case.type == TestType.EDGE:
                report["edge_tests"] += 1
            elif test_case.type == TestType.PERFORMANCE:
                report["performance_tests"] += 1
            elif test_case.type == TestType.SECURITY:
                report["security_tests"] += 1
            
            if test_case.source_story_id:
                report["stories_covered"].add(test_case.source_story_id)
            
            report["priority_distribution"][test_case.priority.value] += 1
        
        unique_stories = set(ctx.story_id for ctx in contexts)
        report["total_stories"] = len(unique_stories)
        report["stories_covered"] = list(report["stories_covered"])
        report["stories_covered_count"] = len(report["stories_covered"])
        
        if report["total_stories"] > 0:
            story_coverage = report["stories_covered_count"] / report["total_stories"]
            report["coverage_percentage"] = story_coverage * 100
        
        required_types = {TestType.POSITIVE, TestType.NEGATIVE, TestType.EDGE}
        covered_types = set()
        
        for test_case in test_cases:
            covered_types.add(test_case.type)
        
        missing_types = required_types - covered_types
        report["missing_test_types"] = [t.value for t in missing_types]
        
        report["gaps"] = self._identify_gaps(test_cases, contexts)
        
        self._log_coverage_summary(report)
        
        return report
    
    def _identify_gaps(
        self,
        test_cases: List[TestCase],
        contexts: List[TestContext],
    ) -> List[Dict[str, Any]]:
        """
        Identify specific coverage gaps.
        
        Args:
            test_cases: Generated test cases
            contexts: Original test contexts
            
        Returns:
            List of identified gaps
        """
        gaps = []
        
        context_by_story = defaultdict(list)
        for ctx in contexts:
            context_by_story[ctx.story_id].append(ctx)
        
        testcase_by_story = defaultdict(list)
        for tc in test_cases:
            if tc.source_story_id:
                testcase_by_story[tc.source_story_id].append(tc)
        
        for story_id, story_contexts in context_by_story.items():
            story_test_cases = testcase_by_story.get(story_id, [])
            
            context_types = set(ctx.test_type for ctx in story_contexts)
            testcase_types = set(tc.type for tc in story_test_cases)
            
            missing = context_types - testcase_types
            
            if missing:
                gaps.append({
                    "story_id": story_id,
                    "gap_type": "missing_test_types",
                    "missing_types": [t.value for t in missing],
                })
            
            if len(story_test_cases) == 0:
                gaps.append({
                    "story_id": story_id,
                    "gap_type": "no_test_cases",
                    "message": "No test cases generated for this story",
                })
        
        return gaps
    
    def _log_coverage_summary(self, report: Dict[str, Any]):
        """
        Log coverage summary.
        
        Args:
            report: Coverage report
        """
        logger.info("=" * 60)
        logger.info("COVERAGE ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Test Cases: {report['total_test_cases']}")
        logger.info(f"Stories Covered: {report['stories_covered_count']}/{report['total_stories']}")
        logger.info(f"Coverage: {report['coverage_percentage']:.1f}%")
        logger.info("")
        logger.info("Test Type Distribution:")
        logger.info(f"  Positive: {report['positive_tests']}")
        logger.info(f"  Negative: {report['negative_tests']}")
        logger.info(f"  Edge: {report['edge_tests']}")
        logger.info(f"  Performance: {report['performance_tests']}")
        logger.info(f"  Security: {report['security_tests']}")
        logger.info("")
        logger.info("Priority Distribution:")
        for priority, count in report['priority_distribution'].items():
            logger.info(f"  {priority}: {count}")
        
        if report['missing_test_types']:
            logger.warning(f"Missing test types: {', '.join(report['missing_test_types'])}")
        
        if report['gaps']:
            logger.warning(f"Identified {len(report['gaps'])} coverage gaps")
        
        if report['coverage_percentage'] < self.min_coverage * 100:
            logger.warning(
                f"Coverage {report['coverage_percentage']:.1f}% below minimum "
                f"{self.min_coverage*100:.0f}%"
            )
        else:
            logger.info("Coverage meets minimum threshold [x]")
        
        logger.info("=" * 60)
    
    def get_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """
        Get recommendations for improving coverage.
        
        Args:
            report: Coverage report
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if report['coverage_percentage'] < 80:
            recommendations.append(
                f"Increase coverage from {report['coverage_percentage']:.1f}% to at least 80%"
            )
        
        if report['missing_test_types']:
            recommendations.append(
                f"Add test cases for: {', '.join(report['missing_test_types'])}"
            )
        
        if report['positive_tests'] < report['negative_tests']:
            recommendations.append(
                "Consider adding more positive (happy path) test cases"
            )
        
        if report['edge_tests'] == 0:
            recommendations.append(
                "Add edge case and boundary condition tests"
            )
        
        p1_count = report['priority_distribution'].get('P1', 0)
        if p1_count < report['total_test_cases'] * 0.3:
            recommendations.append(
                "Consider increasing P1 critical test coverage"
            )
        
        return recommendations

