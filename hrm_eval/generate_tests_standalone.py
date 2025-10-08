"""
Standalone test case generator for fulfillment requirements.
Generates comprehensive, structured test cases based on requirements.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_test_cases_for_story(story: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate test cases for a user story."""
    test_cases = []
    story_id = story['id']
    summary = story['summary']
    description = story['description']
    criteria_list = story['acceptance_criteria']
    
    logger.info(f"  Generating tests for {story_id}: {summary}")
    
    for idx, criteria in enumerate(criteria_list, 1):
        criteria_text = criteria['criteria']
        
        # Generate positive test case
        positive_tc = {
            "id": f"{story_id}-TC-P{idx:02d}",
            "type": "positive",
            "priority": "P1" if idx <= 2 else "P2",
            "description": f"Verify {criteria_text}",
            "user_story_id": story_id,
            "user_story_summary": summary,
            "acceptance_criteria": criteria_text,
            "preconditions": [
                "System is in operational state",
                f"User has necessary permissions for {summary}",
                "All dependent services are available",
                "Test data is properly configured"
            ],
            "test_steps": [
                {
                    "step_number": 1,
                    "action": f"Set up test environment for {summary[:50]}",
                    "expected_result": "Environment is ready"
                },
                {
                    "step_number": 2,
                    "action": f"Execute primary action: {criteria_text[:80]}",
                    "expected_result": "Action executes without errors"
                },
                {
                    "step_number": 3,
                    "action": "Verify acceptance criteria is met",
                    "expected_result": criteria_text
                },
                {
                    "step_number": 4,
                    "action": "Validate system state post-execution",
                    "expected_result": "System remains stable and consistent"
                }
            ],
            "expected_results": [
                {
                    "result": criteria_text,
                    "success_criteria": "Acceptance criteria fully satisfied"
                },
                {
                    "result": "No errors or warnings in logs",
                    "success_criteria": "Clean execution"
                }
            ],
            "labels": ["fulfillment", "positive", story_id.lower().replace("-", "_")],
            "automation_level": "automated",
            "estimated_duration_minutes": 5,
            "source": "generated",
            "generation_timestamp": datetime.now().isoformat()
        }
        test_cases.append(positive_tc)
        
        # Generate negative test case (every other criteria)
        if idx % 2 == 0:
            negative_tc = {
                "id": f"{story_id}-TC-N{idx:02d}",
                "type": "negative",
                "priority": "P2",
                "description": f"Verify system handles invalid input for: {criteria_text[:50]}",
                "user_story_id": story_id,
                "user_story_summary": summary,
                "acceptance_criteria": criteria_text,
                "preconditions": [
                    "System is operational",
                    "Invalid test data is prepared"
                ],
                "test_steps": [
                    {
                        "step_number": 1,
                        "action": "Attempt action with invalid/missing parameters",
                        "expected_result": "System rejects invalid input"
                    },
                    {
                        "step_number": 2,
                        "action": "Verify error handling",
                        "expected_result": "Appropriate error message displayed"
                    },
                    {
                        "step_number": 3,
                        "action": "Confirm system stability",
                        "expected_result": "System remains operational"
                    }
                ],
                "expected_results": [
                    {
                        "result": "Invalid input is rejected",
                        "success_criteria": "Proper validation in place"
                    },
                    {
                        "result": "Clear error message provided",
                        "success_criteria": "Good user experience"
                    }
                ],
                "labels": ["fulfillment", "negative", "validation", story_id.lower().replace("-", "_")],
                "automation_level": "automated",
                "estimated_duration_minutes": 3,
                "source": "generated",
                "generation_timestamp": datetime.now().isoformat()
            }
            test_cases.append(negative_tc)
    
    # Generate edge case for the story
    edge_tc = {
        "id": f"{story_id}-TC-E01",
        "type": "edge_case",
        "priority": "P3",
        "description": f"Verify edge cases and boundary conditions for {summary}",
        "user_story_id": story_id,
        "user_story_summary": summary,
        "preconditions": [
            "System at capacity or boundary conditions",
            "Edge case test data prepared"
        ],
        "test_steps": [
            {
                "step_number": 1,
                "action": "Test with maximum allowed values",
                "expected_result": "System handles max values correctly"
            },
            {
                "step_number": 2,
                "action": "Test with minimum allowed values",
                "expected_result": "System handles min values correctly"
            },
            {
                "step_number": 3,
                "action": "Test concurrent operations",
                "expected_result": "No race conditions or conflicts"
            }
        ],
        "expected_results": [
            {
                "result": "All boundary conditions handled gracefully",
                "success_criteria": "Robust system behavior"
            }
        ],
        "labels": ["fulfillment", "edge_case", story_id.lower().replace("-", "_")],
        "automation_level": "manual",
        "estimated_duration_minutes": 10,
        "source": "generated",
        "generation_timestamp": datetime.now().isoformat()
    }
    test_cases.append(edge_tc)
    
    return test_cases


def main():
    """Main generation function."""
    logger.info("="*80)
    logger.info("GENERATING TEST CASES FOR FULFILLMENT REQUIREMENTS")
    logger.info("="*80)
    
    # Load requirements
    logger.info("\n[1/4] Loading requirements...")
    with open("test_data/real_fulfillment_requirements.json") as f:
        requirements = json.load(f)
    
    epic_id = requirements['epic_id']
    epic_title = requirements['title']
    user_stories = requirements['user_stories']
    
    logger.info(f"  Epic: {epic_title}")
    logger.info(f"  User Stories: {len(user_stories)}")
    logger.info(f"  Total Acceptance Criteria: {sum(len(s['acceptance_criteria']) for s in user_stories)}")
    
    # Generate test cases
    logger.info("\n[2/4] Generating test cases...")
    all_test_cases = []
    
    for story in user_stories:
        test_cases = generate_test_cases_for_story(story)
        all_test_cases.extend(test_cases)
        logger.info(f"    [DONE] Generated {len(test_cases)} tests for {story['id']}")
    
    logger.info(f"\n  Total test cases generated: {len(all_test_cases)}")
    
    # Analyze coverage
    logger.info("\n[3/4] Analyzing coverage...")
    total_criteria = sum(len(s['acceptance_criteria']) for s in user_stories)
    positive_tests = len([tc for tc in all_test_cases if tc['type'] == 'positive'])
    negative_tests = len([tc for tc in all_test_cases if tc['type'] == 'negative'])
    edge_tests = len([tc for tc in all_test_cases if tc['type'] == 'edge_case'])
    
    coverage_analysis = {
        "total_acceptance_criteria": total_criteria,
        "test_cases_generated": len(all_test_cases),
        "coverage_ratio": len(all_test_cases) / total_criteria if total_criteria > 0 else 0,
        "coverage_percentage": 100.0,  # All criteria covered
        "test_type_distribution": {
            "positive": positive_tests,
            "negative": negative_tests,
            "edge_case": edge_tests
        },
        "priority_distribution": {
            "P1": len([tc for tc in all_test_cases if tc.get('priority') == 'P1']),
            "P2": len([tc for tc in all_test_cases if tc.get('priority') == 'P2']),
            "P3": len([tc for tc in all_test_cases if tc.get('priority') == 'P3'])
        },
        "automation_level": {
            "automated": len([tc for tc in all_test_cases if tc.get('automation_level') == 'automated']),
            "manual": len([tc for tc in all_test_cases if tc.get('automation_level') == 'manual'])
        }
    }
    
    logger.info(f"  Coverage: {coverage_analysis['coverage_percentage']}%")
    logger.info(f"  Positive tests: {positive_tests}")
    logger.info(f"  Negative tests: {negative_tests}")
    logger.info(f"  Edge case tests: {edge_tests}")
    
    # Build output
    output = {
        "epic_id": epic_id,
        "epic_title": epic_title,
        "generated_at": datetime.now().isoformat(),
        "generation_mode": "standalone",
        "test_cases": all_test_cases,
        "metadata": {
            "total_generated": len(all_test_cases),
            "user_stories_count": len(user_stories),
            "acceptance_criteria_count": total_criteria,
            "generator_version": "1.0.0",
            "generation_approach": "requirements_based"
        },
        "coverage_analysis": coverage_analysis,
        "recommendations": [
            "Review all P1 test cases for completeness",
            "Ensure negative tests cover all validation scenarios",
            "Consider performance testing for high-volume operations",
            "Add security testing for sensitive operations",
            "Validate integration points with external systems"
        ]
    }
    
    # Save to file
    logger.info("\n[4/4] Saving results...")
    output_path = "test_results/generated_test_cases_fulfillment.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"  [DONE] Saved to: {output_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("[DONE] TEST CASE GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Epic: {epic_title}")
    logger.info(f"Total Test Cases: {len(all_test_cases)}")
    logger.info(f"Coverage: {coverage_analysis['coverage_percentage']}%")
    logger.info(f"Output File: {output_path}")
    logger.info("="*80)
    
    # Print first test case as example
    logger.info("\n📝 Example Test Case (first one):")
    print(json.dumps(all_test_cases[0], indent=2))
    
    return output


if __name__ == "__main__":
    result = main()
    print(f"\n[DONE] Generated {result['metadata']['total_generated']} test cases successfully!")
