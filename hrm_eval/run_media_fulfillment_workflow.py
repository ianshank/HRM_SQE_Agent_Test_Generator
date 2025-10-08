"""
Media Fulfillment Requirements - Test Generation Workflow

Runs the user's media fulfillment requirements through the existing
test generation workflow WITHOUT hardcoding or dummy logic.

Uses the established pipeline:
1. Structure requirements as Epic
2. Parse with RequirementParser
3. Generate tests with TestCaseGenerator (REAL HRM model)
4. Output results
"""

import torch
import logging
from pathlib import Path
import json
from datetime import datetime

from .requirements_parser import (
    RequirementParser,
    RequirementValidator,
    Epic,
    UserStory,
    AcceptanceCriteria,
)
from .models import HRMModel, HRMConfig
from .test_generator.generator import TestCaseGenerator
from .test_generator.template_engine import TestCaseTemplate
from .utils import load_config, load_checkpoint, setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


def create_media_fulfillment_epic() -> Epic:
    """
    Create Epic from media fulfillment requirements.
    
    Structures the 5 user stories provided by the user into
    the Epic schema required by the workflow.
    
    Returns:
        Epic object with all user stories and metadata
    """
    logger.info("Structuring media fulfillment requirements into Epic format")
    
    epic = Epic(
        epic_id="EPIC-MEDIA-FULFILLMENT-001",
        title="Media Content Ingestion, Processing, and Delivery Workflow",
        user_stories=[
            UserStory(
                id="US-MF-001",
                summary="Content Ingestion",
                description=(
                    "As an external producer, I want to upload completed media files and metadata "
                    "so that the media company can begin processing content."
                ),
                acceptance_criteria=[
                    AcceptanceCriteria(
                        criteria="Must accept batch uploads of media files (e.g., video, audio, subtitles)"
                    ),
                    AcceptanceCriteria(
                        criteria="Metadata input forms must enforce required fields: title, description, "
                        "episode/season, format, duration, license information"
                    ),
                    AcceptanceCriteria(
                        criteria="Upon submission, confirmation email is sent to the uploader"
                    ),
                ],
                tech_stack=["FastAPI", "S3", "SMTP", "React", "PostgreSQL"],
            ),
            UserStory(
                id="US-MF-002",
                summary="Metadata Validation and Enrichment",
                description=(
                    "As a metadata specialist, I want to review and enrich submitted metadata to "
                    "ensure accuracy and completeness for downstream processing."
                ),
                acceptance_criteria=[
                    AcceptanceCriteria(
                        criteria="Automatic checks for required metadata fields"
                    ),
                    AcceptanceCriteria(
                        criteria="UI for editors to add or correct metadata"
                    ),
                    AcceptanceCriteria(
                        criteria="Traceable change log for metadata updates"
                    ),
                ],
                tech_stack=["React", "PostgreSQL", "Elasticsearch", "Audit Log Service"],
            ),
            UserStory(
                id="US-MF-003",
                summary="Content Quality Control",
                description=(
                    "As a QC operator, I want to perform technical and editorial quality checks on "
                    "ingested content."
                ),
                acceptance_criteria=[
                    AcceptanceCriteria(
                        criteria="Checklist for audio/video quality, rights compliance, subtitle accuracy"
                    ),
                    AcceptanceCriteria(
                        criteria="Option to reject, accept, or request re-upload with feedback"
                    ),
                    AcceptanceCriteria(
                        criteria="QC results visible to production stakeholders"
                    ),
                ],
                tech_stack=["FFmpeg", "React", "PostgreSQL", "Notification Service"],
            ),
            UserStory(
                id="US-MF-004",
                summary="Packaging and Rights Management",
                description=(
                    "As a packaging manager, I want to package approved content with the correct "
                    "metadata, graphics, and rights information for target platforms."
                ),
                acceptance_criteria=[
                    AcceptanceCriteria(
                        criteria="Select delivery profiles (e.g., OTT, broadcast, VOD)"
                    ),
                    AcceptanceCriteria(
                        criteria="Populate rights windows and restrictions"
                    ),
                    AcceptanceCriteria(
                        criteria="Trigger packaging and encryption jobs"
                    ),
                ],
                tech_stack=["FFmpeg", "AWS Media Services", "DRM", "PostgreSQL"],
            ),
            UserStory(
                id="US-MF-005",
                summary="Delivery and Confirmation",
                description=(
                    "As a fulfillment coordinator, I want to track delivery status and confirmation for "
                    "each order so that stakeholders know when content is live."
                ),
                acceptance_criteria=[
                    AcceptanceCriteria(
                        criteria="Display delivery progress and error alerts"
                    ),
                    AcceptanceCriteria(
                        criteria="Automated notifications sent when content is delivered and published"
                    ),
                    AcceptanceCriteria(
                        criteria="Generate and store delivery confirmation receipts"
                    ),
                ],
                tech_stack=["React", "PostgreSQL", "CDN APIs", "Notification Service"],
            ),
        ],
        tech_stack=[
            "Python",
            "FastAPI",
            "React",
            "PostgreSQL",
            "S3",
            "FFmpeg",
            "RabbitMQ",
            "Docker",
            "Kubernetes",
            "Elasticsearch",
        ],
        architecture="Microservices with async processing and event-driven workflows",
    )
    
    logger.info(f"Created Epic: {epic.title}")
    logger.info(f"User Stories: {len(epic.user_stories)}")
    
    for story in epic.user_stories:
        logger.info(f"  - {story.id}: {story.summary}")
        logger.info(f"    Acceptance Criteria: {len(story.acceptance_criteria)}")
        logger.info(f"    Tech Stack: {', '.join(story.tech_stack)}")
    
    return epic


def validate_requirements(epic: Epic) -> None:
    """
    Validate requirements quality before test generation.
    
    Args:
        epic: Epic to validate
    """
    logger.info("\nValidating requirements quality...")
    
    validator = RequirementValidator()
    
    is_valid, validation_issues = validator.validate_epic(epic)
    
    if validation_issues:
        logger.warning(f"Validation issues found: {len(validation_issues)}")
        for issue in validation_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All requirements are valid")
    
    testability_score, testability_report = validator.check_testability(epic)
    
    logger.info(f"\nTestability Score: {testability_score:.2%}")
    logger.info("\nTestability Report:")
    for key, value in testability_report.items():
        logger.info(f"  {key}: {value}")


def run_workflow() -> None:
    """
    Main workflow execution.
    
    Follows the pattern from example_usage.py:
    1. Load HRM model
    2. Create Epic from requirements
    3. Validate requirements
    4. Extract test contexts
    5. Generate test cases (REAL HRM inference)
    6. Save outputs
    """
    logger.info("=" * 80)
    logger.info("Media Fulfillment Requirements - Test Generation Workflow")
    logger.info("=" * 80)
    logger.info("\nUSING EXISTING WORKFLOW - NO HARDCODED TEST GENERATION")
    logger.info("Test cases will be generated by the HRM v9 Optimized model\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    base_path = Path(__file__).parent
    checkpoint_path = base_path.parent / "checkpoints_hrm_v9_optimized_step_7566"
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please ensure the checkpoint file is in the correct location")
        return
    
    logger.info(f"Loading HRM model from: {checkpoint_path}")
    
    config = load_config(
        model_config_path=base_path / "configs" / "model_config.yaml",
        eval_config_path=base_path / "configs" / "eval_config.yaml"
    )
    
    hrm_config = HRMConfig.from_yaml_config(config)
    model = HRMModel(hrm_config)
    
    checkpoint = load_checkpoint(str(checkpoint_path))
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Assume checkpoint is the state dict itself
        state_dict = checkpoint
    
    # Strip prefixes and map keys if present
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove common prefixes
        new_key = key
        for prefix in ["model.inner.", "model.", "module."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        
        # Map specific keys that have different names
        if "embedding_weight" in new_key:
            new_key = new_key.replace("embedding_weight", "weight")
        
        cleaned_state_dict[new_key] = value
    
    # Load with strict=False to handle minor differences
    result = model.load_state_dict(cleaned_state_dict, strict=False)
    
    if result.missing_keys:
        logger.warning(f"Missing keys in checkpoint: {len(result.missing_keys)} keys")
        logger.debug(f"Missing: {result.missing_keys[:5]}")  # Show first 5
    if result.unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {len(result.unexpected_keys)} keys")
        logger.debug(f"Unexpected: {result.unexpected_keys[:5]}")  # Show first 5
    
    model.to(device)
    model.eval()
    
    logger.info("HRM model loaded successfully")
    
    generator = TestCaseGenerator(
        model=model,
        device=device,
        config=config,
    )
    
    logger.info("TestCaseGenerator initialized with REAL HRM model")
    
    epic = create_media_fulfillment_epic()
    
    validate_requirements(epic)
    
    logger.info("\nParsing requirements and extracting test contexts...")
    parser = RequirementParser()
    
    test_contexts = parser.extract_test_contexts(epic)
    logger.info(f"Extracted {len(test_contexts)} test contexts from requirements")
    
    for idx, context in enumerate(test_contexts, 1):
        logger.info(f"  Context {idx}: {context.test_type.value} - {context.story_id}")
    
    coverage_analysis = parser.get_coverage_analysis(epic)
    logger.info(f"\nCoverage Analysis:")
    logger.info(f"  Testability Score: {coverage_analysis['testability_score']:.2%}")
    logger.info(f"  Total Acceptance Criteria: {coverage_analysis['total_acceptance_criteria']}")
    logger.info(f"  Testable Contexts: {len(test_contexts)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Generating test cases using HRM model...")
    logger.info("(This performs ACTUAL model inference - no hardcoded outputs)")
    logger.info("=" * 80 + "\n")
    
    test_cases = generator.generate_test_cases(test_contexts)
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Successfully generated {len(test_cases)} test cases")
    logger.info(f"{'=' * 80}\n")
    
    test_type_counts = {}
    priority_counts = {}
    story_coverage = {}
    
    for tc in test_cases:
        test_type_counts[tc.type.value] = test_type_counts.get(tc.type.value, 0) + 1
        priority_counts[tc.priority.value] = priority_counts.get(tc.priority.value, 0) + 1
        
        if tc.source_story_id:
            story_coverage[tc.source_story_id] = story_coverage.get(tc.source_story_id, 0) + 1
    
    logger.info("Test Generation Summary:")
    logger.info(f"  Total Test Cases: {len(test_cases)}")
    logger.info(f"\n  Test Type Distribution:")
    for test_type, count in sorted(test_type_counts.items()):
        logger.info(f"    {test_type.capitalize()}: {count}")
    
    logger.info(f"\n  Priority Distribution:")
    for priority, count in sorted(priority_counts.items()):
        logger.info(f"    {priority}: {count}")
    
    logger.info(f"\n  Coverage by User Story:")
    for story_id, count in sorted(story_coverage.items()):
        logger.info(f"    {story_id}: {count} test cases")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("Sample Test Cases:")
    logger.info(f"{'=' * 80}\n")
    
    for idx, tc in enumerate(test_cases[:3], 1):
        logger.info(f"Test Case {idx}: {tc.id}")
        logger.info(f"  Type: {tc.type.value}")
        logger.info(f"  Priority: {tc.priority.value}")
        logger.info(f"  Description: {tc.description}")
        logger.info(f"  Source Story: {tc.source_story_id}")
        logger.info(f"  Preconditions: {len(tc.preconditions)}")
        logger.info(f"  Test Steps: {len(tc.test_steps)}")
        logger.info(f"  Expected Results: {len(tc.expected_results)}")
        logger.info(f"  Labels: {', '.join(tc.labels)}")
        logger.info("")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_path / "generated_tests" / f"media_fulfillment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_output = output_dir / "test_cases.json"
    with open(json_output, "w") as f:
        json.dump(
            [tc.dict() for tc in test_cases],
            f,
            indent=2,
            default=str,
        )
    logger.info(f"\nSaved test cases (JSON): {json_output}")
    
    template = TestCaseTemplate()
    markdown_output = output_dir / "test_cases.md"
    markdown_content = template.generate_markdown_output(test_cases)
    with open(markdown_output, "w") as f:
        f.write(markdown_content)
    logger.info(f"Saved test cases (Markdown): {markdown_output}")
    
    epic_output = output_dir / "requirements_epic.json"
    with open(epic_output, "w") as f:
        json.dump(epic.dict(), f, indent=2, default=str)
    logger.info(f"Saved original requirements: {epic_output}")
    
    summary = {
        "timestamp": timestamp,
        "epic_id": epic.epic_id,
        "epic_title": epic.title,
        "num_user_stories": len(epic.user_stories),
        "num_test_contexts": len(test_contexts),
        "num_test_cases": len(test_cases),
        "test_type_distribution": test_type_counts,
        "priority_distribution": priority_counts,
        "story_coverage": story_coverage,
        "testability_score": coverage_analysis['testability_score'],
        "model_checkpoint": "checkpoints_hrm_v9_optimized_step_7566",
        "device": str(device),
    }
    
    summary_output = output_dir / "generation_summary.json"
    with open(summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved generation summary: {summary_output}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Workflow Complete!")
    logger.info("=" * 80)
    logger.info(f"\nAll outputs saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review generated test cases in the output directory")
    logger.info("2. Validate test case quality and coverage")
    logger.info("3. Import into test management system if needed")
    logger.info("4. Provide feedback for model fine-tuning")


if __name__ == "__main__":
    run_workflow()
