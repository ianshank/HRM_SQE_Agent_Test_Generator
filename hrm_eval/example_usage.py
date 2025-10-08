"""
Example usage of the Requirements-to-Test-Cases system.

Demonstrates how to generate test cases from structured requirements
using the HRM v9 Optimized model - NO hardcoded test generation.
"""

import torch
import logging
from pathlib import Path
import json

from requirements_parser import (
    RequirementParser,
    Epic,
    UserStory,
    AcceptanceCriteria,
)
from models import HRMModel
from test_generator import TestCaseGenerator
from test_generator.template_engine import TestCaseTemplate
from utils import load_config, load_checkpoint, setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


def create_example_epic() -> Epic:
    """Create example epic for demonstration."""
    epic = Epic(
        epic_id="EPIC-DEMO-001",
        title="Media Asset Ingestion and Delivery Workflow",
        user_stories=[
            UserStory(
                id="US-001",
                summary="Asset Receipt and Verification",
                description=(
                    "As a media operator, I want to ingest media assets (video, image, audio) "
                    "into the system so that they are prepared for processing."
                ),
                acceptance_criteria=[
                    AcceptanceCriteria(
                        criteria="Uploaded assets are accessible in staging area with metadata"
                    ),
                    AcceptanceCriteria(
                        criteria="System rejects unsupported file formats with error messages"
                    ),
                    AcceptanceCriteria(
                        criteria="Virus scan and checksum results are recorded"
                    ),
                ],
                tech_stack=["FastAPI", "S3", "ClamAV"],
            ),
            UserStory(
                id="US-002",
                summary="Automated Metadata Extraction",
                description=(
                    "As a content manager, I want automated extraction of technical metadata "
                    "so each asset is searchable without manual entry."
                ),
                acceptance_criteria=[
                    AcceptanceCriteria(
                        criteria="All ingested assets have technical metadata populated"
                    ),
                    AcceptanceCriteria(
                        criteria="Search queries filter assets using extracted metadata"
                    ),
                ],
                tech_stack=["FFmpeg", "Elasticsearch"],
            ),
        ],
        tech_stack=["Python", "RabbitMQ", "PostgreSQL", "Docker"],
        architecture="Microservices with async processing",
    )
    
    return epic


def main():
    """Main demonstration function."""
    logger.info("=" * 80)
    logger.info("Requirements to Test Cases - Example Usage")
    logger.info("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    base_path = Path(__file__).parent
    checkpoint_path = base_path.parent / "checkpoints_hrm_v9_optimized_step_7566"
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Please ensure the checkpoint file is in the correct location")
        return
    
    logger.info(f"Loading model from: {checkpoint_path}")
    
    config = load_config(base_path / "configs" / "model_config.yaml")
    
    model = HRMModel(
        vocab_size=config["model"]["vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        num_h_layers=config["model"]["num_h_layers"],
        num_l_layers=config["model"]["num_l_layers"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        dropout=config["model"]["dropout"],
        puzzle_vocab_size=config["model"]["puzzle_vocab_size"],
        q_head_actions=config["model"]["q_head_actions"],
    )
    
    checkpoint = load_checkpoint(str(checkpoint_path))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    gen_config = load_config(base_path / "configs" / "test_generation_config.yaml")
    
    generator = TestCaseGenerator(
        model=model,
        device=device,
        config=gen_config,
    )
    
    logger.info("Generator initialized - using REAL HRM model (no dummy logic)")
    
    logger.info("\nCreating example epic...")
    epic = create_example_epic()
    
    logger.info(f"Epic: {epic.title}")
    logger.info(f"User Stories: {len(epic.user_stories)}")
    for story in epic.user_stories:
        logger.info(f"  - {story.id}: {story.summary}")
        logger.info(f"    Acceptance Criteria: {len(story.acceptance_criteria)}")
    
    logger.info("\nParsing requirements and extracting test contexts...")
    parser = RequirementParser()
    
    test_contexts = parser.extract_test_contexts(epic)
    logger.info(f"Extracted {len(test_contexts)} test contexts")
    
    coverage_analysis = parser.get_coverage_analysis(epic)
    logger.info(f"Testability Score: {coverage_analysis['testability_score']:.2%}")
    
    logger.info("\nGenerating test cases using HRM model...")
    logger.info("(This performs ACTUAL model inference - no hardcoded outputs)")
    
    test_cases = generator.generate_test_cases(test_contexts)
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Generated {len(test_cases)} test cases")
    logger.info(f"{'=' * 80}\n")
    
    test_type_counts = {}
    priority_counts = {}
    
    for tc in test_cases:
        test_type_counts[tc.type.value] = test_type_counts.get(tc.type.value, 0) + 1
        priority_counts[tc.priority.value] = priority_counts.get(tc.priority.value, 0) + 1
    
    logger.info("Test Type Distribution:")
    for test_type, count in sorted(test_type_counts.items()):
        logger.info(f"  {test_type}: {count}")
    
    logger.info("\nPriority Distribution:")
    for priority, count in sorted(priority_counts.items()):
        logger.info(f"  {priority}: {count}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info("Sample Test Cases:")
    logger.info(f"{'=' * 80}\n")
    
    template = TestCaseTemplate()
    
    for idx, tc in enumerate(test_cases[:3], 1):
        logger.info(f"Test Case {idx}: {tc.id}")
        logger.info(f"  Type: {tc.type.value}")
        logger.info(f"  Priority: {tc.priority.value}")
        logger.info(f"  Description: {tc.description}")
        logger.info(f"  Preconditions: {len(tc.preconditions)}")
        logger.info(f"  Test Steps: {len(tc.test_steps)}")
        logger.info(f"  Expected Results: {len(tc.expected_results)}")
        logger.info(f"  Labels: {', '.join(tc.labels)}")
        logger.info("")
    
    output_dir = base_path / "generated_tests"
    output_dir.mkdir(exist_ok=True)
    
    json_output = output_dir / "test_cases.json"
    with open(json_output, "w") as f:
        json.dump(
            [tc.dict() for tc in test_cases],
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Saved test cases to: {json_output}")
    
    markdown_output = output_dir / "test_cases.md"
    markdown_content = template.generate_markdown_output(test_cases)
    with open(markdown_output, "w") as f:
        f.write(markdown_content)
    logger.info(f"Saved markdown report to: {markdown_output}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Example complete!")
    logger.info("=" * 80)
    
    logger.info("\nNext steps:")
    logger.info("1. Review generated test cases in generated_tests/")
    logger.info("2. Try the API: uvicorn api_service.main:app --reload")
    logger.info("3. Fine-tune the model with your own data")
    logger.info("4. Integrate with agent systems for agent-based workflows")


if __name__ == "__main__":
    main()

