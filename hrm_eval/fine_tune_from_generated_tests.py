"""
Fine-tune HRM model from generated test cases.

Takes the test cases generated from the media fulfillment workflow and
uses them to fine-tune the HRM model, creating a feedback loop for
continuous improvement.

NO HARDCODING - Uses existing fine-tuning infrastructure.
"""

import torch
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any
import random

from .requirements_parser import Epic, UserStory, AcceptanceCriteria
from .requirements_parser.schemas import TestCase, UserFeedback, TestType, Priority, TestStep, ExpectedResult
from .models import HRMModel, HRMConfig
from .fine_tuning.data_collector import TrainingDataCollector
from .fine_tuning.fine_tuner import HRMFineTuner, FineTuningConfig
from .utils import load_config, load_checkpoint, setup_logging

setup_logging("INFO")
logger = logging.getLogger(__name__)


def load_generated_test_cases(generation_dir: Path) -> tuple:
    """
    Load generated test cases and requirements from a generation run.
    
    Args:
        generation_dir: Directory containing generation outputs
        
    Returns:
        Tuple of (epic, test_cases list)
    """
    logger.info(f"Loading generated data from {generation_dir}")
    
    # Load requirements epic
    epic_path = generation_dir / "requirements_epic.json"
    if not epic_path.exists():
        raise FileNotFoundError(f"Requirements epic not found: {epic_path}")
    
    with open(epic_path, "r") as f:
        epic_data = json.load(f)
    
    # Reconstruct Epic object
    epic = Epic(**epic_data)
    logger.info(f"Loaded epic: {epic.title} with {len(epic.user_stories)} user stories")
    
    # Load test cases
    test_cases_path = generation_dir / "test_cases.json"
    if not test_cases_path.exists():
        raise FileNotFoundError(f"Test cases not found: {test_cases_path}")
    
    with open(test_cases_path, "r") as f:
        test_cases_data = json.load(f)
    
    # Reconstruct TestCase objects
    test_cases = []
    for tc_data in test_cases_data:
        # Convert nested objects
        tc_data["type"] = TestType(tc_data["type"])
        tc_data["priority"] = Priority(tc_data["priority"])
        tc_data["test_steps"] = [TestStep(**step) for step in tc_data.get("test_steps", [])]
        tc_data["expected_results"] = [ExpectedResult(**er) for er in tc_data.get("expected_results", [])]
        
        test_case = TestCase(**tc_data)
        test_cases.append(test_case)
    
    logger.info(f"Loaded {len(test_cases)} test cases")
    
    return epic, test_cases


def generate_simulated_feedback(
    test_cases: List[TestCase],
    base_rating: float = 4.0,
    rating_variance: float = 0.5,
) -> List[UserFeedback]:
    """
    Generate simulated user feedback for test cases.
    
    In production, this would be real human feedback. For demonstration,
    we simulate ratings based on test case quality heuristics.
    
    Args:
        test_cases: List of test cases to rate
        base_rating: Base rating (1-5 scale)
        rating_variance: Variance in ratings
        
    Returns:
        List of UserFeedback objects
    """
    logger.info(f"Generating simulated feedback for {len(test_cases)} test cases")
    
    feedback_list = []
    
    for tc in test_cases:
        # Calculate quality score based on heuristics
        quality_score = base_rating
        
        # Bonus for comprehensive preconditions
        if len(tc.preconditions) >= 3:
            quality_score += 0.3
        
        # Bonus for detailed test steps
        if len(tc.test_steps) >= 3:
            quality_score += 0.3
        
        # Bonus for clear expected results
        if len(tc.expected_results) >= 2:
            quality_score += 0.2
        
        # Bonus for test data specification
        if tc.test_data:
            quality_score += 0.2
        
        # Add some randomness
        quality_score += random.uniform(-rating_variance, rating_variance)
        
        # Clamp to 1-5 range
        rating = max(1, min(5, int(round(quality_score))))
        
        feedback = UserFeedback(
            test_case_id=tc.id,
            rating=rating,
            feedback_text=f"Auto-generated feedback based on quality metrics",
            timestamp=datetime.now().isoformat(),
        )
        
        feedback_list.append(feedback)
        logger.debug(f"{tc.id}: rating={rating}")
    
    avg_rating = sum(f.rating for f in feedback_list) / len(feedback_list)
    logger.info(f"Generated feedback with average rating: {avg_rating:.2f}/5.0")
    
    return feedback_list


def split_training_data(
    examples: List[Dict[str, Any]],
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Split training data into train and validation sets.
    
    Args:
        examples: List of training examples
        validation_split: Fraction for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_examples, val_examples)
    """
    logger.info(f"Splitting data: {len(examples)} examples, {validation_split:.0%} validation")
    
    random.seed(seed)
    
    # Shuffle examples
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * (1 - validation_split))
    train_examples = shuffled[:split_idx]
    val_examples = shuffled[split_idx:]
    
    logger.info(f"Train: {len(train_examples)}, Validation: {len(val_examples)}")
    
    return train_examples, val_examples


def save_data_split(
    train_examples: List[Dict],
    val_examples: List[Dict],
    output_dir: Path,
) -> tuple:
    """
    Save train and validation data to JSONL files.
    
    Args:
        train_examples: Training examples
        val_examples: Validation examples
        output_dir: Output directory
        
    Returns:
        Tuple of (train_path, val_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "training_data.jsonl"
    val_path = output_dir / "validation_data.jsonl"
    
    logger.info(f"Saving training data to {train_path}")
    with open(train_path, "w") as f:
        for example in train_examples:
            f.write(json.dumps(example) + "\n")
    
    logger.info(f"Saving validation data to {val_path}")
    with open(val_path, "w") as f:
        for example in val_examples:
            f.write(json.dumps(example) + "\n")
    
    return str(train_path), str(val_path)


def run_fine_tuning_workflow(generation_dir_name: str = "media_fulfillment_20251007_220527") -> None:
    """
    Complete fine-tuning workflow.
    
    Args:
        generation_dir_name: Name of the generation directory to use
    """
    logger.info("=" * 80)
    logger.info("HRM Model Fine-Tuning from Generated Test Cases")
    logger.info("=" * 80)
    logger.info("\nUSING EXISTING FINE-TUNING INFRASTRUCTURE")
    logger.info("Creating feedback loop: Generated Tests → Training Data → Fine-tuned Model\n")
    
    base_path = Path(__file__).parent
    generation_dir = base_path / "generated_tests" / generation_dir_name
    
    if not generation_dir.exists():
        logger.error(f"Generation directory not found: {generation_dir}")
        logger.error("Please run the test generation workflow first")
        return
    
    # Step 1: Load generated data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Load Generated Test Cases and Requirements")
    logger.info("=" * 80 + "\n")
    
    epic, test_cases = load_generated_test_cases(generation_dir)
    
    logger.info(f"Epic: {epic.title}")
    logger.info(f"User Stories: {len(epic.user_stories)}")
    logger.info(f"Test Cases: {len(test_cases)}")
    
    # Step 2: Generate feedback
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Generate Simulated User Feedback")
    logger.info("=" * 80 + "\n")
    logger.info("Note: In production, this would be real human feedback")
    
    feedback = generate_simulated_feedback(test_cases, base_rating=4.0)
    
    # Save feedback
    feedback_dir = base_path / "training_data" / "media_fulfillment_fine_tuning"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    
    feedback_path = feedback_dir / "feedback_simulated.json"
    with open(feedback_path, "w") as f:
        json.dump([fb.dict() for fb in feedback], f, indent=2)
    logger.info(f"Saved feedback to {feedback_path}")
    
    # Step 3: Prepare training data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Prepare Training Data")
    logger.info("=" * 80 + "\n")
    
    collector = TrainingDataCollector(output_dir=str(feedback_dir))
    
    collector.collect_from_generation(
        requirements=[epic],
        generated_tests=test_cases,
        feedback=feedback,
    )
    
    # Optionally augment with existing SQE data
    sqe_data_path = base_path.parent / "sqe_agent_real_data.jsonl"
    if sqe_data_path.exists():
        logger.info(f"Augmenting with existing SQE data from {sqe_data_path}")
        collector.augment_with_sqe_data(str(sqe_data_path))
    
    # Get statistics
    stats = collector.get_statistics()
    logger.info("\nTraining Data Statistics:")
    logger.info(f"  Total Examples: {stats['total_examples']}")
    logger.info(f"  Sources: {stats['sources']}")
    logger.info(f"  Test Types: {stats['test_types']}")
    logger.info(f"  Priorities: {stats['priorities']}")
    
    # Save statistics
    stats_path = feedback_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nSaved statistics to {stats_path}")
    
    # Step 4: Split data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Split into Train/Validation Sets")
    logger.info("=" * 80 + "\n")
    
    train_examples, val_examples = split_training_data(
        collector.examples,
        validation_split=0.2,
    )
    
    train_path, val_path = save_data_split(
        train_examples,
        val_examples,
        feedback_dir,
    )
    
    logger.info(f"Training data: {train_path}")
    logger.info(f"Validation data: {val_path}")
    
    # Step 5: Load base model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Load Base HRM Model")
    logger.info("=" * 80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    checkpoint_path = base_path.parent / "checkpoints_hrm_v9_optimized_step_7566"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    config = load_config(
        model_config_path=base_path / "configs" / "model_config.yaml",
        eval_config_path=base_path / "configs" / "eval_config.yaml"
    )
    
    hrm_config = HRMConfig.from_yaml_config(config)
    model = HRMModel(hrm_config)
    
    checkpoint = load_checkpoint(str(checkpoint_path))
    
    # Handle checkpoint format
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Clean state dict
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ["model.inner.", "model.", "module."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        if "embedding_weight" in new_key:
            new_key = new_key.replace("embedding_weight", "weight")
        cleaned_state_dict[new_key] = value
    
    model.load_state_dict(cleaned_state_dict, strict=False)
    
    logger.info("Base model loaded successfully")
    
    # Step 6: Configure fine-tuning
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Configure Fine-Tuning")
    logger.info("=" * 80 + "\n")
    
    ft_config = FineTuningConfig(
        learning_rate=1e-5,
        epochs=3,
        batch_size=8,
        validation_split=0.2,
        warmup_steps=50,
        gradient_clip=1.0,
        save_every_n_steps=100,
        eval_every_n_steps=50,
    )
    
    logger.info("Fine-tuning configuration:")
    logger.info(f"  Learning Rate: {ft_config.learning_rate}")
    logger.info(f"  Epochs: {ft_config.epochs}")
    logger.info(f"  Batch Size: {ft_config.batch_size}")
    logger.info(f"  Warmup Steps: {ft_config.warmup_steps}")
    logger.info(f"  Gradient Clip: {ft_config.gradient_clip}")
    
    # Step 7: Run fine-tuning
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Run Fine-Tuning")
    logger.info("=" * 80 + "\n")
    
    checkpoint_dir = base_path / "fine_tuned_checkpoints" / "media_fulfillment"
    
    fine_tuner = HRMFineTuner(
        model=model,
        device=device,
        config=ft_config,
        output_dir=str(checkpoint_dir),
    )
    
    logger.info("Starting fine-tuning process...")
    logger.info("This may take several minutes depending on data size and hardware\n")
    
    try:
        training_results = fine_tuner.fine_tune(
            training_data_path=train_path,
            val_data_path=val_path,
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("Fine-Tuning Complete!")
        logger.info("=" * 80 + "\n")
        
        logger.info("Training Results:")
        logger.info(f"  Total Steps: {training_results['total_steps']}")
        logger.info(f"  Final Train Loss: {training_results['train_loss_history'][-1]:.4f}")
        if training_results['val_loss_history']:
            logger.info(f"  Final Val Loss: {training_results['val_loss_history'][-1]:.4f}")
            logger.info(f"  Best Val Loss: {training_results['best_val_loss']:.4f}")
        logger.info(f"  Best Checkpoint: {training_results['best_checkpoint']}")
        
        # Save training results
        results_path = checkpoint_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"\nSaved training results to {results_path}")
        
        # Step 8: Summary
        logger.info("\n" + "=" * 80)
        logger.info("Fine-Tuning Workflow Complete!")
        logger.info("=" * 80 + "\n")
        
        logger.info("Outputs saved to:")
        logger.info(f"  Training Data: {feedback_dir}")
        logger.info(f"  Checkpoints: {checkpoint_dir}")
        logger.info(f"  Best Model: {training_results['best_checkpoint']}")
        
        logger.info("\nNext steps:")
        logger.info("1. Evaluate fine-tuned model on new requirements")
        logger.info("2. Compare test case quality: base vs fine-tuned")
        logger.info("3. Deploy fine-tuned model if improvements verified")
        logger.info("4. Collect more feedback and iterate")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_fine_tuning_workflow()
