"""
Training data collector for fine-tuning HRM model.

Collects and prepares training examples from generated test cases
and user feedback for model fine-tuning.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

from ..requirements_parser.schemas import Epic, TestCase, UserFeedback
from ..convert_sqe_data import HRMDataConverter

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """
    Collects training data for fine-tuning HRM model.
    
    Aggregates examples from:
    - Generated test cases with human feedback
    - Corrected/edited test cases
    - New requirement-test case pairs
    """
    
    def __init__(self, output_dir: str = "training_data"):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to store training data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.converter = HRMDataConverter()
        self.examples = []
        
        logger.info(f"TrainingDataCollector initialized (output_dir={output_dir})")
    
    def collect_from_generation(
        self,
        requirements: List[Epic],
        generated_tests: List[TestCase],
        feedback: Optional[List[UserFeedback]] = None,
    ):
        """
        Collect training examples from generation results.
        
        Args:
            requirements: Original requirements (epics)
            generated_tests: Generated test cases
            feedback: Optional user feedback on test cases
        """
        logger.info(
            f"Collecting training data from {len(requirements)} epics, "
            f"{len(generated_tests)} test cases"
        )
        
        feedback_by_id = {}
        if feedback:
            feedback_by_id = {fb.test_case_id: fb for fb in feedback}
        
        for epic in requirements:
            for story in epic.user_stories:
                story_tests = [
                    tc for tc in generated_tests
                    if tc.source_story_id == story.id
                ]
                
                if not story_tests:
                    continue
                
                for test_case in story_tests:
                    fb = feedback_by_id.get(test_case.id)
                    
                    if fb and fb.rating < 3:
                        logger.debug(f"Skipping low-rated test case: {test_case.id}")
                        continue
                    
                    example = self._create_training_example(
                        epic=epic,
                        story=story,
                        test_case=test_case,
                        feedback=fb,
                    )
                    
                    self.examples.append(example)
        
        logger.info(f"Collected {len(self.examples)} training examples")
    
    def _create_training_example(
        self,
        epic: Epic,
        story: Any,
        test_case: TestCase,
        feedback: Optional[UserFeedback],
    ) -> Dict[str, Any]:
        """
        Create a single training example in HRM format.
        
        Args:
            epic: Parent epic
            story: User story
            test_case: Generated test case
            feedback: Optional feedback
            
        Returns:
            Training example dictionary
        """
        requirement_text = f"{epic.title} | {story.summary} | {story.description}"
        
        test_case_text = self._test_case_to_text(test_case)
        
        if feedback and feedback.corrections:
            test_case_text = feedback.corrections.get("completion", test_case_text)
        
        input_tokens = self.converter.text_to_tokens(requirement_text, max_len=100)
        output_tokens = self.converter.text_to_tokens(test_case_text, max_len=200)
        
        puzzle_id = hash(story.id) % 1000
        
        example = {
            "puzzle_id": puzzle_id,
            "input_sequence": input_tokens,
            "target_sequence": output_tokens,
            "solution_steps": [
                {"action": 0, "state": None} for _ in range(len(output_tokens))
            ],
            "metadata": {
                "source": "requirements_generation",
                "epic_id": epic.epic_id,
                "story_id": story.id,
                "test_case_id": test_case.id,
                "test_type": test_case.type.value,
                "priority": test_case.priority.value,
                "has_feedback": feedback is not None,
                "feedback_rating": feedback.rating if feedback else None,
                "timestamp": datetime.now().isoformat(),
            }
        }
        
        return example
    
    def _test_case_to_text(self, test_case: TestCase) -> str:
        """
        Convert test case to text representation.
        
        Args:
            test_case: Test case object
            
        Returns:
            Text representation
        """
        parts = []
        
        parts.append(f"Test: {test_case.description}")
        parts.append(f"Type: {test_case.type.value}")
        parts.append(f"Priority: {test_case.priority.value}")
        
        if test_case.preconditions:
            parts.append(f"Preconditions: {'; '.join(test_case.preconditions[:2])}")
        
        steps_text = "; ".join([step.action for step in test_case.test_steps[:3]])
        parts.append(f"Steps: {steps_text}")
        
        results_text = "; ".join([r.result for r in test_case.expected_results[:3]])
        parts.append(f"Expected: {results_text}")
        
        return " | ".join(parts)
    
    def augment_with_sqe_data(self, sqe_file_path: str):
        """
        Augment training data with existing SQE examples.
        
        Converts SQE format (prompt/completion) to HRM format (input_sequence/target_sequence).
        
        Args:
            sqe_file_path: Path to SQE JSONL file
        """
        logger.info(f"Augmenting with SQE data from {sqe_file_path}")
        
        sqe_path = Path(sqe_file_path)
        if not sqe_path.exists():
            logger.warning(f"SQE file not found: {sqe_file_path}")
            return
        
        sqe_count = 0
        with open(sqe_path, "r") as f:
            for line in f:
                sqe_example = json.loads(line)
                
                # Convert SQE format to HRM training format
                if "prompt" in sqe_example and "completion" in sqe_example:
                    # SQE format: {prompt, completion}
                    # Convert to HRM format: {puzzle_id, input_sequence, target_sequence, ...}
                    converted_example = self._convert_sqe_to_hrm_format(sqe_example)
                    self.examples.append(converted_example)
                    sqe_count += 1
                elif "input_sequence" in sqe_example:
                    # Already in HRM format
                    self.examples.append(sqe_example)
                    sqe_count += 1
                else:
                    logger.warning(f"Skipping SQE example with unknown format: {list(sqe_example.keys())}")
        
        logger.info(f"Added {sqe_count} SQE examples, total: {len(self.examples)}")
    
    def _convert_sqe_to_hrm_format(self, sqe_example: Dict) -> Dict[str, Any]:
        """
        Convert SQE format to HRM training format.
        
        Args:
            sqe_example: Example in SQE format {prompt, completion}
            
        Returns:
            Example in HRM format
        """
        # Convert text to tokens
        input_tokens = self.converter.text_to_tokens(sqe_example["prompt"], max_len=100)
        output_tokens = self.converter.text_to_tokens(sqe_example["completion"], max_len=200)
        
        # Create HRM format example
        puzzle_id = abs(hash(sqe_example.get("prompt", ""))) % 1000
        
        return {
            "puzzle_id": puzzle_id,
            "input_sequence": input_tokens,
            "target_sequence": output_tokens,
            "solution_steps": [
                {"action": 0, "state": None} for _ in range(len(output_tokens))
            ],
            "metadata": {
                "source": "sqe_augmentation",
                "original_format": "prompt_completion",
                "timestamp": datetime.now().isoformat(),
            }
        }
    
    def save_training_data(self, filename: str = "training_data.jsonl"):
        """
        Save collected training data to file.
        
        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving {len(self.examples)} training examples to {output_path}")
        
        with open(output_path, "w") as f:
            for example in self.examples:
                f.write(json.dumps(example) + "\n")
        
        logger.info(f"Training data saved to {output_path}")
        
        return str(output_path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected training data.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_examples": len(self.examples),
            "sources": {},
            "test_types": {},
            "priorities": {},
        }
        
        for example in self.examples:
            metadata = example.get("metadata", {})
            
            source = metadata.get("source", "unknown")
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            
            test_type = metadata.get("test_type", "unknown")
            stats["test_types"][test_type] = stats["test_types"].get(test_type, 0) + 1
            
            priority = metadata.get("priority", "unknown")
            stats["priorities"][priority] = stats["priorities"].get(priority, 0) + 1
        
        return stats

