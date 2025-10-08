"""
Evaluator for fine-tuned HRM models.

Compares fine-tuned model performance against base model
to measure improvement.
"""

import torch
from typing import Dict, Any, List
import logging

from ..models import HRMModel
from ..test_generator.generator import TestCaseGenerator
from ..requirements_parser import RequirementParser, Epic
from ..requirements_parser.schemas import TestContext

logger = logging.getLogger(__name__)


class FineTuningEvaluator:
    """
    Evaluates fine-tuned model performance.
    
    Compares:
    - Base model vs fine-tuned model
    - Test case quality metrics
    - Coverage improvements
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize evaluator.
        
        Args:
            device: Device for computation
        """
        self.device = device
        logger.info("FineTuningEvaluator initialized")
    
    def evaluate_improvement(
        self,
        base_model: HRMModel,
        fine_tuned_model: HRMModel,
        test_epics: List[Epic],
        config: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate improvement from fine-tuning.
        
        Args:
            base_model: Original base model
            fine_tuned_model: Fine-tuned model
            test_epics: Test epics for evaluation
            config: Generation configuration
            
        Returns:
            Comparison metrics
        """
        logger.info(f"Evaluating fine-tuning improvement on {len(test_epics)} epics")
        
        base_generator = TestCaseGenerator(base_model, self.device, config)
        fine_tuned_generator = TestCaseGenerator(fine_tuned_model, self.device, config)
        
        parser = RequirementParser()
        
        base_metrics = self._evaluate_model(base_generator, test_epics, parser)
        fine_tuned_metrics = self._evaluate_model(fine_tuned_generator, test_epics, parser)
        
        improvement = {
            "base_model": base_metrics,
            "fine_tuned_model": fine_tuned_metrics,
            "improvements": {
                "coverage_improvement": (
                    fine_tuned_metrics["avg_coverage"] - base_metrics["avg_coverage"]
                ),
                "test_cases_per_epic_change": (
                    fine_tuned_metrics["avg_test_cases_per_epic"] -
                    base_metrics["avg_test_cases_per_epic"]
                ),
            },
        }
        
        logger.info(f"Coverage improvement: {improvement['improvements']['coverage_improvement']:.2%}")
        
        return improvement
    
    def _evaluate_model(
        self,
        generator: TestCaseGenerator,
        epics: List[Epic],
        parser: RequirementParser,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            generator: Test case generator
            epics: Test epics
            parser: Requirement parser
            
        Returns:
            Evaluation metrics
        """
        total_test_cases = 0
        total_coverage = 0.0
        
        for epic in epics:
            test_contexts = parser.extract_test_contexts(epic)
            
            test_cases = generator.generate_test_cases(test_contexts)
            
            total_test_cases += len(test_cases)
            
            coverage = len(set(tc.source_story_id for tc in test_cases)) / len(epic.user_stories)
            total_coverage += coverage
        
        metrics = {
            "total_test_cases": total_test_cases,
            "avg_test_cases_per_epic": total_test_cases / len(epics) if epics else 0,
            "avg_coverage": total_coverage / len(epics) if epics else 0,
        }
        
        return metrics

