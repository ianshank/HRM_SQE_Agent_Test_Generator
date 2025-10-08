"""
Workflow manager for multi-agent orchestration.

Manages coordination between HRM, SQE, and RAG components.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Manages multi-agent workflows for test generation.
    
    Coordinates between:
    - HRM model for test generation
    - SQE agent for orchestration
    - RAG for context retrieval
    - Vector indexer for knowledge base updates
    """
    
    def __init__(
        self,
        hybrid_generator: Any,
        vector_indexer: Optional[Any] = None,
        auto_index: bool = True,
    ):
        """
        Initialize workflow manager.
        
        Args:
            hybrid_generator: Hybrid test generator
            vector_indexer: Optional vector indexer
            auto_index: Automatically index generated tests
        """
        self.hybrid_generator = hybrid_generator
        self.vector_indexer = vector_indexer
        self.auto_index = auto_index
        
        logger.info(
            f"WorkflowManager initialized "
            f"(auto_index={auto_index}, indexer={vector_indexer is not None})"
        )
    
    def execute_workflow(
        self,
        requirements: Dict[str, Any],
        workflow_type: str = "full",  # "full", "generate_only", "validate_only"
    ) -> Dict[str, Any]:
        """
        Execute complete workflow.
        
        Args:
            requirements: Requirements (Epic dict)
            workflow_type: Type of workflow to execute
            
        Returns:
            Workflow result with test cases and metadata
        """
        logger.info(f"Executing {workflow_type} workflow")
        
        if workflow_type == "validate_only":
            return self._validate_workflow(requirements)
        elif workflow_type == "generate_only":
            return self._generate_workflow(requirements)
        else:
            return self._full_workflow(requirements)
    
    def _full_workflow(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full workflow: validate → generate → analyze → index.
        
        Args:
            requirements: Requirements
            
        Returns:
            Complete workflow result
        """
        logger.info("Executing full workflow")
        
        result = {
            "workflow_type": "full",
            "steps": [],
            "test_cases": [],
            "status": "in_progress",
        }
        
        validation = self._validate_requirements(requirements)
        result["steps"].append({
            "step": "validation",
            "status": "complete",
            "result": validation,
        })
        
        if not validation.get("is_valid"):
            logger.warning("Requirements validation failed")
            result["status"] = "validation_failed"
            return result
        
        generation = self.hybrid_generator.generate(requirements)
        result["test_cases"] = generation.get("test_cases", [])
        result["steps"].append({
            "step": "generation",
            "status": "complete",
            "result": generation.get("metadata", {}),
        })
        
        if self.auto_index and self.vector_indexer and result["test_cases"]:
            try:
                self.vector_indexer.index_test_cases(
                    result["test_cases"],
                    batch_size=100,
                    show_progress=False,
                )
                
                result["steps"].append({
                    "step": "indexing",
                    "status": "complete",
                    "indexed_count": len(result["test_cases"]),
                })
                
                logger.info(f"Indexed {len(result['test_cases'])} test cases")
                
            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                result["steps"].append({
                    "step": "indexing",
                    "status": "failed",
                    "error": str(e),
                })
        
        result["status"] = "complete"
        
        logger.info(
            f"Full workflow complete: {len(result['test_cases'])} test cases generated"
        )
        
        return result
    
    def _generate_workflow(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generation workflow only."""
        logger.info("Executing generation-only workflow")
        
        generation = self.hybrid_generator.generate(requirements)
        
        return {
            "workflow_type": "generate_only",
            "test_cases": generation.get("test_cases", []),
            "metadata": generation.get("metadata", {}),
            "status": "complete",
        }
    
    def _validate_workflow(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation workflow only."""
        logger.info("Executing validation-only workflow")
        
        validation = self._validate_requirements(requirements)
        
        return {
            "workflow_type": "validate_only",
            "validation": validation,
            "status": "complete",
        }
    
    def _validate_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate requirements."""
        try:
            from ..requirements_parser import RequirementValidator
            
            validator = RequirementValidator(strict_mode=False)
            
            is_valid, issues = validator.validate_epic(requirements)
            testability_score, report = validator.check_testability(requirements)
            
            return {
                "is_valid": is_valid,
                "issues": issues,
                "testability_score": testability_score,
                "report": report,
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return {
                "is_valid": False,
                "error": str(e),
            }
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics."""
        stats = {
            "auto_index_enabled": self.auto_index,
            "indexer_available": self.vector_indexer is not None,
            "hybrid_generator_available": self.hybrid_generator is not None,
        }
        
        if self.vector_indexer:
            stats["indexer_stats"] = self.vector_indexer.get_indexing_stats()
        
        return stats
