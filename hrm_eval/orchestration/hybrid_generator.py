"""
Hybrid test generator combining HRM, SQE, and RAG.

NO HARDCODING - all generation through actual models/workflows.
"""

from typing import Dict, Any, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


class HybridTestGenerator:
    """
    Hybrid test generator combining:
    - HRM model for test case generation
    - RAG for historical context
    - SQE agent for workflow orchestration
    
    NO HARDCODING - all generation through models/workflows.
    """
    
    def __init__(
        self,
        hrm_generator: Any,
        sqe_agent: Optional[Any] = None,
        rag_retriever: Optional[Any] = None,
        mode: str = "hybrid",  # "hrm_only", "sqe_only", "hybrid"
        merge_strategy: str = "weighted",  # "weighted", "union", "intersection"
        hrm_weight: float = 0.6,
        sqe_weight: float = 0.4,
    ):
        """
        Initialize hybrid test generator.
        
        Args:
            hrm_generator: HRM test case generator
            sqe_agent: Optional SQE agent
            rag_retriever: Optional RAG retriever
            mode: Generation mode
            merge_strategy: How to merge results
            hrm_weight: Weight for HRM results in weighted merge
            sqe_weight: Weight for SQE results in weighted merge
        """
        self.hrm_generator = hrm_generator
        self.sqe_agent = sqe_agent
        self.rag_retriever = rag_retriever
        self.mode = mode
        self.merge_strategy = merge_strategy
        self.hrm_weight = hrm_weight
        self.sqe_weight = sqe_weight
        
        logger.info(
            f"HybridTestGenerator initialized "
            f"(mode={mode}, strategy={merge_strategy}, "
            f"HRM={hrm_generator is not None}, SQE={sqe_agent is not None}, RAG={rag_retriever is not None})"
        )
    
    def generate(
        self,
        requirements: Dict[str, Any],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate test cases using hybrid approach.
        
        Args:
            requirements: Structured requirements (Epic dict)
            context: Optional additional context
            
        Returns:
            Generated test cases with metadata
        """
        logger.info(f"Generating test cases in {self.mode} mode")
        
        start_time = time.time()
        
        try:
            if self.mode == "hrm_only":
                result = self._generate_hrm_only(requirements, context)
            elif self.mode == "sqe_only":
                result = self._generate_sqe_only(requirements, context)
            else:
                result = self._generate_hybrid(requirements, context)
            
            generation_time = time.time() - start_time
            result["metadata"]["generation_time_seconds"] = generation_time
            
            logger.info(
                f"Generation complete: {len(result.get('test_cases', []))} tests "
                f"in {generation_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid generation failed: {e}", exc_info=True)
            return {
                "test_cases": [],
                "metadata": {
                    "status": "error",
                    "error": str(e),
                },
            }
    
    def _generate_hrm_only(
        self,
        requirements: Dict[str, Any],
        context: Optional[str],
    ) -> Dict[str, Any]:
        """Generate using HRM model only."""
        logger.info("Generating with HRM only")
        
        from ..requirements_parser import RequirementParser
        from ..requirements_parser.schemas import Epic
        
        parser = RequirementParser()
        
        # Convert dict to Epic if needed
        if isinstance(requirements, dict):
            try:
                epic = Epic(**requirements)
            except Exception as e:
                logger.warning(f"Failed to convert requirements to Epic: {e}")
                epic = requirements
        else:
            epic = requirements
        
        test_contexts = parser.extract_test_contexts(epic)
        
        test_cases = self.hrm_generator.generate_test_cases(
            test_contexts=test_contexts,
        )
        
        test_cases_dicts = [
            self._test_case_to_dict(tc)
            for tc in test_cases
        ]
        
        return {
            "test_cases": test_cases_dicts,
            "metadata": {
                "mode": "hrm_only",
                "hrm_generated": len(test_cases_dicts),
                "sqe_enhanced": False,
                "rag_context_used": False,
                "status": "success",
            },
        }
    
    def _generate_sqe_only(
        self,
        requirements: Dict[str, Any],
        context: Optional[str],
    ) -> Dict[str, Any]:
        """Generate using SQE agent only."""
        if not self.sqe_agent:
            logger.warning("SQE agent not available")
            return self._generate_hrm_only(requirements, context)
        
        logger.info("Generating with SQE agent only")
        
        result = self.sqe_agent.generate_from_epic(requirements)
        
        test_cases = result.get("hrm_test_cases", [])
        
        return {
            "test_cases": test_cases,
            "metadata": {
                "mode": "sqe_only",
                "sqe_generated": len(test_cases),
                "coverage_analysis": result.get("test_coverage_analysis", {}),
                "status": "success",
            },
        }
    
    def _generate_hybrid(
        self,
        requirements: Dict[str, Any],
        context: Optional[str],
    ) -> Dict[str, Any]:
        """
        Hybrid generation using both HRM and SQE.
        
        1. Retrieve similar tests from RAG
        2. Generate initial tests with HRM
        3. Enhance with SQE agent orchestration
        4. Merge and deduplicate
        """
        logger.info("Generating with hybrid approach (HRM + SQE + RAG)")
        
        from ..requirements_parser import RequirementParser
        from ..requirements_parser.schemas import Epic
        
        # Convert dict to Epic if needed
        if isinstance(requirements, dict):
            try:
                epic = Epic(**requirements)
            except Exception as e:
                logger.warning(f"Failed to convert requirements to Epic: {e}")
                epic = requirements
        else:
            epic = requirements
        
        similar_tests = []
        rag_context = ""
        
        if self.rag_retriever:
            try:
                similar_tests = self.rag_retriever.retrieve_similar_test_cases(
                    requirements,
                    top_k=5,
                    min_similarity=0.7
                )
                
                rag_context = self.rag_retriever.build_context(
                    requirements,
                    similar_tests
                )
                
                logger.info(f"Retrieved {len(similar_tests)} similar tests from RAG")
                
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        parser = RequirementParser()
        test_contexts = parser.extract_test_contexts(epic)
        
        hrm_test_cases = self.hrm_generator.generate_test_cases(
            test_contexts=test_contexts,
        )
        
        hrm_tests_dicts = [
            self._test_case_to_dict(tc)
            for tc in hrm_test_cases
        ]
        
        logger.info(f"HRM generated {len(hrm_tests_dicts)} test cases")
        
        sqe_tests_dicts = []
        coverage_analysis = {}
        
        if self.sqe_agent:
            try:
                sqe_result = self.sqe_agent.generate_from_epic(requirements)
                
                sqe_tests_dicts = sqe_result.get("hrm_test_cases", [])
                coverage_analysis = sqe_result.get("test_coverage_analysis", {})
                
                logger.info(f"SQE agent generated {len(sqe_tests_dicts)} test cases")
                
            except Exception as e:
                logger.warning(f"SQE generation failed: {e}")
        
        merged_tests = self._merge_test_cases(
            hrm_tests_dicts,
            sqe_tests_dicts
        )
        
        logger.info(f"Merged to {len(merged_tests)} unique test cases")
        
        return {
            "test_cases": merged_tests,
            "metadata": {
                "mode": "hybrid",
                "hrm_generated": len(hrm_tests_dicts),
                "sqe_generated": len(sqe_tests_dicts),
                "merged_count": len(merged_tests),
                "sqe_enhanced": bool(sqe_tests_dicts),
                "rag_context_used": bool(similar_tests),
                "rag_similar_count": len(similar_tests),
                "coverage_analysis": coverage_analysis,
                "merge_strategy": self.merge_strategy,
                "status": "success",
            },
        }
    
    def _merge_test_cases(
        self,
        hrm_tests: List[Dict[str, Any]],
        sqe_tests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge test cases from HRM and SQE.
        
        Args:
            hrm_tests: Tests from HRM generator
            sqe_tests: Tests from SQE agent
            
        Returns:
            Merged test cases
        """
        if self.merge_strategy == "union":
            return self._merge_union(hrm_tests, sqe_tests)
        elif self.merge_strategy == "intersection":
            return self._merge_intersection(hrm_tests, sqe_tests)
        else:  # weighted
            return self._merge_weighted(hrm_tests, sqe_tests)
    
    def _merge_union(
        self,
        hrm_tests: List[Dict[str, Any]],
        sqe_tests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge by union - all unique tests."""
        all_tests = hrm_tests + sqe_tests
        
        unique_tests = {}
        for test in all_tests:
            desc = test.get("description", "")
            if desc not in unique_tests:
                unique_tests[desc] = test
        
        return list(unique_tests.values())
    
    def _merge_intersection(
        self,
        hrm_tests: List[Dict[str, Any]],
        sqe_tests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge by intersection - only tests present in both."""
        hrm_descs = {test.get("description", ""): test for test in hrm_tests}
        sqe_descs = {test.get("description", ""): test for test in sqe_tests}
        
        common_descs = set(hrm_descs.keys()) & set(sqe_descs.keys())
        
        return [hrm_descs[desc] for desc in common_descs]
    
    def _merge_weighted(
        self,
        hrm_tests: List[Dict[str, Any]],
        sqe_tests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge with weighted preference."""
        hrm_count = int(len(hrm_tests) * self.hrm_weight)
        sqe_count = int(len(sqe_tests) * self.sqe_weight)
        
        selected_hrm = hrm_tests[:hrm_count]
        selected_sqe = sqe_tests[:sqe_count]
        
        return self._merge_union(selected_hrm, selected_sqe)
    
    def _test_case_to_dict(self, test_case: Any) -> Dict[str, Any]:
        """Convert TestCase object to dictionary."""
        return {
            "id": test_case.id,
            "description": test_case.description,
            "type": test_case.type.value if hasattr(test_case.type, 'value') else str(test_case.type),
            "priority": test_case.priority.value if hasattr(test_case.priority, 'value') else str(test_case.priority),
            "labels": test_case.labels if hasattr(test_case, 'labels') else [],
            "preconditions": test_case.preconditions if hasattr(test_case, 'preconditions') else [],
            "test_steps": [
                {"step": s.step_number, "action": s.action}
                for s in test_case.test_steps
            ] if hasattr(test_case, 'test_steps') else [],
            "expected_results": [
                r.result for r in test_case.expected_results
            ] if hasattr(test_case, 'expected_results') else [],
        }
