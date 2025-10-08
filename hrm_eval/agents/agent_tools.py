"""
Custom tools for the SQE Agent.

Tools integrate with HRM generator and RAG retriever.
NO HARDCODED TEST GENERATION - all via actual models/workflows.
"""

from langchain_core.tools import BaseTool
from typing import Dict, Any, List, Optional
from pydantic import Field
import logging

logger = logging.getLogger(__name__)


class TestCaseGeneratorTool(BaseTool):
    """
    Tool for generating test cases using HRM model.
    
    NO HARDCODING - uses actual HRM model inference with RAG context.
    """
    
    name: str = "test_case_generator"
    description: str = (
        "Generate test cases using HRM model with RAG context. "
        "Provides comprehensive test coverage for requirements."
    )
    
    # Pydantic fields for tool dependencies
    hrm_generator: Optional[Any] = Field(default=None, exclude=True)
    rag_retriever: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, hrm_generator=None, rag_retriever=None, **kwargs):
        """
        Initialize test case generator tool.
        
        Args:
            hrm_generator: HRM-based test generator
            rag_retriever: RAG retriever for historical context
            **kwargs: Additional tool arguments
        """
        super().__init__(
            hrm_generator=hrm_generator,
            rag_retriever=rag_retriever,
            **kwargs
        )
        
        logger.info(
            f"TestCaseGeneratorTool initialized "
            f"(HRM: {hrm_generator is not None}, RAG: {rag_retriever is not None})"
        )
    
    def _run(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate test cases using HRM model (NO HARDCODING).
        
        Args:
            requirements: Structured requirements dictionary
            
        Returns:
            Dictionary with test_cases and context
        """
        logger.info("Generating test cases via HRM model")
        
        rag_context = ""
        similar_tests = []
        
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
                
                logger.info(f"Retrieved {len(similar_tests)} similar test cases from RAG")
                
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        test_cases = []
        
        if self.hrm_generator:
            try:
                from ..requirements_parser import RequirementParser
                
                parser = RequirementParser()
                
                test_contexts = parser.extract_test_contexts(requirements)
                
                test_cases = self.hrm_generator.generate_test_cases(
                    test_contexts=test_contexts,
                )
                
                test_cases_dicts = [
                    {
                        "id": tc.id,
                        "description": tc.description,
                        "type": tc.type.value if hasattr(tc.type, 'value') else str(tc.type),
                        "priority": tc.priority.value if hasattr(tc.priority, 'value') else str(tc.priority),
                        "test_steps": [
                            {"step": s.step_number, "action": s.action}
                            for s in tc.test_steps
                        ] if hasattr(tc, 'test_steps') else [],
                        "expected_results": [
                            r.result for r in tc.expected_results
                        ] if hasattr(tc, 'expected_results') else [],
                    }
                    for tc in test_cases
                ]
                
                logger.info(f"HRM model generated {len(test_cases)} test cases")
                
                return {
                    "test_cases": test_cases_dicts,
                    "context": rag_context,
                    "similar_tests_count": len(similar_tests),
                    "status": "success",
                }
                
            except Exception as e:
                logger.error(f"HRM generation failed: {e}", exc_info=True)
                return {
                    "test_cases": [],
                    "context": rag_context,
                    "status": "error",
                    "error": str(e),
                }
        else:
            return {
                "test_cases": [],
                "context": rag_context,
                "status": "pending",
                "message": "HRM generator not available - agent will generate tests",
            }


class CoverageAnalyzerTool(BaseTool):
    """Tool for analyzing test coverage."""
    
    name: str = "coverage_analyzer"
    description: str = (
        "Analyze test coverage for requirements. "
        "Identifies gaps and provides recommendations."
    )
    
    def __init__(self, **kwargs):
        """Initialize coverage analyzer tool."""
        super().__init__(**kwargs)
        logger.info("CoverageAnalyzerTool initialized")
    
    def _run(self, test_cases: List[Dict[str, Any]], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze coverage using actual CoverageAnalyzer.
        
        Args:
            test_cases: List of test case dictionaries
            requirements: Requirements dictionary
            
        Returns:
            Coverage analysis report
        """
        try:
            from ..test_generator import CoverageAnalyzer
            from ..requirements_parser import RequirementParser
            from ..requirements_parser.schemas import TestCase, TestStep, ExpectedResult, TestType, Priority
            
            analyzer = CoverageAnalyzer()
            parser = RequirementParser()
            
            test_contexts = parser.extract_test_contexts(requirements)
            
            test_case_objects = []
            for tc_dict in test_cases:
                try:
                    test_case = TestCase(
                        id=tc_dict.get('id', 'TC-000'),
                        type=TestType(tc_dict.get('type', 'positive')),
                        priority=Priority(tc_dict.get('priority', 'P2')),
                        description=tc_dict.get('description', ''),
                        preconditions=tc_dict.get('preconditions', []),
                        test_steps=[
                            TestStep(step_number=s.get('step', i+1), action=s.get('action', ''))
                            for i, s in enumerate(tc_dict.get('test_steps', []))
                        ],
                        expected_results=[
                            ExpectedResult(result=r if isinstance(r, str) else r.get('result', ''))
                            for r in tc_dict.get('expected_results', [])
                        ],
                        labels=tc_dict.get('labels', []),
                        source_story_id=tc_dict.get('source_story_id'),
                    )
                    test_case_objects.append(test_case)
                except Exception as e:
                    logger.warning(f"Failed to convert test case: {e}")
                    continue
            
            report = analyzer.analyze_coverage(test_case_objects, test_contexts)
            
            recommendations = analyzer.get_recommendations(report)
            
            logger.info(
                f"Coverage analysis complete: {report.get('coverage_percentage', 0):.1f}% coverage"
            )
            
            return {
                "coverage_percentage": report.get('coverage_percentage', 0),
                "positive_tests": report.get('positive_tests', 0),
                "negative_tests": report.get('negative_tests', 0),
                "edge_tests": report.get('edge_tests', 0),
                "stories_covered": report.get('stories_covered_count', 0),
                "total_stories": report.get('total_stories', 0),
                "missing_test_types": report.get('missing_test_types', []),
                "recommendations": recommendations,
                "gaps": report.get('gaps', []),
                "status": "success",
            }
            
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "coverage_percentage": 0,
            }


class RequirementValidatorTool(BaseTool):
    """Tool for validating requirements quality."""
    
    name: str = "requirement_validator"
    description: str = (
        "Validate requirements for completeness and testability. "
        "Provides quality score and improvement suggestions."
    )
    
    def __init__(self, **kwargs):
        """Initialize requirement validator tool."""
        super().__init__(**kwargs)
        logger.info("RequirementValidatorTool initialized")
    
    def _run(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate requirements using RequirementValidator.
        
        Args:
            requirements: Requirements dictionary (Epic)
            
        Returns:
            Validation report
        """
        try:
            from ..requirements_parser import RequirementValidator
            
            validator = RequirementValidator(strict_mode=True)
            
            is_valid, issues = validator.validate_epic(requirements)
            
            testability_score, testability_report = validator.check_testability(requirements)
            
            logger.info(
                f"Requirement validation: valid={is_valid}, "
                f"testability={testability_score:.2%}"
            )
            
            return {
                "is_valid": is_valid,
                "issues": issues,
                "testability_score": testability_score,
                "testability_report": testability_report,
                "status": "success",
            }
            
        except Exception as e:
            logger.error(f"Requirement validation failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "is_valid": False,
            }


class TestCaseIndexerTool(BaseTool):
    """Tool for indexing generated test cases into RAG vector store."""
    
    name: str = "test_case_indexer"
    description: str = (
        "Index generated test cases into vector store for future RAG retrieval. "
        "Builds historical knowledge base."
    )
    
    # Pydantic field for vector indexer dependency
    vector_indexer: Optional[Any] = Field(default=None, exclude=True)
    
    def __init__(self, vector_indexer=None, **kwargs):
        """
        Initialize test case indexer tool.
        
        Args:
            vector_indexer: VectorIndexer instance
            **kwargs: Additional tool arguments
        """
        super().__init__(vector_indexer=vector_indexer, **kwargs)
        logger.info(f"TestCaseIndexerTool initialized (indexer: {vector_indexer is not None})")
    
    def _run(self, test_cases: List[Dict[str, Any]], source: str = "sqe_agent") -> Dict[str, Any]:
        """
        Index test cases into vector store.
        
        Args:
            test_cases: List of test case dictionaries
            source: Source identifier
            
        Returns:
            Indexing result
        """
        if not self.vector_indexer:
            logger.warning("Vector indexer not available")
            return {
                "status": "skipped",
                "message": "Vector indexer not configured",
            }
        
        try:
            self.vector_indexer.index_test_cases(
                test_cases,
                batch_size=100,
                show_progress=False,
            )
            
            stats = self.vector_indexer.get_indexing_stats()
            
            logger.info(f"Indexed {len(test_cases)} test cases from {source}")
            
            return {
                "status": "success",
                "indexed_count": len(test_cases),
                "source": source,
                "stats": stats,
            }
            
        except Exception as e:
            logger.error(f"Test case indexing failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }
