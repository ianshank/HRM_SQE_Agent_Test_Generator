"""
LangGraph workflow builder for SQE Agent.

Builds multi-node workflow integrating RAG, HRM, and LLM reasoning.
"""

from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, START, END
import logging

from .agent_state import SQEState

logger = logging.getLogger(__name__)


class SQEWorkflowBuilder:
    """
    Build LangGraph workflow for SQE agent.
    
    Creates workflow with nodes for:
    - Requirement analysis
    - RAG context retrieval
    - HRM test generation
    - Coverage analysis
    - Finalization
    """
    
    def __init__(
        self,
        llm: Any,
        rag_retriever: Optional[Any] = None,
        hrm_generator: Optional[Any] = None,
    ):
        """
        Initialize workflow builder.
        
        Args:
            llm: LangChain LLM for agent reasoning
            rag_retriever: Optional RAG retriever
            hrm_generator: Optional HRM generator
        """
        self.llm = llm
        self.rag_retriever = rag_retriever
        self.hrm_generator = hrm_generator
        
        logger.info(
            f"SQEWorkflowBuilder initialized "
            f"(RAG: {rag_retriever is not None}, HRM: {hrm_generator is not None})"
        )
    
    def build(self) -> Any:
        """
        Build the complete workflow graph.
        
        Returns:
            Compiled LangGraph workflow
        """
        workflow = StateGraph(SQEState)
        
        workflow.add_node("analyze_requirements", self._analyze_requirements)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_tests", self._generate_tests)
        workflow.add_node("analyze_coverage", self._analyze_coverage)
        workflow.add_node("finalize", self._finalize)
        
        workflow.add_edge(START, "analyze_requirements")
        workflow.add_edge("analyze_requirements", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_tests")
        workflow.add_edge("generate_tests", "analyze_coverage")
        workflow.add_edge("analyze_coverage", "finalize")
        workflow.add_edge("finalize", END)
        
        logger.info("Workflow graph built with 5 nodes")
        
        return workflow.compile()
    
    def _analyze_requirements(self, state: SQEState) -> SQEState:
        """
        Analyze requirements node.
        
        Extracts project type and validates requirements.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Node: analyze_requirements")
        
        state["current_step"] = "analyze_requirements"
        
        try:
            combined_text = (
                state.get("story_breakdown", "") + " " + 
                state.get("dev_plan", "")
            ).lower()
            
            if "fastapi" in combined_text or "python" in combined_text:
                project_type = "FastAPI Python Microservice"
            elif "react" in combined_text or "typescript" in combined_text:
                project_type = "React TypeScript Frontend"
            else:
                project_type = "Generic Application"
            
            state["project_type"] = project_type
            state["analysis_complete"] = True
            
            logger.info(f"Project type identified: {project_type}")
            
        except Exception as e:
            logger.error(f"Requirement analysis failed: {e}", exc_info=True)
            state["error"] = str(e)
        
        return state
    
    def _retrieve_context(self, state: SQEState) -> SQEState:
        """
        Retrieve RAG context node.
        
        Retrieves similar test cases from vector store.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with RAG context
        """
        logger.info("Node: retrieve_context")
        
        state["current_step"] = "retrieve_context"
        
        if not self.rag_retriever:
            logger.info("RAG retriever not available, skipping")
            state["rag_context"] = ""
            state["rag_retrieval_complete"] = True
            return state
        
        try:
            requirements = state.get("requirements", {})
            
            if not requirements:
                requirement_dict = {
                    "summary": state.get("story_breakdown", ""),
                    "description": state.get("dev_plan", ""),
                }
            else:
                requirement_dict = requirements
            
            similar_tests = self.rag_retriever.retrieve_similar_test_cases(
                requirement_dict,
                top_k=5,
                min_similarity=0.7
            )
            
            rag_context = self.rag_retriever.build_context(
                requirement_dict,
                similar_tests
            )
            
            state["rag_context"] = rag_context
            state["rag_retrieval_complete"] = True
            
            logger.info(f"Retrieved {len(similar_tests)} similar test cases")
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}", exc_info=True)
            state["rag_context"] = ""
            state["error"] = f"RAG retrieval: {str(e)}"
        
        return state
    
    def _generate_tests(self, state: SQEState) -> SQEState:
        """
        Generate tests using HRM model node.
        
        Uses HRM generator with RAG context for test generation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with generated tests
        """
        logger.info("Node: generate_tests")
        
        state["current_step"] = "generate_tests"
        
        if not self.hrm_generator:
            logger.warning("HRM generator not available")
            state["hrm_test_cases"] = []
            state["test_generation_complete"] = True
            return state
        
        try:
            from ..requirements_parser import RequirementParser, Epic, UserStory, AcceptanceCriteria
            
            requirements = state.get("requirements")
            
            if requirements and isinstance(requirements, dict):
                epic = Epic(**requirements)
            else:
                epic = Epic(
                    epic_id="workflow-generated",
                    title=state.get("project_type", "Project"),
                    user_stories=[
                        UserStory(
                            id="US-001",
                            summary=state.get("story_breakdown", "")[:100],
                            description=state.get("story_breakdown", ""),
                            acceptance_criteria=[
                                AcceptanceCriteria(
                                    criteria=line.strip()
                                )
                                for line in state.get("dev_plan", "").split("\n")
                                if line.strip() and len(line.strip()) > 10
                            ][:5],
                        )
                    ],
                )
            
            parser = RequirementParser()
            test_contexts = parser.extract_test_contexts(epic)
            
            logger.info(f"Extracted {len(test_contexts)} test contexts")
            
            test_cases = self.hrm_generator.generate_test_cases(
                test_contexts=test_contexts,
            )
            
            test_cases_dicts = [
                {
                    "id": tc.id,
                    "description": tc.description,
                    "type": tc.type.value if hasattr(tc.type, 'value') else str(tc.type),
                    "priority": tc.priority.value if hasattr(tc.priority, 'value') else str(tc.priority),
                    "labels": tc.labels,
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
            
            state["hrm_test_cases"] = test_cases_dicts
            state["test_generation_complete"] = True
            
            logger.info(f"Generated {len(test_cases)} test cases via HRM")
            
        except Exception as e:
            logger.error(f"Test generation failed: {e}", exc_info=True)
            state["hrm_test_cases"] = []
            state["error"] = f"Test generation: {str(e)}"
        
        return state
    
    def _analyze_coverage(self, state: SQEState) -> SQEState:
        """
        Analyze coverage node.
        
        Analyzes test coverage and provides recommendations.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with coverage analysis
        """
        logger.info("Node: analyze_coverage")
        
        state["current_step"] = "analyze_coverage"
        
        try:
            test_cases = state.get("hrm_test_cases", [])
            
            if not test_cases:
                logger.warning("No test cases to analyze")
                state["test_coverage_analysis"] = {
                    "coverage_percentage": 0,
                    "status": "no_tests",
                }
                return state
            
            from ..test_generator import CoverageAnalyzer
            from ..requirements_parser import RequirementParser
            
            parser = RequirementParser()
            
            requirements = state.get("requirements")
            if requirements and isinstance(requirements, dict):
                epic = requirements
            else:
                return state
            
            test_contexts = parser.extract_test_contexts(epic)
            
            from ..requirements_parser.schemas import TestCase, TestStep, ExpectedResult, TestType, Priority
            
            test_case_objects = []
            for tc_dict in test_cases:
                try:
                    test_case = TestCase(
                        id=tc_dict.get('id', 'TC-000'),
                        type=TestType(tc_dict.get('type', 'positive')),
                        priority=Priority(tc_dict.get('priority', 'P2')),
                        description=tc_dict.get('description', ''),
                        preconditions=[],
                        test_steps=[
                            TestStep(step_number=s.get('step', i+1), action=s.get('action', ''))
                            for i, s in enumerate(tc_dict.get('test_steps', []))
                        ],
                        expected_results=[
                            ExpectedResult(result=r if isinstance(r, str) else r.get('result', ''))
                            for r in tc_dict.get('expected_results', [])
                        ],
                        labels=tc_dict.get('labels', []),
                    )
                    test_case_objects.append(test_case)
                except Exception as e:
                    logger.warning(f"Failed to convert test case: {e}")
                    continue
            
            analyzer = CoverageAnalyzer()
            report = analyzer.analyze_coverage(test_case_objects, test_contexts)
            
            state["test_coverage_analysis"] = report
            
            logger.info(f"Coverage analysis: {report.get('coverage_percentage', 0):.1f}%")
            
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}", exc_info=True)
            state["test_coverage_analysis"] = {"error": str(e)}
        
        return state
    
    def _finalize(self, state: SQEState) -> SQEState:
        """
        Finalize workflow node.
        
        Prepares final output and summary.
        
        Args:
            state: Current workflow state
            
        Returns:
            Final state
        """
        logger.info("Node: finalize")
        
        state["current_step"] = "complete"
        
        test_cases = state.get("hrm_test_cases", [])
        coverage = state.get("test_coverage_analysis", {})
        
        summary = {
            "total_test_cases": len(test_cases),
            "coverage_percentage": coverage.get("coverage_percentage", 0),
            "project_type": state.get("project_type", "Unknown"),
            "rag_used": bool(state.get("rag_context")),
            "status": "success" if not state.get("error") else "partial",
        }
        
        state["test_plan"] = f"""
# Test Plan Summary

**Project Type:** {summary['project_type']}
**Total Test Cases Generated:** {summary['total_test_cases']}
**Coverage:** {summary['coverage_percentage']:.1f}%
**RAG Context Used:** {summary['rag_used']}

## Test Cases

{len(test_cases)} test cases generated via HRM model with RAG context.

## Coverage Analysis

{coverage.get('positive_tests', 0)} positive tests
{coverage.get('negative_tests', 0)} negative tests
{coverage.get('edge_tests', 0)} edge tests
"""
        
        logger.info(f"Workflow complete: {summary['total_test_cases']} tests, {summary['coverage_percentage']:.1f}% coverage")
        
        return state
