"""
Software Quality Engineering (SQE) Agent using LangGraph.

Integrates with HRM model and RAG for enhanced test generation.
NO HARDCODED TEST GENERATION - uses workflow orchestration.
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
import logging

from .agent_state import SQEState
from .agent_tools import TestCaseGeneratorTool, CoverageAnalyzerTool
from .workflow_builder import SQEWorkflowBuilder

logger = logging.getLogger(__name__)


class SQEAgent:
    """
    Software Quality Engineering Agent using LangGraph.
    
    Orchestrates test case generation through:
    - Requirement analysis
    - RAG-based context retrieval
    - HRM model inference
    - Coverage analysis
    - Quality validation
    
    NO HARDCODING - all test generation through models/workflows.
    """
    
    def __init__(
        self,
        llm: Any,
        rag_retriever: Optional[Any] = None,
        hrm_generator: Optional[Any] = None,
        enable_rag: bool = True,
        enable_hrm: bool = True,
    ):
        """
        Initialize SQE Agent.
        
        Args:
            llm: LangChain LLM for agent reasoning
            rag_retriever: RAG retriever for historical context
            hrm_generator: HRM-based test generator
            enable_rag: Enable RAG retrieval
            enable_hrm: Enable HRM generation
        """
        self.llm = llm
        self.rag_retriever = rag_retriever if enable_rag else None
        self.hrm_generator = hrm_generator if enable_hrm else None
        self.enable_rag = enable_rag
        self.enable_hrm = enable_hrm
        
        workflow_builder = SQEWorkflowBuilder(
            llm=llm,
            rag_retriever=self.rag_retriever,
            hrm_generator=self.hrm_generator,
        )
        
        self.workflow = workflow_builder.build()
        
        self.test_generator_tool = TestCaseGeneratorTool(
            hrm_generator=self.hrm_generator,
            rag_retriever=self.rag_retriever,
        )
        
        self.coverage_analyzer_tool = CoverageAnalyzerTool()
        
        logger.info(
            f"SQEAgent initialized "
            f"(RAG: {self.enable_rag}, HRM: {self.enable_hrm})"
        )
    
    def generate_test_plan(
        self,
        story_breakdown: str,
        dev_plan: str,
        requirements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive test plan.
        
        Args:
            story_breakdown: User story breakdown text
            dev_plan: Development plan text
            requirements: Optional structured requirements (Epic dict)
            
        Returns:
            Dictionary with test plan, test cases, coverage analysis
        """
        logger.info("Generating test plan via SQE Agent workflow")
        
        initial_state = {
            "messages": [
                HumanMessage(
                    content=f"Story: {story_breakdown}\n\nDev Plan: {dev_plan}"
                )
            ],
            "story_breakdown": story_breakdown,
            "dev_plan": dev_plan,
            "requirements": requirements or {},
            "project_type": "",
            "test_plan": "",
            "automated_test_cases": "",
            "test_suites": [],
            "test_coverage_analysis": {},
            "rag_context": "",
            "hrm_test_cases": [],
            "current_step": "initialize",
            "analysis_complete": False,
            "test_generation_complete": False,
            "rag_retrieval_complete": False,
            "error": "",
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            
            logger.info(
                f"Workflow complete: {len(result.get('hrm_test_cases', []))} test cases"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "status": "failed",
                "test_cases": [],
            }
    
    def generate_from_epic(self, epic: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate test plan from Epic dictionary.
        
        Args:
            epic: Epic dictionary with user_stories
            
        Returns:
            Test plan with generated test cases
        """
        logger.info(f"Generating test plan for epic: {epic.get('epic_id', 'unknown')}")
        
        story_breakdown = "\n\n".join([
            f"Story {i+1}: {story.get('summary', '')}\n{story.get('description', '')}"
            for i, story in enumerate(epic.get('user_stories', []))
        ])
        
        dev_plan = "\n".join([
            f"Tech Stack: {', '.join(epic.get('tech_stack', []))}",
            f"Architecture: {epic.get('architecture', 'N/A')}",
        ])
        
        return self.generate_test_plan(
            story_breakdown=story_breakdown,
            dev_plan=dev_plan,
            requirements=epic,
        )
    
    def validate_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate requirements for testability.
        
        Args:
            requirements: Requirements dictionary (Epic)
            
        Returns:
            Validation report
        """
        try:
            from ..requirements_parser import RequirementValidator
            
            validator = RequirementValidator(strict_mode=True)
            
            is_valid, issues = validator.validate_epic(requirements)
            testability_score, report = validator.check_testability(requirements)
            
            logger.info(
                f"Requirements validation: valid={is_valid}, "
                f"testability={testability_score:.2%}"
            )
            
            return {
                "is_valid": is_valid,
                "issues": issues,
                "testability_score": testability_score,
                "report": report,
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "is_valid": False,
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current agent status and configuration.
        
        Returns:
            Status dictionary
        """
        return {
            "agent_type": "SQE",
            "llm_configured": self.llm is not None,
            "rag_enabled": self.enable_rag,
            "rag_available": self.rag_retriever is not None,
            "hrm_enabled": self.enable_hrm,
            "hrm_available": self.hrm_generator is not None,
            "workflow_compiled": self.workflow is not None,
            "tools": [
                "test_case_generator",
                "coverage_analyzer",
            ],
        }
