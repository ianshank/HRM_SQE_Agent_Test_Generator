"""
LangGraph workflow connector for test generation.

Integrates test case generation into LangGraph-based workflows
with support for iterative refinement and review loops.
"""

from typing import Dict, Any, Optional, Callable
import logging

from ..test_generator import TestCaseGenerator
from ..requirements_parser import RequirementParser, Epic
from ..requirements_parser.schemas import TestCase, UserFeedback

logger = logging.getLogger(__name__)


class WorkflowNode:
    """Workflow node for LangGraph integration."""
    
    def __init__(self, name: str, handler: Callable):
        """
        Initialize workflow node.
        
        Args:
            name: Node name
            handler: Node handler function
        """
        self.name = name
        self.handler = handler


class WorkflowConnector:
    """
    Connects test case generator to LangGraph workflows.
    
    Provides nodes for:
    - Test case generation
    - Coverage analysis
    - Iterative refinement based on feedback
    """
    
    def __init__(
        self,
        generator: TestCaseGenerator,
        enable_refinement: bool = True,
    ):
        """
        Initialize workflow connector.
        
        Args:
            generator: Test case generator
            enable_refinement: Enable iterative refinement
        """
        self.generator = generator
        self.parser = RequirementParser()
        self.enable_refinement = enable_refinement
        
        logger.info(
            f"WorkflowConnector initialized (refinement={enable_refinement})"
        )
    
    def create_test_generation_node(self) -> WorkflowNode:
        """
        Create workflow node for test generation.
        
        Returns:
            WorkflowNode for test generation
        """
        async def generate_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Handler for test generation node.
            
            Args:
                state: Workflow state
                
            Returns:
                Updated state with test cases
            """
            logger.info("Executing test generation node")
            
            epic_data = state.get("epic")
            if not epic_data:
                logger.error("No epic data in workflow state")
                return state
            
            epic = Epic(**epic_data)
            
            test_contexts = self.parser.extract_test_contexts(epic)
            
            test_cases = self.generator.generate_test_cases(test_contexts)
            
            state["test_cases"] = [tc.dict() for tc in test_cases]
            state["generation_complete"] = True
            
            logger.info(f"Generated {len(test_cases)} test cases")
            
            return state
        
        return WorkflowNode("test_generation", generate_handler)
    
    def create_refinement_node(self) -> WorkflowNode:
        """
        Create workflow node for iterative refinement.
        
        Returns:
            WorkflowNode for refinement
        """
        async def refinement_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Handler for refinement node.
            
            Args:
                state: Workflow state with feedback
                
            Returns:
                Updated state with refined test cases
            """
            logger.info("Executing refinement node")
            
            if not self.enable_refinement:
                logger.info("Refinement disabled, skipping")
                return state
            
            feedback_data = state.get("feedback", [])
            if not feedback_data:
                logger.info("No feedback provided, skipping refinement")
                return state
            
            test_cases = [TestCase(**tc) for tc in state.get("test_cases", [])]
            
            for feedback_item in feedback_data:
                test_case_id = feedback_item.get("test_case_id")
                
                test_case = next(
                    (tc for tc in test_cases if tc.id == test_case_id),
                    None
                )
                
                if test_case and feedback_item.get("rating", 5) < 3:
                    logger.info(f"Refining test case: {test_case_id}")
            
            state["test_cases"] = [tc.dict() for tc in test_cases]
            state["refinement_complete"] = True
            
            return state
        
        return WorkflowNode("test_refinement", refinement_handler)
    
    def create_coverage_analysis_node(self) -> WorkflowNode:
        """
        Create workflow node for coverage analysis.
        
        Returns:
            WorkflowNode for coverage analysis
        """
        async def coverage_handler(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Handler for coverage analysis node.
            
            Args:
                state: Workflow state
                
            Returns:
                Updated state with coverage analysis
            """
            logger.info("Executing coverage analysis node")
            
            epic_data = state.get("epic")
            if not epic_data:
                return state
            
            epic = Epic(**epic_data)
            
            coverage_analysis = self.parser.get_coverage_analysis(epic)
            
            state["coverage_analysis"] = coverage_analysis
            state["coverage_complete"] = True
            
            logger.info(
                f"Coverage analysis: {coverage_analysis['testability_score']:.2%} testability"
            )
            
            return state
        
        return WorkflowNode("coverage_analysis", coverage_handler)
    
    def create_workflow_graph(self) -> Dict[str, Any]:
        """
        Create complete workflow graph definition.
        
        Returns:
            Workflow graph definition for LangGraph
        """
        nodes = {
            "test_generation": self.create_test_generation_node(),
            "coverage_analysis": self.create_coverage_analysis_node(),
        }
        
        if self.enable_refinement:
            nodes["test_refinement"] = self.create_refinement_node()
        
        edges = [
            ("START", "coverage_analysis"),
            ("coverage_analysis", "test_generation"),
        ]
        
        if self.enable_refinement:
            edges.append(("test_generation", "test_refinement"))
            edges.append(("test_refinement", "END"))
        else:
            edges.append(("test_generation", "END"))
        
        workflow_graph = {
            "nodes": nodes,
            "edges": edges,
            "entry_point": "coverage_analysis",
        }
        
        logger.info(f"Created workflow graph with {len(nodes)} nodes")
        
        return workflow_graph

