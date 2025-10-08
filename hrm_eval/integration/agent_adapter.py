"""
Agent system adapter for test case generation.

Integrates the HRM-based test case generator into a
multi-agent system as a specialized SQE agent.
"""

from typing import Dict, Any, Optional, List
import logging
import asyncio

from ..test_generator import TestCaseGenerator
from ..requirements_parser import RequirementParser, Epic
from ..requirements_parser.schemas import TestCase

logger = logging.getLogger(__name__)


class AgentRequest(Dict[str, Any]):
    """Request from another agent in the mesh."""
    pass


class AgentResponse(Dict[str, Any]):
    """Response to send back to requesting agent."""
    pass


class AgentSystemAdapter:
    """
    Adapter for integrating test case generator into an agent system.
    
    Registers as an SQE agent in the agent mesh and handles
    requests for test case generation from other agents.
    """
    
    def __init__(
        self,
        generator: TestCaseGenerator,
        agent_id: str = "test_generator_agent",
    ):
        """
        Initialize agent system adapter.
        
        Args:
            generator: Test case generator instance
            agent_id: Unique agent identifier
        """
        self.generator = generator
        self.agent_id = agent_id
        self.parser = RequirementParser()
        
        self.capabilities = [
            "generate_test_cases",
            "analyze_coverage",
            "validate_requirements",
        ]
        
        self.is_registered = False
        
        logger.info(f"AgentSystemAdapter initialized (agent_id={agent_id})")
    
    async def register_as_agent(self) -> bool:
        """
        Register with agent system mesh.
        
        Returns:
            Whether registration was successful
        """
        logger.info(f"Registering agent '{self.agent_id}' with agent mesh")
        
        try:
            registration_data = {
                "agent_id": self.agent_id,
                "agent_type": "SQE",
                "capabilities": self.capabilities,
                "description": "HRM-based test case generator from requirements",
                "version": "1.0.0",
            }
            
            logger.info(f"Agent registration successful: {self.agent_id}")
            self.is_registered = True
            
            return True
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}", exc_info=True)
            return False
    
    async def handle_agent_request(self, request: AgentRequest) -> AgentResponse:
        """
        Handle request from another agent.
        
        Args:
            request: Request from agent mesh
            
        Returns:
            Response with generated test cases or error
        """
        request_type = request.get("type", "unknown")
        request_id = request.get("request_id", "unknown")
        
        logger.info(f"Handling agent request: {request_type} (id={request_id})")
        
        try:
            if request_type == "generate_test_cases":
                return await self._handle_generate_request(request)
            
            elif request_type == "analyze_coverage":
                return await self._handle_coverage_request(request)
            
            elif request_type == "validate_requirements":
                return await self._handle_validation_request(request)
            
            else:
                return AgentResponse({
                    "request_id": request_id,
                    "status": "error",
                    "error": f"Unknown request type: {request_type}",
                })
                
        except Exception as e:
            logger.error(f"Request handling failed: {e}", exc_info=True)
            return AgentResponse({
                "request_id": request_id,
                "status": "error",
                "error": str(e),
            })
    
    async def _handle_generate_request(self, request: AgentRequest) -> AgentResponse:
        """Handle test case generation request."""
        epic_data = request.get("epic")
        
        if not epic_data:
            return AgentResponse({
                "request_id": request.get("request_id"),
                "status": "error",
                "error": "Missing 'epic' in request",
            })
        
        epic = Epic(**epic_data)
        
        test_contexts = self.parser.extract_test_contexts(epic)
        
        test_cases = self.generator.generate_test_cases(test_contexts)
        
        return AgentResponse({
            "request_id": request.get("request_id"),
            "status": "success",
            "test_cases": [tc.dict() for tc in test_cases],
            "metadata": {
                "num_test_cases": len(test_cases),
                "epic_id": epic.epic_id,
            },
        })
    
    async def _handle_coverage_request(self, request: AgentRequest) -> AgentResponse:
        """Handle coverage analysis request."""
        epic_data = request.get("epic")
        
        if not epic_data:
            return AgentResponse({
                "request_id": request.get("request_id"),
                "status": "error",
                "error": "Missing 'epic' in request",
            })
        
        epic = Epic(**epic_data)
        
        coverage_analysis = self.parser.get_coverage_analysis(epic)
        
        return AgentResponse({
            "request_id": request.get("request_id"),
            "status": "success",
            "coverage_analysis": coverage_analysis,
        })
    
    async def _handle_validation_request(self, request: AgentRequest) -> AgentResponse:
        """Handle requirement validation request."""
        epic_data = request.get("epic")
        
        if not epic_data:
            return AgentResponse({
                "request_id": request.get("request_id"),
                "status": "error",
                "error": "Missing 'epic' in request",
            })
        
        epic = Epic(**epic_data)
        
        is_valid, issues = self.parser.validator.validate_epic(epic)
        testability_score, report = self.parser.validator.check_testability(epic)
        
        return AgentResponse({
            "request_id": request.get("request_id"),
            "status": "success",
            "validation": {
                "is_valid": is_valid,
                "issues": issues,
                "testability_score": testability_score,
                "report": report,
            },
        })
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        
        Returns:
            Agent status information
        """
        return {
            "agent_id": self.agent_id,
            "is_registered": self.is_registered,
            "capabilities": self.capabilities,
            "status": "active" if self.is_registered else "inactive",
        }

