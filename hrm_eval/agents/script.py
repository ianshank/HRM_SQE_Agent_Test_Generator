# Create a comprehensive LangGraph version of the CrewAI SQE Agent
# This will be a complete working example with proper state management and tool integration

langgraph_sqs_agent_code = '''
"""
LangGraph Software Quality Engineering (SQE) Agent

Converted from CrewAI implementation to LangGraph architecture.
This agent creates comprehensive test plans, test cases, and automated validations.
"""

import json
import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Enums from original code
class TestType(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    API = "api"
    UI = "ui"
    PERFORMANCE = "performance"
    SECURITY = "security"
    E2E = "end_to_end"
    REGRESSION = "regression"
    SMOKE = "smoke"

class TestPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AutomationLevel(Enum):
    MANUAL = "manual"
    SEMI_AUTOMATED = "semi_automated"
    FULLY_AUTOMATED = "fully_automated"

# LangGraph State Definition
class SQEState(TypedDict):
    """State for the SQE Agent workflow"""
    messages: Annotated[list[BaseMessage], add_messages]
    story_breakdown: str
    dev_plan: str
    project_type: str
    test_plan: str
    automated_test_cases: str
    test_suites: list[dict[str, Any]]
    test_coverage_analysis: dict[str, Any]
    current_step: str
    analysis_complete: bool
    test_generation_complete: bool

# Tool for Test Case Generation
class TestCaseGeneratorTool(BaseTool):
    name: str = "test_case_generator"
    description: str = "Generates comprehensive test cases, plans, and automated validations"
    
    def _run(self, story_breakdown: str, dev_plan: str) -> dict[str, Any]:
        """Generate test specification based on inputs"""
        # Analyze project type
        combined_text = (story_breakdown + " " + dev_plan).lower()
        
        if "fastapi" in combined_text or "python" in combined_text:
            project_type = "FastAPI Python Microservice"
            frameworks = {
                "unit": "pytest",
                "api": "pytest + httpx",
                "ui": "N/A (API-only)"
            }
        elif "react" in combined_text or "typescript" in combined_text:
            project_type = "React TypeScript Frontend"
            frameworks = {
                "unit": "Jest + React Testing Library",
                "api": "MSW (Mock Service Worker)",
                "ui": "Cypress / Playwright"
            }
        else:
            project_type = "Node.js Express Application"
            frameworks = {
                "unit": "Jest + Supertest",
                "api": "Supertest",
                "ui": "Cypress"
            }
        
        # Generate comprehensive test plan
        test_plan = f"""
# TEST PLAN: {project_type} Quality Assurance

## Test Objectives
- Validate all user stories and acceptance criteria
- Ensure API functionality and data integrity
- Verify user interface behavior and usability
- Confirm security and performance requirements
- Validate integration points and error handling

## Test Strategy
1. **Unit Testing** - Framework: {frameworks['unit']}
   - Coverage Target: 85%+
   - Business logic validation
   - Error handling scenarios

2. **API Testing** - Framework: {frameworks['api']}
   - Endpoint functionality validation
   - Authentication and authorization testing
   - Error response validation

3. **Integration Testing**
   - Database integration testing
   - External API integration validation
   - Service-to-service communication testing

4. **Performance Testing**
   - Load testing for concurrent users
   - API response time validation
   - Database query performance testing

## Test Environments
- Development: Local development environment
- Testing: Dedicated QA environment with test data
- Staging: Production-like environment for final validation

## Entry/Exit Criteria
- Entry: Development code complete, unit tests passing 80%
- Exit: All critical/high tests pass, 85% coverage, no high-severity defects
        """
        
        # Generate automated test cases
        automated_test_cases = f"""
# AUTOMATED TEST CASES: {project_type}

## Unit Tests Example
```python
def test_user_creation_success():
    user_data = {{"email": "test@example.com", "name": "Test User"}}
    result = create_user(user_data)
    assert result.email == "test@example.com"
    assert result.active is True

def test_user_creation_duplicate_email():
    user_data = {{"email": "duplicate@example.com", "name": "Test User"}}
    with pytest.raises(ValueError, match="Email already exists"):
        create_user(user_data)
```

## API Integration Tests
```python
def test_api_create_user():
    response = client.post("/users", json={{"email": "test@example.com", "name": "Test User"}})
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"

def test_api_get_user_not_found():
    response = client.get("/users/99999")
    assert response.status_code == 404
```

## CI/CD Integration
```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run Unit Tests
        run: pytest tests/unit/ --cov=src --cov-report=html
      - name: Run Integration Tests  
        run: pytest tests/integration/
```
        """
        
        # Generate test suites
        test_suites = [
            {
                "suite_id": "unit_tests",
                "name": "Unit Test Suite",
                "description": "Comprehensive unit testing for business logic",
                "test_types": [TestType.UNIT.value],
                "priority": TestPriority.HIGH.value,
                "automation_level": AutomationLevel.FULLY_AUTOMATED.value
            },
            {
                "suite_id": "api_tests",
                "name": "API Test Suite", 
                "description": "API endpoint validation and integration testing",
                "test_types": [TestType.API.value, TestType.INTEGRATION.value],
                "priority": TestPriority.CRITICAL.value,
                "automation_level": AutomationLevel.FULLY_AUTOMATED.value
            },
            {
                "suite_id": "e2e_tests",
                "name": "End-to-End Test Suite",
                "description": "Complete user workflow validation",
                "test_types": [TestType.E2E.value, TestType.FUNCTIONAL.value],
                "priority": TestPriority.MEDIUM.value,
                "automation_level": AutomationLevel.SEMI_AUTOMATED.value
            }
        ]
        
        # Generate coverage analysis
        coverage_analysis = {
            "target_coverage": 85,
            "critical_paths": [
                "User authentication and authorization",
                "Data validation and sanitization", 
                "API error handling",
                "Database transactions"
            ],
            "risk_areas": [
                "External API integrations",
                "Payment processing workflows",
                "File upload and processing",
                "Security and access controls"
            ],
            "automation_ratio": {
                "unit_tests": 100,
                "api_tests": 95,
                "integration_tests": 80,
                "e2e_tests": 60
            }
        }
        
        return {
            "project_type": project_type,
            "test_plan": test_plan.strip(),
            "automated_test_cases": automated_test_cases.strip(),
            "test_suites": test_suites,
            "test_coverage_analysis": coverage_analysis
        }

# Node Functions
def analyze_requirements(state: SQEState) -> dict[str, Any]:
    """Analyze the story breakdown and development plan"""
    messages = state["messages"]
    story_breakdown = state.get("story_breakdown", "")
    dev_plan = state.get("dev_plan", "")
    
    if not story_breakdown or not dev_plan:
        # Extract from messages if not provided directly
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                if "story" in content:
                    story_breakdown = msg.content
                elif "development" in content or "dev plan" in content:
                    dev_plan = msg.content
    
    # Determine project type from inputs
    combined_text = (story_breakdown + " " + dev_plan).lower()
    
    if "fastapi" in combined_text or "python" in combined_text:
        project_type = "FastAPI Python Microservice"
    elif "react" in combined_text or "typescript" in combined_text:
        project_type = "React TypeScript Frontend"
    else:
        project_type = "Node.js Express Application"
    
    analysis_msg = AIMessage(
        content=f"Analysis complete. Project type identified as: {project_type}. "
                f"Story breakdown and development plan have been processed."
    )
    
    return {
        "messages": [analysis_msg],
        "project_type": project_type,
        "current_step": "analysis_complete",
        "analysis_complete": True
    }

def generate_test_plan(state: SQEState) -> dict[str, Any]:
    """Generate comprehensive test plan using the tool"""
    story_breakdown = state.get("story_breakdown", "")
    dev_plan = state.get("dev_plan", "")
    
    # Use the test case generator tool
    tool = TestCaseGeneratorTool()
    result = tool._run(story_breakdown, dev_plan)
    
    test_plan_msg = AIMessage(
        content=f"Test plan generated for {result['project_type']}. "
                f"Includes {len(result['test_suites'])} test suites with comprehensive coverage analysis."
    )
    
    return {
        "messages": [test_plan_msg],
        "test_plan": result["test_plan"],
        "automated_test_cases": result["automated_test_cases"],
        "test_suites": result["test_suites"],
        "test_coverage_analysis": result["test_coverage_analysis"],
        "current_step": "test_generation_complete",
        "test_generation_complete": True
    }

def finalize_output(state: SQEState) -> dict[str, Any]:
    """Finalize and format the complete SQE output"""
    
    summary_msg = AIMessage(
        content=f"""
# Software Quality Engineering Analysis Complete

## Project Type: {state['project_type']}

## Generated Deliverables:
[DONE] Comprehensive Test Plan
[DONE] Automated Test Cases ({len(state['test_suites'])} test suites)
[DONE] Test Coverage Analysis (Target: {state['test_coverage_analysis']['target_coverage']}%)
[DONE] CI/CD Integration Guidelines

## Test Suite Summary:
{chr(10).join([f"- {suite['name']}: {suite['description']}" for suite in state['test_suites']])}

## Next Steps:
1. Review and approve test plan
2. Set up test environments
3. Implement automated test cases
4. Configure CI/CD pipeline
5. Execute test validation

The complete SQE package is ready for implementation.
        """
    )
    
    return {
        "messages": [summary_msg],
        "current_step": "complete"
    }

# Routing Functions
def route_after_analysis(state: SQEState) -> Literal["generate_test_plan", "__end__"]:
    """Route after requirements analysis"""
    if state.get("analysis_complete", False):
        return "generate_test_plan"
    return "__end__"

def route_after_test_generation(state: SQEState) -> Literal["finalize_output", "__end__"]:
    """Route after test plan generation"""
    if state.get("test_generation_complete", False):
        return "finalize_output"
    return "__end__"

# Build the LangGraph
def create_sqs_agent() -> StateGraph:
    """Create the SQE Agent LangGraph"""
    
    # Initialize the graph
    graph_builder = StateGraph(SQEState)
    
    # Add nodes
    graph_builder.add_node("analyze_requirements", analyze_requirements)
    graph_builder.add_node("generate_test_plan", generate_test_plan)
    graph_builder.add_node("finalize_output", finalize_output)
    
    # Add edges
    graph_builder.add_edge(START, "analyze_requirements")
    
    # Add conditional edges
    graph_builder.add_conditional_edges(
        "analyze_requirements",
        route_after_analysis,
        {
            "generate_test_plan": "generate_test_plan",
            "__end__": END
        }
    )
    
    graph_builder.add_conditional_edges(
        "generate_test_plan", 
        route_after_test_generation,
        {
            "finalize_output": "finalize_output",
            "__end__": END
        }
    )
    
    graph_builder.add_edge("finalize_output", END)
    
    return graph_builder.compile()

# Usage Example
def run_sqs_agent_example():
    """Example of how to use the SQE Agent"""
    
    # Create the agent
    sqs_agent = create_sqs_agent()
    
    # Example input
    initial_state = {
        "messages": [
            HumanMessage(content="Create test plan for user authentication system")
        ],
        "story_breakdown": """
        User Story: As a user, I want to securely log in to the application
        Acceptance Criteria:
        - User can enter email and password
        - System validates credentials
        - Successful login redirects to dashboard
        - Failed login shows error message
        - Account lockout after 3 failed attempts
        """,
        "dev_plan": """
        Development Plan:
        - FastAPI backend with JWT authentication
        - SQLAlchemy for user data persistence
        - Bcrypt for password hashing
        - Rate limiting for login attempts
        - Redis for session management
        """,
        "current_step": "start",
        "analysis_complete": False,
        "test_generation_complete": False
    }
    
    # Run the agent
    result = sqs_agent.invoke(initial_state)
    
    return result

# Print the complete code structure
print("LangGraph SQE Agent Implementation:")
print("=" * 50)
print("[DONE] State Management: SQEState with TypedDict")
print("[DONE] Tool Integration: TestCaseGeneratorTool")
print("[DONE] Node Functions: analyze_requirements, generate_test_plan, finalize_output")
print("[DONE] Conditional Routing: route_after_analysis, route_after_test_generation")
print("[DONE] Graph Structure: START → analyze → generate → finalize → END")
'''

# Save the code to a file for easy reference
with open('langgraph_sqs_agent.py', 'w') as f:
    f.write(langgraph_sqs_agent_code)

print("LangGraph SQE Agent code has been generated and saved to 'langgraph_sqs_agent.py'")
print("\nKey Components:")
print("1. SQEState - TypedDict for state management")
print("2. TestCaseGeneratorTool - Converted from CrewAI tool")
print("3. Node functions for each processing step")
print("4. Conditional routing based on completion status")
print("5. Comprehensive test plan generation")