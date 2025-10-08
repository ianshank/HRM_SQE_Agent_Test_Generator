"""
State management for SQE Agent workflows.

Defines TypedDict state and enums for LangGraph workflows.
"""

from typing import TypedDict, Annotated, Any, Dict, List
from enum import Enum
import operator


class TestType(Enum):
    """Test case type classification."""
    
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
    """Test case priority levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AutomationLevel(Enum):
    """Test automation level."""
    
    MANUAL = "manual"
    SEMI_AUTOMATED = "semi_automated"
    FULLY_AUTOMATED = "fully_automated"


class SQEState(TypedDict):
    """
    State for the SQE Agent workflow.
    
    Used by LangGraph to track state across workflow nodes.
    """
    
    messages: Annotated[List[Any], operator.add]
    
    story_breakdown: str
    
    dev_plan: str
    
    project_type: str
    
    test_plan: str
    
    automated_test_cases: str
    
    test_suites: List[Dict[str, Any]]
    
    test_coverage_analysis: Dict[str, Any]
    
    rag_context: str
    
    hrm_test_cases: List[Dict[str, Any]]
    
    requirements: Dict[str, Any]
    
    current_step: str
    
    analysis_complete: bool
    
    test_generation_complete: bool
    
    rag_retrieval_complete: bool
    
    error: str
