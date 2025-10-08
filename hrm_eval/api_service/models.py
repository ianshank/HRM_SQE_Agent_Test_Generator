"""
Pydantic models for API requests and responses.

Defines request/response schemas for the REST API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

from ..requirements_parser.schemas import (
    Epic,
    TestCase,
    GenerationMetadata,
    CoverageReport,
    TestType,
)


class GenerationOptions(BaseModel):
    """Options for test case generation."""
    
    test_types: List[TestType] = Field(
        default=[TestType.POSITIVE, TestType.NEGATIVE, TestType.EDGE],
        description="Types of test cases to generate"
    )
    min_coverage: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum coverage threshold"
    )
    priority_distribution: Optional[Dict[str, float]] = Field(
        default=None,
        description="Desired priority distribution (P1/P2/P3)"
    )
    include_performance_tests: bool = Field(
        default=False,
        description="Include performance test cases"
    )
    include_security_tests: bool = Field(
        default=False,
        description="Include security test cases"
    )
    max_test_cases_per_story: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum test cases per user story"
    )
    
    class Config:
        """Pydantic config."""
        frozen = False


class TestCaseResponse(BaseModel):
    """Response containing generated test cases."""
    
    test_cases: List[TestCase] = Field(..., description="Generated test cases")
    metadata: GenerationMetadata = Field(..., description="Generation metadata")
    coverage_report: CoverageReport = Field(..., description="Coverage analysis")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )
    
    class Config:
        """Pydantic config."""
        frozen = False


class BatchGenerationRequest(BaseModel):
    """Request for batch generation of test cases."""
    
    epics: List[Epic] = Field(..., min_items=1, description="List of epics to process")
    options: Optional[GenerationOptions] = Field(None, description="Generation options")
    
    class Config:
        """Pydantic config."""
        frozen = False


class BatchTestCaseResponse(BaseModel):
    """Response for batch generation."""
    
    results: List[TestCaseResponse] = Field(..., description="Results per epic")
    total_test_cases: int = Field(..., description="Total test cases generated")
    total_time_seconds: float = Field(..., description="Total generation time")
    success_count: int = Field(..., description="Number of successful generations")
    failure_count: int = Field(..., description="Number of failed generations")
    
    class Config:
        """Pydantic config."""
        frozen = False


class HealthStatus(str, Enum):
    """Health status values."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: HealthStatus = Field(..., description="Overall health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_checkpoint: Optional[str] = Field(None, description="Active checkpoint")
    device: str = Field(..., description="Computation device")
    uptime_seconds: float = Field(..., description="Service uptime")
    version: str = Field(..., description="API version")
    
    class Config:
        """Pydantic config."""
        frozen = False


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    
    class Config:
        """Pydantic config."""
        frozen = False

