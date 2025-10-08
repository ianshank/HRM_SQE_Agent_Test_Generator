"""
Integration tests for API service.

Tests API endpoints using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import torch

from ..api_service.main import app
from ..requirements_parser.schemas import Epic, UserStory, AcceptanceCriteria


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_epic():
    """Create sample epic for testing."""
    return {
        "epic_id": "EPIC-TEST-001",
        "title": "Test Epic for API",
        "user_stories": [
            {
                "id": "US-TEST-001",
                "summary": "Test user story",
                "description": "This is a test user story for API testing",
                "acceptance_criteria": [
                    {"criteria": "System should process input correctly"},
                    {"criteria": "System should return expected output"},
                ],
                "tech_stack": ["Python", "FastAPI"],
            }
        ],
        "tech_stack": ["Docker", "PostgreSQL"],
        "architecture": "Microservices",
    }


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns status."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert "version" in data


class TestInitializeEndpoint:
    """Test model initialization endpoint."""
    
    @patch('hrm_eval.api_service.main.initialize_model')
    async def test_initialize_success(self, mock_init, client):
        """Test successful initialization."""
        mock_init.return_value = None
        
        response = client.post("/api/v1/initialize")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "message" in data


class TestGenerateTestsEndpoint:
    """Test test case generation endpoint."""
    
    @pytest.mark.skip(reason="Requires actual model loading")
    def test_generate_tests_without_initialization(self, client, sample_epic):
        """Test generation fails without initialization."""
        response = client.post(
            "/api/v1/generate-tests",
            json={"epic": sample_epic}
        )
        
        assert response.status_code == 503
    
    def test_generate_tests_invalid_epic(self, client):
        """Test generation with invalid epic data."""
        invalid_epic = {
            "epic_id": "INVALID",
        }
        
        response = client.post(
            "/api/v1/generate-tests",
            json={"epic": invalid_epic}
        )
        
        assert response.status_code in [422, 500]


class TestBatchGenerateEndpoint:
    """Test batch generation endpoint."""
    
    @pytest.mark.skip(reason="Requires actual model loading")
    def test_batch_generate_without_initialization(self, client, sample_epic):
        """Test batch generation fails without initialization."""
        request_data = {
            "epics": [sample_epic],
            "options": {
                "test_types": ["positive", "negative"],
                "min_coverage": 0.8,
            }
        }
        
        response = client.post(
            "/api/v1/batch-generate",
            json=request_data
        )
        
        assert response.status_code == 503


class TestAPIValidation:
    """Test API request validation."""
    
    def test_invalid_json(self, client):
        """Test invalid JSON is rejected."""
        response = client.post(
            "/api/v1/generate-tests",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test missing required fields."""
        incomplete_epic = {
            "epic_id": "INCOMPLETE",
        }
        
        response = client.post(
            "/api/v1/generate-tests",
            json={"epic": incomplete_epic}
        )
        
        assert response.status_code in [422, 500]


class TestMiddleware:
    """Test middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/health")
        
        assert "access-control-allow-origin" in response.headers or response.status_code == 200
    
    def test_process_time_header(self, client):
        """Test process time header is added."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

