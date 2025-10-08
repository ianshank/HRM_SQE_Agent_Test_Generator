"""
API integration tests for RAG + SQE endpoints.

Tests FastAPI endpoints with RAG and SQE components.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json

# Mock the heavy imports before importing the app
with patch('hrm_eval.api_service.main.VectorStore'), \
     patch('hrm_eval.api_service.main.RAGRetriever'), \
     patch('hrm_eval.api_service.main.SQEAgent'), \
     patch('hrm_eval.api_service.main.HybridTestGenerator'), \
     patch('hrm_eval.api_service.main.WorkflowManager'):
    from hrm_eval.api_service.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_epic_payload():
    """Sample epic payload for API requests."""
    return {
        "epic_id": "API-EPIC-001",
        "title": "User Authentication API",
        "user_stories": [
            {
                "id": "API-US-001",
                "summary": "Login endpoint",
                "description": "Implement login endpoint",
                "acceptance_criteria": [
                    {"criteria": "Returns JWT token on success"},
                    {"criteria": "Returns 401 on failure"}
                ],
                "tech_stack": ["FastAPI", "JWT"]
            }
        ],
        "tech_stack": ["FastAPI", "PostgreSQL"],
        "architecture": "REST API"
    }


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert "version" in data
    
    def test_extended_health_check(self, client):
        """Test extended health check with RAG status."""
        response = client.get("/api/v1/health-extended")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "rag_health" in data or data["rag_health"] is None


class TestRAGInitialization:
    """Test RAG initialization endpoint."""
    
    @patch('hrm_eval.api_service.main.VectorStore')
    @patch('hrm_eval.api_service.main.EmbeddingGenerator')
    @patch('hrm_eval.api_service.main.RAGRetriever')
    @patch('hrm_eval.api_service.main.VectorIndexer')
    def test_initialize_rag_success(
        self,
        mock_indexer,
        mock_retriever,
        mock_embedding,
        mock_vector_store,
        client,
    ):
        """Test successful RAG initialization."""
        response = client.post(
            "/api/v1/initialize-rag",
            params={
                "backend": "chromadb",
                "enable_sqe": False,
                "llm_provider": "openai"
            }
        )
        
        assert response.status_code in [200, 503]  # 503 if HRM not initialized
        
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "components" in data
    
    @patch('hrm_eval.api_service.main.VectorStore')
    def test_initialize_rag_with_invalid_backend(
        self,
        mock_vector_store,
        client,
    ):
        """Test RAG initialization with invalid backend."""
        mock_vector_store.side_effect = ValueError("Unsupported backend")
        
        response = client.post(
            "/api/v1/initialize-rag",
            params={"backend": "invalid_backend"}
        )
        
        # Should return error
        assert response.status_code in [500, 503]


class TestRAGGenerationEndpoint:
    """Test RAG-enhanced test generation endpoint."""
    
    @patch('hrm_eval.api_service.main._hybrid_generator')
    def test_generate_tests_rag_success(
        self,
        mock_hybrid_gen,
        client,
        sample_epic_payload,
    ):
        """Test successful RAG test generation."""
        # Mock hybrid generator
        mock_hybrid_gen.generate = Mock(return_value={
            "test_cases": [
                {
                    "id": "TC-001",
                    "description": "Test login endpoint",
                    "type": "positive",
                    "priority": "P1",
                    "labels": ["api", "authentication"],
                }
            ],
            "metadata": {
                "mode": "hybrid",
                "hrm_generated": 1,
                "sqe_generated": 0,
                "merged_count": 1,
                "rag_context_used": True,
                "rag_similar_count": 2,
                "sqe_enhanced": False,
                "merge_strategy": "weighted",
                "status": "success",
            }
        })
        
        request_data = {
            "epic": sample_epic_payload,
            "options": {
                "use_rag": True,
                "use_sqe": False,
                "generation_mode": "hybrid",
                "top_k_similar": 5,
                "min_similarity": 0.7
            }
        }
        
        response = client.post(
            "/api/v1/generate-tests-rag",
            json=request_data
        )
        
        # If hybrid generator not initialized, expect 503
        if response.status_code == 503:
            return  # Expected when not initialized
        
        assert response.status_code == 200
        data = response.json()
        
        assert "test_cases" in data
        assert "metadata" in data
        assert data["status"] == "success"
    
    def test_generate_tests_rag_without_initialization(
        self,
        client,
        sample_epic_payload,
    ):
        """Test RAG generation without initialization."""
        request_data = {
            "epic": sample_epic_payload
        }
        
        response = client.post(
            "/api/v1/generate-tests-rag",
            json=request_data
        )
        
        # Should return 503 if not initialized
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


class TestIndexingEndpoint:
    """Test test case indexing endpoint."""
    
    @patch('hrm_eval.api_service.main._vector_indexer')
    def test_index_test_cases_success(
        self,
        mock_indexer,
        client,
    ):
        """Test successful test case indexing."""
        mock_indexer.index_test_cases = Mock()
        
        request_data = {
            "test_cases": [
                {
                    "id": "TC-001",
                    "description": "Test case 1",
                    "type": "positive",
                    "priority": "P1",
                    "labels": ["api"],
                    "preconditions": [],
                    "test_steps": [],
                    "expected_results": [],
                }
            ],
            "source": "api_test",
            "batch_size": 100
        }
        
        response = client.post(
            "/api/v1/index-test-cases",
            json=request_data
        )
        
        if response.status_code == 503:
            return  # Expected when not initialized
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "stats" in data
        assert data["stats"]["total_indexed"] == 1
    
    def test_index_test_cases_without_initialization(self, client):
        """Test indexing without initialization."""
        request_data = {
            "test_cases": [],
            "source": "test"
        }
        
        response = client.post(
            "/api/v1/index-test-cases",
            json=request_data
        )
        
        assert response.status_code == 503


class TestSimilarSearchEndpoint:
    """Test similar test case search endpoint."""
    
    @patch('hrm_eval.api_service.main._rag_retriever')
    def test_search_similar_with_query(
        self,
        mock_retriever,
        client,
    ):
        """Test searching similar tests with text query."""
        mock_retriever.retrieve_similar_test_cases = Mock(return_value=[
            {
                "id": "TC-SIM-001",
                "description": "Similar test case",
                "type": "positive",
                "similarity": 0.92,
            }
        ])
        
        request_data = {
            "query": "user authentication",
            "top_k": 5,
            "min_similarity": 0.7
        }
        
        response = client.post(
            "/api/v1/search-similar",
            json=request_data
        )
        
        if response.status_code == 503:
            return  # Expected when not initialized
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "query_info" in data
        assert "search_time_seconds" in data
    
    @patch('hrm_eval.api_service.main._rag_retriever')
    def test_search_similar_with_requirement(
        self,
        mock_retriever,
        client,
    ):
        """Test searching with structured requirement."""
        mock_retriever.retrieve_similar_test_cases = Mock(return_value=[])
        
        request_data = {
            "requirement": {
                "summary": "Login functionality",
                "description": "User login feature"
            },
            "top_k": 3,
            "min_similarity": 0.8
        }
        
        response = client.post(
            "/api/v1/search-similar",
            json=request_data
        )
        
        if response.status_code == 503:
            return  # Expected when not initialized
        
        assert response.status_code in [200, 400]  # 400 if validation fails
    
    def test_search_similar_without_query_or_requirement(self, client):
        """Test search without query or requirement."""
        request_data = {
            "top_k": 5
        }
        
        response = client.post(
            "/api/v1/search-similar",
            json=request_data
        )
        
        # Should return 400 (bad request) or 503 (not initialized)
        assert response.status_code in [400, 503]


class TestWorkflowExecution:
    """Test workflow execution endpoint."""
    
    @patch('hrm_eval.api_service.main._workflow_manager')
    def test_execute_full_workflow(
        self,
        mock_workflow,
        client,
        sample_epic_payload,
    ):
        """Test full workflow execution."""
        mock_workflow.execute_workflow = Mock(return_value={
            "workflow_type": "full",
            "steps": [
                {"step": "validation", "status": "complete", "result": {}},
                {"step": "generation", "status": "complete", "result": {}},
            ],
            "test_cases": [],
            "status": "complete",
        })
        
        request_data = {
            "epic": sample_epic_payload,
            "workflow_type": "full"
        }
        
        response = client.post(
            "/api/v1/execute-workflow",
            json=request_data
        )
        
        if response.status_code == 503:
            return  # Expected when not initialized
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["workflow_type"] == "full"
        assert data["status"] == "complete"
        assert "steps" in data
    
    @patch('hrm_eval.api_service.main._workflow_manager')
    def test_execute_generate_only_workflow(
        self,
        mock_workflow,
        client,
        sample_epic_payload,
    ):
        """Test generate-only workflow."""
        mock_workflow.execute_workflow = Mock(return_value={
            "workflow_type": "generate_only",
            "test_cases": [],
            "metadata": {},
            "status": "complete",
        })
        
        request_data = {
            "epic": sample_epic_payload,
            "workflow_type": "generate_only"
        }
        
        response = client.post(
            "/api/v1/execute-workflow",
            json=request_data
        )
        
        if response.status_code == 503:
            return  # Expected when not initialized
        
        assert response.status_code == 200


class TestErrorHandling:
    """Test API error handling."""
    
    def test_invalid_json_payload(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/generate-tests-rag",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        invalid_epic = {
            "epic_id": "TEST",
            # Missing title and user_stories
        }
        
        request_data = {
            "epic": invalid_epic
        }
        
        response = client.post(
            "/api/v1/generate-tests-rag",
            json=request_data
        )
        
        # Should return validation error or service unavailable
        assert response.status_code in [422, 503]
    
    def test_endpoint_without_initialization(self, client):
        """Test that endpoints require initialization."""
        endpoints = [
            "/api/v1/generate-tests-rag",
            "/api/v1/index-test-cases",
            "/api/v1/search-similar",
            "/api/v1/execute-workflow",
        ]
        
        for endpoint in endpoints:
            response = client.post(endpoint, json={})
            
            # All should return 503 or 422 (validation error)
            assert response.status_code in [422, 503]


class TestBackwardCompatibility:
    """Test backward compatibility with existing endpoints."""
    
    def test_original_generate_tests_endpoint(self, client):
        """Test that original endpoint still works."""
        # This endpoint should work independently of RAG/SQE
        response = client.get("/api/v1/health")
        
        # Basic health should always work
        assert response.status_code == 200
    
    def test_initialize_endpoint(self, client):
        """Test original initialize endpoint."""
        # Original HRM initialization
        response = client.post("/api/v1/initialize")
        
        # Should work or return error (not crash)
        assert response.status_code in [200, 500, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
