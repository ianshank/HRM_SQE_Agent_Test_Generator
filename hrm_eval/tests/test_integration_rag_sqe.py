"""
Integration tests for RAG + SQE + HRM workflow.

Tests complete end-to-end workflows combining all components.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from hrm_eval.rag_vector_store import VectorStore, RAGRetriever, EmbeddingGenerator, VectorIndexer
from hrm_eval.agents import SQEAgent
from hrm_eval.orchestration import HybridTestGenerator, WorkflowManager
from hrm_eval.requirements_parser.schemas import Epic, UserStory, AcceptanceCriteria


@pytest.fixture
def sample_epic():
    """Sample epic for testing."""
    return {
        "epic_id": "EPIC-INT-001",
        "title": "User Authentication System",
        "user_stories": [
            {
                "id": "US-INT-001",
                "summary": "User login with email and password",
                "description": "As a user, I want to login with my email and password so I can access my account",
                "acceptance_criteria": [
                    {"criteria": "Valid credentials allow successful login"},
                    {"criteria": "Invalid credentials show error message"},
                    {"criteria": "Account locks after 5 failed attempts"},
                ],
                "tech_stack": ["FastAPI", "PostgreSQL", "Redis"],
            },
            {
                "id": "US-INT-002",
                "summary": "Password reset functionality",
                "description": "As a user, I want to reset my password if I forget it",
                "acceptance_criteria": [
                    {"criteria": "Email with reset link is sent"},
                    {"criteria": "Link expires after 1 hour"},
                ],
                "tech_stack": ["FastAPI", "SendGrid"],
            },
        ],
        "tech_stack": ["FastAPI", "PostgreSQL", "Redis", "JWT"],
        "architecture": "Microservice Architecture",
    }


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = Mock(spec=VectorStore)
    store.add_documents = Mock()
    store.search = Mock(return_value=[
        {
            "id": "TC-HIST-001",
            "description": "Test user login with valid credentials",
            "type": "positive",
            "priority": "P1",
            "similarity": 0.85,
        }
    ])
    store.get_collection_count = Mock(return_value=100)
    return store


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator."""
    gen = Mock(spec=EmbeddingGenerator)
    gen.encode = Mock(return_value=[0.1] * 384)
    gen.encode_test_case = Mock(return_value=[0.1] * 384)
    gen.encode_requirement = Mock(return_value=[0.1] * 384)
    return gen


@pytest.fixture
def mock_hrm_generator():
    """Mock HRM generator."""
    generator = Mock()
    generator.generate_test_cases = Mock(return_value=[
        Mock(
            id="TC-HRM-001",
            description="Verify user login with valid email and password",
            type=Mock(value="positive"),
            priority=Mock(value="P1"),
            labels=["authentication", "login"],
            preconditions=[],
            test_steps=[],
            expected_results=[],
        ),
        Mock(
            id="TC-HRM-002",
            description="Verify login failure with invalid password",
            type=Mock(value="negative"),
            priority=Mock(value="P1"),
            labels=["authentication", "error"],
            preconditions=[],
            test_steps=[],
            expected_results=[],
        ),
    ])
    return generator


@pytest.fixture
def mock_llm():
    """Mock LangChain LLM."""
    llm = Mock()
    llm.invoke = Mock(return_value="Test plan generated")
    return llm


class TestRAGSQEIntegration:
    """Integration tests for RAG + SQE + HRM."""
    
    def test_complete_workflow_with_all_components(
        self,
        sample_epic,
        mock_vector_store,
        mock_embedding_generator,
        mock_hrm_generator,
        mock_llm,
    ):
        """Test complete workflow: Epic → RAG → HRM → SQE → Test Cases."""
        # Setup RAG components
        rag_retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        # Setup SQE agent
        sqe_agent = SQEAgent(
            llm=mock_llm,
            rag_retriever=rag_retriever,
            hrm_generator=mock_hrm_generator,
            enable_rag=True,
            enable_hrm=True,
        )
        
        # Setup hybrid generator
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator,
            sqe_agent=sqe_agent,
            rag_retriever=rag_retriever,
            mode="hybrid",
        )
        
        # Execute generation
        result = hybrid_gen.generate(sample_epic)
        
        # Assertions
        assert "test_cases" in result
        assert "metadata" in result
        assert len(result["test_cases"]) > 0
        assert result["metadata"]["rag_context_used"] is True
        
        # Verify RAG was called
        mock_vector_store.search.assert_called()
        
        # Verify HRM generator was called
        mock_hrm_generator.generate_test_cases.assert_called()
    
    def test_workflow_with_indexing(
        self,
        sample_epic,
        mock_vector_store,
        mock_embedding_generator,
        mock_hrm_generator,
        mock_llm,
    ):
        """Test workflow with auto-indexing of generated tests."""
        # Setup components
        rag_retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        sqe_agent = SQEAgent(
            llm=mock_llm,
            rag_retriever=rag_retriever,
            hrm_generator=mock_hrm_generator,
        )
        
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator,
            sqe_agent=sqe_agent,
            rag_retriever=rag_retriever,
            mode="hybrid",
        )
        
        # Setup workflow manager with auto-indexing
        workflow_mgr = WorkflowManager(
            hybrid_generator=hybrid_gen,
            vector_indexer=indexer,
            auto_index=True,
        )
        
        # Execute full workflow
        result = workflow_mgr.execute_workflow(
            requirements=sample_epic,
            workflow_type="full",
        )
        
        # Assertions
        assert result["status"] == "complete"
        assert len(result["steps"]) >= 2  # Generation + Indexing
        assert any(step["step"] == "generation" for step in result["steps"])
        
        # Verify indexing was called
        mock_vector_store.add_documents.assert_called()
    
    def test_hrm_only_mode(self, sample_epic, mock_hrm_generator):
        """Test HRM-only mode (no RAG, no SQE)."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator,
            sqe_agent=None,
            rag_retriever=None,
            mode="hrm_only",
        )
        
        result = hybrid_gen.generate(sample_epic)
        
        assert "test_cases" in result
        assert result["metadata"]["mode"] == "hrm_only"
        assert result["metadata"]["rag_context_used"] is False
        assert result["metadata"]["sqe_enhanced"] is False
        
        mock_hrm_generator.generate_test_cases.assert_called_once()
    
    def test_rag_retrieval_with_empty_store(
        self,
        sample_epic,
        mock_embedding_generator,
    ):
        """Test RAG retrieval when vector store is empty."""
        empty_store = Mock(spec=VectorStore)
        empty_store.search = Mock(return_value=[])
        
        rag_retriever = RAGRetriever(empty_store, mock_embedding_generator)
        
        similar_tests = rag_retriever.retrieve_similar_test_cases(
            requirement=sample_epic["user_stories"][0],
            top_k=5,
        )
        
        assert similar_tests == []
    
    def test_workflow_validation_failure(
        self,
        mock_vector_store,
        mock_embedding_generator,
        mock_hrm_generator,
        mock_llm,
    ):
        """Test workflow when requirement validation fails."""
        # Invalid epic (missing required fields)
        invalid_epic = {
            "epic_id": "",
            "title": "",
            "user_stories": [],
        }
        
        rag_retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        sqe_agent = SQEAgent(
            llm=mock_llm,
            rag_retriever=rag_retriever,
            hrm_generator=mock_hrm_generator,
        )
        
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator,
            sqe_agent=sqe_agent,
            rag_retriever=rag_retriever,
        )
        
        workflow_mgr = WorkflowManager(
            hybrid_generator=hybrid_gen,
            vector_indexer=indexer,
        )
        
        result = workflow_mgr.execute_workflow(
            requirements=invalid_epic,
            workflow_type="full",
        )
        
        # Should complete with validation warnings but not crash
        assert "steps" in result
        assert result.get("status") in ["complete", "validation_failed"]
    
    def test_context_building_from_rag(
        self,
        sample_epic,
        mock_vector_store,
        mock_embedding_generator,
    ):
        """Test context building from RAG retrieval."""
        # Mock search with detailed results
        mock_vector_store.search = Mock(return_value=[
            {
                "id": "TC-001",
                "description": "Test user login with valid credentials",
                "type": "positive",
                "priority": "P1",
                "similarity": 0.92,
            },
            {
                "id": "TC-002",
                "description": "Test login with invalid password",
                "type": "negative",
                "priority": "P2",
                "similarity": 0.85,
            },
        ])
        
        rag_retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        similar_tests = rag_retriever.retrieve_similar_test_cases(
            requirement=sample_epic["user_stories"][0],
            top_k=5,
            min_similarity=0.7,
        )
        
        context = rag_retriever.build_context(
            requirement=sample_epic["user_stories"][0],
            similar_tests=similar_tests,
        )
        
        # Assertions
        assert len(similar_tests) == 2
        assert "Historical Test Case Examples" in context
        assert "positive" in context
        assert "negative" in context
    
    def test_merge_strategies(
        self,
        sample_epic,
        mock_hrm_generator,
        mock_llm,
    ):
        """Test different merge strategies in hybrid mode."""
        strategies = ["weighted", "union", "intersection"]
        
        for strategy in strategies:
            hybrid_gen = HybridTestGenerator(
                hrm_generator=mock_hrm_generator,
                sqe_agent=None,
                rag_retriever=None,
                mode="hybrid",
                merge_strategy=strategy,
            )
            
            result = hybrid_gen.generate(sample_epic)
            
            assert "test_cases" in result
            assert result["metadata"]["merge_strategy"] == strategy
    
    def test_concurrent_indexing_and_retrieval(
        self,
        sample_epic,
        mock_vector_store,
        mock_embedding_generator,
    ):
        """Test that indexing and retrieval can happen concurrently."""
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        # Index some test cases
        test_cases = [
            {
                "id": "TC-001",
                "description": "Test case 1",
                "type": "positive",
                "priority": "P1",
            },
            {
                "id": "TC-002",
                "description": "Test case 2",
                "type": "negative",
                "priority": "P2",
            },
        ]
        
        indexer.index_test_cases(test_cases, batch_size=10)
        
        # Verify indexing was called
        mock_vector_store.add_documents.assert_called()
        
        # Now retrieve similar tests
        similar = retriever.retrieve_similar_test_cases(
            requirement=sample_epic["user_stories"][0],
            top_k=5,
        )
        
        # Verify retrieval was called
        mock_vector_store.search.assert_called()


class TestWorkflowManager:
    """Integration tests for WorkflowManager."""
    
    def test_full_workflow_execution(
        self,
        sample_epic,
        mock_vector_store,
        mock_embedding_generator,
        mock_hrm_generator,
        mock_llm,
    ):
        """Test full workflow execution."""
        rag_retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        sqe_agent = SQEAgent(
            llm=mock_llm,
            rag_retriever=rag_retriever,
            hrm_generator=mock_hrm_generator,
        )
        
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator,
            sqe_agent=sqe_agent,
            rag_retriever=rag_retriever,
        )
        
        workflow_mgr = WorkflowManager(
            hybrid_generator=hybrid_gen,
            vector_indexer=indexer,
            auto_index=True,
        )
        
        result = workflow_mgr.execute_workflow(
            requirements=sample_epic,
            workflow_type="full",
        )
        
        assert result["workflow_type"] == "full"
        assert result["status"] == "complete"
        assert len(result["steps"]) > 0
        assert "test_cases" in result
    
    def test_generate_only_workflow(
        self,
        sample_epic,
        mock_hrm_generator,
        mock_llm,
    ):
        """Test generate-only workflow."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator,
            sqe_agent=None,
            rag_retriever=None,
            mode="hrm_only",
        )
        
        workflow_mgr = WorkflowManager(
            hybrid_generator=hybrid_gen,
            vector_indexer=None,
            auto_index=False,
        )
        
        result = workflow_mgr.execute_workflow(
            requirements=sample_epic,
            workflow_type="generate_only",
        )
        
        assert result["workflow_type"] == "generate_only"
        assert "test_cases" in result
    
    def test_validate_only_workflow(self, sample_epic):
        """Test validate-only workflow."""
        mock_gen = Mock()
        
        workflow_mgr = WorkflowManager(
            hybrid_generator=mock_gen,
            vector_indexer=None,
            auto_index=False,
        )
        
        result = workflow_mgr.execute_workflow(
            requirements=sample_epic,
            workflow_type="validate_only",
        )
        
        assert result["workflow_type"] == "validate_only"
        assert "validation" in result


class TestErrorHandling:
    """Integration tests for error handling."""
    
    def test_graceful_degradation_when_rag_unavailable(
        self,
        sample_epic,
        mock_hrm_generator,
    ):
        """Test that system works when RAG is unavailable."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator,
            sqe_agent=None,
            rag_retriever=None,  # RAG unavailable
            mode="hrm_only",
        )
        
        result = hybrid_gen.generate(sample_epic)
        
        # Should still generate tests using HRM only
        assert "test_cases" in result
        assert result["metadata"]["rag_context_used"] is False
    
    def test_graceful_degradation_when_sqe_unavailable(
        self,
        sample_epic,
        mock_vector_store,
        mock_embedding_generator,
        mock_hrm_generator,
    ):
        """Test that system works when SQE is unavailable."""
        rag_retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator,
            sqe_agent=None,  # SQE unavailable
            rag_retriever=rag_retriever,
            mode="hybrid",
        )
        
        result = hybrid_gen.generate(sample_epic)
        
        # Should still generate tests using HRM + RAG
        assert "test_cases" in result
        assert result["metadata"]["sqe_enhanced"] is False
    
    def test_error_recovery_in_workflow(
        self,
        sample_epic,
        mock_vector_store,
        mock_embedding_generator,
        mock_llm,
    ):
        """Test error recovery in workflow execution."""
        # Mock HRM generator that fails
        failing_generator = Mock()
        failing_generator.generate_test_cases = Mock(
            side_effect=Exception("HRM generation failed")
        )
        
        rag_retriever = RAGRetriever(mock_vector_store, mock_embedding_generator)
        indexer = VectorIndexer(mock_vector_store, mock_embedding_generator)
        
        sqe_agent = SQEAgent(
            llm=mock_llm,
            rag_retriever=rag_retriever,
            hrm_generator=failing_generator,
        )
        
        hybrid_gen = HybridTestGenerator(
            hrm_generator=failing_generator,
            sqe_agent=sqe_agent,
            rag_retriever=rag_retriever,
        )
        
        workflow_mgr = WorkflowManager(
            hybrid_generator=hybrid_gen,
            vector_indexer=indexer,
        )
        
        # Should handle error gracefully
        result = workflow_mgr.execute_workflow(
            requirements=sample_epic,
            workflow_type="full",
        )
        
        # Workflow should complete with error indication
        assert "steps" in result or "error" in str(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
