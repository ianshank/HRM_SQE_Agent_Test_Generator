"""
Integration tests for HybridTestGenerator.

Tests the hybrid generation combining HRM + SQE + RAG.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

from hrm_eval.orchestration import HybridTestGenerator
from hrm_eval.rag_vector_store import RAGRetriever


@pytest.fixture
def sample_requirements():
    """Sample requirements for testing."""
    return {
        "epic_id": "EPIC-HYB-001",
        "title": "Payment Processing",
        "user_stories": [
            {
                "id": "US-HYB-001",
                "summary": "Credit card payment",
                "description": "Process credit card payments securely",
                "acceptance_criteria": [
                    {"criteria": "Card validation"},
                    {"criteria": "Secure payment processing"},
                ],
                "tech_stack": ["Stripe", "FastAPI"],
            }
        ],
        "tech_stack": ["Stripe", "FastAPI", "PostgreSQL"],
        "architecture": "Payment Gateway Integration",
    }


@pytest.fixture
def mock_hrm_generator_with_tests():
    """Mock HRM generator that returns test cases."""
    generator = Mock()
    generator.generate_test_cases = Mock(return_value=[
        Mock(
            id=f"TC-HRM-{i:03d}",
            description=f"HRM generated test {i}",
            type=Mock(value="positive" if i % 2 == 0 else "negative"),
            priority=Mock(value="P1" if i < 3 else "P2"),
            labels=["hrm", "generated"],
            preconditions=[],
            test_steps=[],
            expected_results=[],
        )
        for i in range(1, 11)  # Generate 10 test cases
    ])
    return generator


@pytest.fixture
def mock_sqe_agent_with_tests():
    """Mock SQE agent that returns test cases."""
    agent = Mock()
    agent.generate_from_epic = Mock(return_value={
        "hrm_test_cases": [
            {
                "id": f"TC-SQE-{i:03d}",
                "description": f"SQE generated test {i}",
                "type": "positive" if i % 3 == 0 else "negative",
                "priority": "P1" if i < 4 else "P3",
                "labels": ["sqe", "enhanced"],
            }
            for i in range(1, 6)  # Generate 5 test cases
        ],
        "test_coverage_analysis": {
            "coverage_percentage": 85.5,
            "positive_tests": 3,
            "negative_tests": 2,
        }
    })
    return agent


@pytest.fixture
def mock_rag_retriever_with_similar():
    """Mock RAG retriever with similar tests."""
    retriever = Mock(spec=RAGRetriever)
    retriever.retrieve_similar_test_cases = Mock(return_value=[
        {
            "id": "TC-HIST-001",
            "description": "Historical test for payment",
            "type": "positive",
            "similarity": 0.92,
        },
        {
            "id": "TC-HIST-002",
            "description": "Historical negative test",
            "type": "negative",
            "similarity": 0.85,
        },
    ])
    retriever.build_context = Mock(return_value="Historical context from RAG")
    return retriever


class TestHybridGenerationModes:
    """Test different generation modes."""
    
    def test_hrm_only_mode(
        self,
        sample_requirements,
        mock_hrm_generator_with_tests,
    ):
        """Test HRM-only mode generates correct number of tests."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=None,
            rag_retriever=None,
            mode="hrm_only",
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        assert result["metadata"]["mode"] == "hrm_only"
        assert result["metadata"]["hrm_generated"] == 10
        assert result["metadata"]["sqe_generated"] == 0
        assert result["metadata"]["rag_context_used"] is False
        assert result["metadata"]["sqe_enhanced"] is False
    
    def test_sqe_only_mode(
        self,
        sample_requirements,
        mock_sqe_agent_with_tests,
    ):
        """Test SQE-only mode."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=None,
            sqe_agent=mock_sqe_agent_with_tests,
            rag_retriever=None,
            mode="sqe_only",
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        assert result["metadata"]["mode"] == "sqe_only"
        assert result["metadata"]["sqe_generated"] == 5
        assert len(result["test_cases"]) == 5
    
    def test_hybrid_mode_combines_hrm_and_sqe(
        self,
        sample_requirements,
        mock_hrm_generator_with_tests,
        mock_sqe_agent_with_tests,
        mock_rag_retriever_with_similar,
    ):
        """Test hybrid mode combines HRM and SQE results."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=mock_sqe_agent_with_tests,
            rag_retriever=mock_rag_retriever_with_similar,
            mode="hybrid",
            merge_strategy="weighted",
            hrm_weight=0.6,
            sqe_weight=0.4,
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        assert result["metadata"]["mode"] == "hybrid"
        assert result["metadata"]["hrm_generated"] == 10
        assert result["metadata"]["sqe_generated"] == 5
        assert result["metadata"]["rag_context_used"] is True
        assert result["metadata"]["sqe_enhanced"] is True
        assert result["metadata"]["merge_strategy"] == "weighted"
        
        # Verify both HRM and SQE were called
        mock_hrm_generator_with_tests.generate_test_cases.assert_called_once()
        mock_sqe_agent_with_tests.generate_from_epic.assert_called_once()
        mock_rag_retriever_with_similar.retrieve_similar_test_cases.assert_called()


class TestMergeStrategies:
    """Test different merge strategies."""
    
    def test_weighted_merge_strategy(
        self,
        sample_requirements,
        mock_hrm_generator_with_tests,
        mock_sqe_agent_with_tests,
    ):
        """Test weighted merge strategy."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=mock_sqe_agent_with_tests,
            rag_retriever=None,
            mode="hybrid",
            merge_strategy="weighted",
            hrm_weight=0.7,
            sqe_weight=0.3,
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        # With 10 HRM and 5 SQE tests, weighted (0.7, 0.3) should give ~7 HRM + 1-2 SQE
        assert result["metadata"]["merge_strategy"] == "weighted"
        merged_count = result["metadata"]["merged_count"]
        assert merged_count > 0
    
    def test_union_merge_strategy(
        self,
        sample_requirements,
        mock_hrm_generator_with_tests,
        mock_sqe_agent_with_tests,
    ):
        """Test union merge strategy (all unique tests)."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=mock_sqe_agent_with_tests,
            rag_retriever=None,
            mode="hybrid",
            merge_strategy="union",
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        assert result["metadata"]["merge_strategy"] == "union"
        # Union should include most/all tests from both
        assert result["metadata"]["merged_count"] >= 10  # At least HRM tests
    
    def test_intersection_merge_strategy(
        self,
        sample_requirements,
        mock_hrm_generator_with_tests,
        mock_sqe_agent_with_tests,
    ):
        """Test intersection merge strategy (only common tests)."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=mock_sqe_agent_with_tests,
            rag_retriever=None,
            mode="hybrid",
            merge_strategy="intersection",
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        assert result["metadata"]["merge_strategy"] == "intersection"
        # Intersection should have fewer tests (only overlaps)
        assert result["metadata"]["merged_count"] <= min(10, 5)


class TestRAGIntegration:
    """Test RAG integration in hybrid generator."""
    
    def test_rag_context_retrieval(
        self,
        sample_requirements,
        mock_hrm_generator_with_tests,
        mock_rag_retriever_with_similar,
    ):
        """Test that RAG context is retrieved and used."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=None,
            rag_retriever=mock_rag_retriever_with_similar,
            mode="hybrid",
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        # Verify RAG retrieval was called
        mock_rag_retriever_with_similar.retrieve_similar_test_cases.assert_called_once()
        mock_rag_retriever_with_similar.build_context.assert_called_once()
        
        assert result["metadata"]["rag_context_used"] is True
        assert result["metadata"]["rag_similar_count"] == 2
    
    def test_generation_without_rag(
        self,
        sample_requirements,
        mock_hrm_generator_with_tests,
    ):
        """Test generation works without RAG."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=None,
            rag_retriever=None,  # No RAG
            mode="hrm_only",
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        assert result["metadata"]["rag_context_used"] is False
        assert result["metadata"]["rag_similar_count"] == 0
    
    def test_rag_with_empty_results(
        self,
        sample_requirements,
        mock_hrm_generator_with_tests,
    ):
        """Test RAG with no similar tests found."""
        empty_rag = Mock(spec=RAGRetriever)
        empty_rag.retrieve_similar_test_cases = Mock(return_value=[])
        empty_rag.build_context = Mock(return_value="")
        
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=None,
            rag_retriever=empty_rag,
            mode="hybrid",
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        # Should still work with empty RAG results
        assert "test_cases" in result
        assert result["metadata"]["rag_similar_count"] == 0


class TestGenerationMetadata:
    """Test metadata tracking in generation."""
    
    def test_metadata_completeness(
        self,
        sample_requirements,
        mock_hrm_generator_with_tests,
        mock_sqe_agent_with_tests,
        mock_rag_retriever_with_similar,
    ):
        """Test that all metadata fields are populated."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=mock_sqe_agent_with_tests,
            rag_retriever=mock_rag_retriever_with_similar,
            mode="hybrid",
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        metadata = result["metadata"]
        
        # Check all expected metadata fields
        assert "mode" in metadata
        assert "hrm_generated" in metadata
        assert "sqe_generated" in metadata
        assert "merged_count" in metadata
        assert "rag_context_used" in metadata
        assert "rag_similar_count" in metadata
        assert "sqe_enhanced" in metadata
        assert "merge_strategy" in metadata
        assert "generation_time_seconds" in metadata
        assert "status" in metadata
    
    def test_generation_timing(
        self,
        sample_requirements,
        mock_hrm_generator_with_tests,
    ):
        """Test that generation time is tracked."""
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=None,
            rag_retriever=None,
            mode="hrm_only",
        )
        
        result = hybrid_gen.generate(sample_requirements)
        
        assert "generation_time_seconds" in result["metadata"]
        assert result["metadata"]["generation_time_seconds"] >= 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_requirements(self, mock_hrm_generator_with_tests):
        """Test generation with empty requirements."""
        empty_reqs = {
            "epic_id": "EMPTY",
            "title": "",
            "user_stories": [],
        }
        
        hybrid_gen = HybridTestGenerator(
            hrm_generator=mock_hrm_generator_with_tests,
            sqe_agent=None,
            rag_retriever=None,
            mode="hrm_only",
        )
        
        result = hybrid_gen.generate(empty_reqs)
        
        # Should handle empty requirements gracefully
        assert "test_cases" in result
        assert "metadata" in result
    
    def test_large_test_set_merging(self):
        """Test merging large sets of test cases."""
        # Mock HRM with 100 tests
        large_hrm = Mock()
        large_hrm.generate_test_cases = Mock(return_value=[
            Mock(
                id=f"TC-HRM-{i:04d}",
                description=f"HRM test {i}",
                type=Mock(value="positive"),
                priority=Mock(value="P2"),
                labels=["hrm"],
                preconditions=[],
                test_steps=[],
                expected_results=[],
            )
            for i in range(100)
        ])
        
        # Mock SQE with 50 tests
        large_sqe = Mock()
        large_sqe.generate_from_epic = Mock(return_value={
            "hrm_test_cases": [
                {
                    "id": f"TC-SQE-{i:04d}",
                    "description": f"SQE test {i}",
                    "type": "positive",
                    "priority": "P2",
                }
                for i in range(50)
            ]
        })
        
        hybrid_gen = HybridTestGenerator(
            hrm_generator=large_hrm,
            sqe_agent=large_sqe,
            rag_retriever=None,
            mode="hybrid",
            merge_strategy="union",
        )
        
        requirements = {"epic_id": "LARGE", "user_stories": []}
        result = hybrid_gen.generate(requirements)
        
        # Should handle large sets without issues
        assert result["metadata"]["merged_count"] > 0
        assert result["metadata"]["hrm_generated"] == 100
        assert result["metadata"]["sqe_generated"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
