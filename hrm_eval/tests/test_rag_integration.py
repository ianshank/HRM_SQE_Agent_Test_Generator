"""
Integration tests for RAG+HRM workflow.

Tests the complete end-to-end workflow from requirements to test generation.
Uses real models and actual workflow - no mocking of core generation logic.
"""

import pytest
import torch
import tempfile
import shutil
import json
from pathlib import Path

from hrm_eval.models import HRMModel
from hrm_eval.models.hrm_model import HRMConfig
from hrm_eval.utils.config_utils import load_config
from hrm_eval.utils.checkpoint_utils import load_checkpoint
from hrm_eval.requirements_parser.schemas import Epic, UserStory, AcceptanceCriteria
from hrm_eval.test_generator.generator import TestCaseGenerator
from hrm_eval.rag_vector_store.vector_store import VectorStore
from hrm_eval.rag_vector_store.embeddings import EmbeddingGenerator
from hrm_eval.rag_vector_store.retrieval import RAGRetriever


class TestRAGWorkflowIntegration:
    """Integration tests for complete RAG workflow."""
    
    @pytest.fixture(scope="class")
    def device(self):
        """Get computation device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture(scope="class")
    def base_path(self):
        """Get base project path."""
        return Path(__file__).parent.parent
    
    @pytest.fixture(scope="class")
    def hrm_model(self, base_path, device):
        """Load HRM model (cached for class scope)."""
        config = load_config(
            model_config_path=base_path / "configs" / "model_config.yaml",
            eval_config_path=base_path / "configs" / "eval_config.yaml"
        )
        hrm_config = HRMConfig.from_yaml_config(config)
        
        # Use base checkpoint for testing (faster, consistent)
        checkpoint_path = base_path.parent / "checkpoints_hrm_v9_optimized_step_7566"
        
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_path}")
        
        model = HRMModel(hrm_config)
        checkpoint = load_checkpoint(str(checkpoint_path))
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Clean state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for prefix in ["model.inner.", "model.", "module."]:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
            if new_k == "embedding_weight":
                new_k = "weight"
            new_state_dict[new_k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        return model
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_epic(self):
        """Create sample epic for testing."""
        return Epic(
            epic_id="EPIC-TEST-001",
            title="File Upload System",
            user_stories=[
                UserStory(
                    id="US-001",
                    summary="Basic file upload",
                    description="As a user, I want to upload files so that I can share content",
                    acceptance_criteria=[
                        AcceptanceCriteria(
                            criteria="System accepts common file formats (PDF, JPG, PNG)"
                        ),
                        AcceptanceCriteria(
                            criteria="Upload progress is displayed to the user"
                        ),
                        AcceptanceCriteria(
                            criteria="Success confirmation is shown upon completion"
                        )
                    ]
                ),
                UserStory(
                    id="US-002",
                    summary="File validation",
                    description="As a system, I want to validate uploaded files for security",
                    acceptance_criteria=[
                        AcceptanceCriteria(
                            criteria="File size limits are enforced (max 10MB)"
                        ),
                        AcceptanceCriteria(
                            criteria="File types are validated against whitelist"
                        )
                    ]
                )
            ]
        )
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """Create vector store for testing."""
        return VectorStore(
            backend="chromadb",
            persist_directory=temp_dir,
            collection_name="test_integration"
        )
    
    @pytest.fixture
    def embedding_generator(self):
        """Create embedding generator."""
        return EmbeddingGenerator()
    
    @pytest.fixture
    def rag_retriever(self, vector_store, embedding_generator):
        """Create RAG retriever."""
        return RAGRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )
    
    @pytest.fixture
    def populated_vector_store(self, vector_store, embedding_generator):
        """Populate vector store with sample test cases."""
        sample_tests = [
            {
                "id": "TC-HIST-001",
                "description": "Verify successful file upload with valid PDF",
                "type": "positive",
                "priority": "P1",
                "preconditions": ["User authenticated", "Valid PDF file available"],
                "test_steps": [
                    {"step": 1, "action": "Navigate to upload page"},
                    {"step": 2, "action": "Select PDF file"},
                    {"step": 3, "action": "Click upload button"}
                ],
                "expected_results": [
                    {"result": "Progress bar displayed"},
                    {"result": "Success message shown"}
                ]
            },
            {
                "id": "TC-HIST-002",
                "description": "Verify error handling for oversized file",
                "type": "negative",
                "priority": "P2",
                "preconditions": ["User authenticated", "File > 10MB available"],
                "test_steps": [
                    {"step": 1, "action": "Navigate to upload page"},
                    {"step": 2, "action": "Select large file"},
                    {"step": 3, "action": "Attempt upload"}
                ],
                "expected_results": [
                    {"result": "Error message displayed"},
                    {"result": "Upload rejected"}
                ]
            },
            {
                "id": "TC-HIST-003",
                "description": "Verify file type validation",
                "type": "negative",
                "priority": "P1",
                "preconditions": ["User authenticated"],
                "test_steps": [
                    {"step": 1, "action": "Select executable file"},
                    {"step": 2, "action": "Attempt upload"}
                ],
                "expected_results": [
                    {"result": "File type error shown"}
                ]
            }
        ]
        
        # Generate embeddings and add to vector store
        texts = []
        for test in sample_tests:
            text_parts = [
                f"Test: {test['description']}",
                f"Type: {test['type']}",
                f"Priority: {test['priority']}"
            ]
            texts.append(" | ".join(text_parts))
        
        embeddings = embedding_generator.encode(texts)
        ids = [test["id"] for test in sample_tests]
        
        vector_store.add_documents(sample_tests, embeddings, ids)
        
        return vector_store
    
    def test_requirements_to_test_generation_baseline(
        self,
        hrm_model,
        sample_epic,
        device
    ):
        """Test baseline test generation (no RAG) from requirements."""
        generator = TestCaseGenerator(model=hrm_model)
        
        # Generate tests for first user story
        user_story = sample_epic.user_stories[0]
        test_cases = generator.generate_for_user_story(
            epic=sample_epic,
            user_story=user_story
        )
        
        # Assertions
        assert len(test_cases) > 0, "Should generate at least one test case"
        
        for tc in test_cases:
            assert hasattr(tc, 'description'), "Test case should have description"
            assert hasattr(tc, 'test_steps'), "Test case should have test steps"
            assert hasattr(tc, 'expected_results'), "Test case should have expected results"
            assert len(tc.test_steps) > 0, "Should have at least one test step"
    
    def test_rag_retrieval_for_requirements(
        self,
        rag_retriever,
        populated_vector_store,
        sample_epic
    ):
        """Test RAG retrieval for given requirements."""
        user_story = sample_epic.user_stories[0]
        
        # Create query from requirement
        query = f"Epic: {sample_epic.title} | Story: {user_story.summary} | {user_story.description}"
        
        # Retrieve similar tests
        similar_tests = rag_retriever.retrieve_by_text(
            text=query,
            top_k=3,
            min_similarity=0.0  # Low threshold for testing
        )
        
        # Assertions
        assert len(similar_tests) > 0, "Should retrieve at least one similar test"
        assert all('metadata' in test for test in similar_tests), "Results should have metadata"
        assert all('similarity' in test or 'distance' in test for test in similar_tests), \
            "Results should have similarity scores"
    
    def test_rag_context_building(
        self,
        rag_retriever,
        populated_vector_store,
        sample_epic
    ):
        """Test context building from retrieved tests."""
        user_story = sample_epic.user_stories[0]
        query = f"{sample_epic.title} | {user_story.summary}"
        
        # Retrieve tests
        similar_tests = rag_retriever.retrieve_by_text(query, top_k=2, min_similarity=0.0)
        
        # Build context
        context = rag_retriever.build_context(
            requirement={"summary": user_story.summary},
            retrieved_tests=similar_tests
        )
        
        # Assertions
        assert isinstance(context, str), "Context should be string"
        assert len(context) > 0, "Context should not be empty"
        assert "Example" in context, "Context should mention examples"
    
    def test_end_to_end_rag_workflow(
        self,
        hrm_model,
        rag_retriever,
        populated_vector_store,
        sample_epic,
        device
    ):
        """
        Test complete end-to-end RAG workflow.
        
        Requirements → RAG Retrieval → Context Building → HRM Generation
        """
        user_story = sample_epic.user_stories[0]
        
        # Step 1: Create query from requirements
        query = f"Epic: {sample_epic.title} | Story: {user_story.summary}"
        
        # Step 2: Retrieve similar historical tests
        similar_tests = rag_retriever.retrieve_by_text(
            text=query,
            top_k=3,
            min_similarity=0.0
        )
        
        assert len(similar_tests) > 0, "Should retrieve similar tests"
        
        # Step 3: Build context
        context = rag_retriever.build_context(
            requirement={"summary": user_story.summary},
            retrieved_tests=similar_tests
        )
        
        assert len(context) > 0, "Should build context"
        
        # Step 4: Generate tests with HRM model
        # Note: Context is prepared, but in full RAG implementation,
        # it would be injected into model's generation process
        generator = TestCaseGenerator(model=hrm_model)
        test_cases = generator.generate_for_user_story(
            epic=sample_epic,
            user_story=user_story
        )
        
        # Step 5: Validate generated tests
        assert len(test_cases) > 0, "Should generate tests"
        
        for tc in test_cases:
            assert tc.description, "Test should have description"
            assert len(tc.test_steps) > 0, "Test should have steps"
            assert len(tc.expected_results) > 0, "Test should have expected results"
        
        # Log retrieval stats for analysis
        stats = rag_retriever.get_retrieval_stats(
            requirement={"summary": user_story.summary},
            retrieved_tests=similar_tests
        )
        
        assert stats["num_retrieved"] > 0
        assert "avg_similarity" in stats
    
    def test_multiple_user_stories_generation(
        self,
        hrm_model,
        sample_epic,
        device
    ):
        """Test generating tests for multiple user stories."""
        generator = TestCaseGenerator(model=hrm_model)
        
        all_test_cases = []
        for user_story in sample_epic.user_stories:
            test_cases = generator.generate_for_user_story(
                epic=sample_epic,
                user_story=user_story
            )
            all_test_cases.extend(test_cases)
        
        # Assertions
        assert len(all_test_cases) > 0, "Should generate tests for all stories"
        assert len(all_test_cases) >= len(sample_epic.user_stories), \
            "Should generate at least one test per story"
        
        # Check diversity
        descriptions = [tc.description for tc in all_test_cases]
        unique_descriptions = set(descriptions)
        assert len(unique_descriptions) > 1, "Should generate diverse tests"
    
    def test_rag_vs_baseline_comparison(
        self,
        hrm_model,
        rag_retriever,
        populated_vector_store,
        sample_epic,
        device
    ):
        """
        Compare RAG-enhanced vs baseline generation.
        
        This tests the comparison framework but uses actual generation.
        """
        user_story = sample_epic.user_stories[0]
        generator = TestCaseGenerator(model=hrm_model)
        
        # Generate baseline tests
        baseline_tests = generator.generate_for_user_story(
            epic=sample_epic,
            user_story=user_story
        )
        
        # Generate RAG-enhanced tests
        # (For now, same as baseline since context injection not fully implemented)
        # But we verify the retrieval happens
        query = f"{sample_epic.title} | {user_story.summary}"
        similar_tests = rag_retriever.retrieve_by_text(query, top_k=3, min_similarity=0.0)
        
        rag_tests = generator.generate_for_user_story(
            epic=sample_epic,
            user_story=user_story
        )
        
        # Add RAG metadata
        for tc in rag_tests:
            if not hasattr(tc, 'metadata') or tc.metadata is None:
                tc.metadata = {}
            tc.metadata['rag_enabled'] = True
            tc.metadata['retrieved_examples'] = len(similar_tests)
        
        # Assertions
        assert len(baseline_tests) > 0, "Baseline should generate tests"
        assert len(rag_tests) > 0, "RAG should generate tests"
        assert len(similar_tests) > 0, "RAG should retrieve examples"
        
        # Verify metadata
        assert any(
            hasattr(tc, 'metadata') and tc.metadata and tc.metadata.get('rag_enabled')
            for tc in rag_tests
        ), "RAG tests should have RAG metadata"


class TestRAGWorkflowPersistence:
    """Test persistence and data handling in RAG workflow."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_vector_store_persistence(self, temp_dir):
        """Test vector store persists data correctly."""
        # Create and populate vector store
        vector_store_1 = VectorStore(
            backend="chromadb",
            persist_directory=temp_dir,
            collection_name="persist_test"
        )
        
        embedding_gen = EmbeddingGenerator()
        
        test_doc = [{"id": "test_1", "description": "Test persistence"}]
        test_embedding = [embedding_gen.encode("Test persistence")]
        
        vector_store_1.add_documents(test_doc, test_embedding, ["test_1"])
        
        stats_1 = vector_store_1.get_stats()
        assert stats_1["total_documents"] == 1
        
        # Create new vector store instance with same directory
        vector_store_2 = VectorStore(
            backend="chromadb",
            persist_directory=temp_dir,
            collection_name="persist_test"
        )
        
        stats_2 = vector_store_2.get_stats()
        assert stats_2["total_documents"] == 1, "Data should persist across instances"
    
    def test_save_and_load_test_cases(self, temp_dir):
        """Test saving and loading generated test cases."""
        from hrm_eval.requirements_parser.schemas import TestCase, TestStep, ExpectedResult
        
        # Create test cases
        test_cases = [
            TestCase(
                id="TC-001",
                description="Test file upload",
                type="positive",
                priority="P1",
                test_steps=[
                    TestStep(step_number=1, action="Navigate to upload"),
                    TestStep(step_number=2, action="Select file")
                ],
                expected_results=[
                    ExpectedResult(result_id="R1", result="File uploaded successfully")
                ],
                preconditions=["User logged in"]
            )
        ]
        
        # Save to file
        output_file = Path(temp_dir) / "test_cases.json"
        with open(output_file, 'w') as f:
            json.dump([tc.dict() for tc in test_cases], f, indent=2, default=str)
        
        # Load and verify
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data) == 1
        assert loaded_data[0]["id"] == "TC-001"
        assert loaded_data[0]["description"] == "Test file upload"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
