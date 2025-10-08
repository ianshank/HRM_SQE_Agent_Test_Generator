"""
Sanity tests for RAG+HRM system.

Quick smoke tests to verify the system is functioning correctly.
These tests should run fast and catch major issues.
"""

import pytest
import torch
from pathlib import Path

from hrm_eval.models import HRMModel
from hrm_eval.models.hrm_model import HRMConfig
from hrm_eval.utils.config_utils import load_config
from hrm_eval.rag_vector_store.vector_store import VectorStore
from hrm_eval.rag_vector_store.embeddings import EmbeddingGenerator
from hrm_eval.requirements_parser.schemas import Epic, UserStory, AcceptanceCriteria
from hrm_eval.test_generator.generator import TestCaseGenerator


class TestSystemSanity:
    """Sanity checks for the overall system."""
    
    @pytest.fixture(scope="class")
    def base_path(self):
        """Get base project path."""
        return Path(__file__).parent.parent
    
    def test_project_structure(self, base_path):
        """Verify project structure is intact."""
        required_dirs = [
            "models",
            "rag_vector_store",
            "requirements_parser",
            "test_generator",
            "configs",
            "tests"
        ]
        
        for dir_name in required_dirs:
            dir_path = base_path / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"
            assert dir_path.is_dir(), f"Path is not a directory: {dir_name}"
    
    def test_config_files_exist(self, base_path):
        """Verify configuration files exist."""
        required_configs = [
            "configs/model_config.yaml",
            "configs/eval_config.yaml"
        ]
        
        for config_file in required_configs:
            config_path = base_path / config_file
            assert config_path.exists(), f"Config file missing: {config_file}"
    
    def test_config_loading(self, base_path):
        """Verify configurations load without errors."""
        config = load_config(
            model_config_path=base_path / "configs" / "model_config.yaml",
            eval_config_path=base_path / "configs" / "eval_config.yaml"
        )
        
        assert config is not None
        assert hasattr(config, 'model')
        assert hasattr(config, 'evaluation')
    
    def test_hrm_config_creation(self, base_path):
        """Verify HRM config can be created from YAML."""
        config = load_config(
            model_config_path=base_path / "configs" / "model_config.yaml",
            eval_config_path=base_path / "configs" / "eval_config.yaml"
        )
        
        hrm_config = HRMConfig.from_yaml_config(config)
        
        assert hrm_config is not None
        assert hasattr(hrm_config, 'h_hidden_size')
        assert hasattr(hrm_config, 'l_hidden_size')
        assert hasattr(hrm_config, 'vocab_size')
    
    def test_hrm_model_instantiation(self, base_path):
        """Verify HRM model can be instantiated."""
        config = load_config(
            model_config_path=base_path / "configs" / "model_config.yaml",
            eval_config_path=base_path / "configs" / "eval_config.yaml"
        )
        
        hrm_config = HRMConfig.from_yaml_config(config)
        model = HRMModel(hrm_config)
        
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'eval')


class TestRAGComponentsSanity:
    """Sanity checks for RAG components."""
    
    def test_embedding_generator_instantiation(self):
        """Verify embedding generator can be created."""
        embedding_gen = EmbeddingGenerator()
        
        assert embedding_gen is not None
        assert hasattr(embedding_gen, 'encode')
        assert embedding_gen.embedding_dim > 0
    
    def test_embedding_generation_basic(self):
        """Verify basic embedding generation works."""
        embedding_gen = EmbeddingGenerator()
        
        text = "Test embedding generation"
        embedding = embedding_gen.encode(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedding_gen.embedding_dim
        assert all(isinstance(x, float) for x in embedding)
    
    def test_vector_store_creation(self):
        """Verify vector store can be created."""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            vector_store = VectorStore(
                backend="chromadb",
                persist_directory=temp_dir
            )
            
            assert vector_store is not None
            assert hasattr(vector_store, 'add_documents')
            assert hasattr(vector_store, 'search')
            
            stats = vector_store.get_stats()
            assert 'total_documents' in stats
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestRequirementsProcessingSanity:
    """Sanity checks for requirements processing."""
    
    def test_epic_creation(self):
        """Verify Epic objects can be created."""
        epic = Epic(
            epic_id="EPIC-001",
            title="Test Epic",
            user_stories=[
                UserStory(
                    id="US-001",
                    summary="Test story",
                    description="Test",
                    acceptance_criteria=[]
                )
            ]
        )
        
        assert epic.epic_id == "EPIC-001"
        assert epic.title == "Test Epic"
        assert isinstance(epic.user_stories, list)
    
    def test_user_story_creation(self):
        """Verify UserStory objects can be created."""
        user_story = UserStory(
            id="US-001",
            summary="Test user story",
            description="Test description",
            acceptance_criteria=[]
        )
        
        assert user_story.id == "US-001"
        assert user_story.summary == "Test user story"
        assert isinstance(user_story.acceptance_criteria, list)
    
    def test_acceptance_criteria_creation(self):
        """Verify AcceptanceCriteria objects can be created."""
        ac = AcceptanceCriteria(
            criteria="Test criterion"
        )
        
        assert ac.criteria == "Test criterion"
    
    def test_epic_with_nested_structure(self):
        """Verify Epic can be created with full nested structure."""
        epic = Epic(
            epic_id="EPIC-001",
            title="Test Epic",
            user_stories=[
                UserStory(
                    id="US-001",
                    summary="User story 1",
                    description="Description 1",
                    acceptance_criteria=[
                        AcceptanceCriteria(
                            criteria="Criterion 1"
                        ),
                        AcceptanceCriteria(
                            criteria="Criterion 2"
                        )
                    ]
                )
            ]
        )
        
        assert len(epic.user_stories) == 1
        assert len(epic.user_stories[0].acceptance_criteria) == 2


class TestTestGenerationSanity:
    """Sanity checks for test generation."""
    
    @pytest.fixture(scope="class")
    def device(self):
        """Get device."""
        return torch.device("cpu")  # Use CPU for sanity tests
    
    @pytest.fixture(scope="class")
    def simple_model(self, device):
        """Create a simple model for sanity testing."""
        from hrm_eval.models.hrm_model import HRMConfig
        
        # Minimal config for testing
        config = HRMConfig(
            vocab_size=100,
            embed_dim=32,
            num_puzzles=1,
            h_num_layers=1,
            h_hidden_size=32,
            h_intermediate_size=64,
            h_num_attention_heads=2,
            h_dropout=0.1,
            l_num_layers=1,
            l_hidden_size=32,
            l_intermediate_size=64,
            l_num_attention_heads=2,
            l_dropout=0.1,
            num_actions=2
        )
        
        model = HRMModel(config)
        model.to(device)
        model.eval()
        
        return model
    
    def test_test_generator_instantiation(self, simple_model, device):
        """Verify test generator can be created."""
        from hrm_eval.utils.config_utils import load_config
        
        # Load minimal config
        base_path = Path(__file__).parent.parent
        config = load_config(
            model_config_path=base_path / "configs" / "model_config.yaml",
            eval_config_path=base_path / "configs" / "eval_config.yaml"
        )
        
        generator = TestCaseGenerator(
            model=simple_model,
            device=device,
            config=config
        )
        
        assert generator is not None
        assert hasattr(generator, 'generate_for_user_story')
        assert generator.model is not None
    
    def test_test_case_schema(self):
        """Verify test case schema is valid."""
        from hrm_eval.requirements_parser.schemas import TestCase, TestStep, ExpectedResult
        
        test_case = TestCase(
            id="TC-001",
            description="Test description",
            type="positive",
            priority="P1",
            test_steps=[
                TestStep(step_number=1, action="Step 1")
            ],
            expected_results=[
                ExpectedResult(result_id="R1", result="Result 1")
            ],
            preconditions=["Precondition 1"]
        )
        
        assert test_case.id == "TC-001"
        assert len(test_case.test_steps) == 1
        assert len(test_case.expected_results) == 1
        assert len(test_case.preconditions) == 1


class TestSystemIntegrationSanity:
    """Sanity checks for system integration."""
    
    def test_imports_work(self):
        """Verify all major imports work without errors."""
        try:
            from hrm_eval.models import HRMModel
            from hrm_eval.rag_vector_store.vector_store import VectorStore
            from hrm_eval.rag_vector_store.embeddings import EmbeddingGenerator
            from hrm_eval.rag_vector_store.retrieval import RAGRetriever
            from hrm_eval.requirements_parser.schemas import Epic, UserStory
            from hrm_eval.test_generator.generator import TestCaseGenerator
            from hrm_eval.utils.config_utils import load_config
            
            # If we get here, all imports succeeded
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_torch_available(self):
        """Verify PyTorch is available."""
        import torch
        
        assert torch is not None
        assert hasattr(torch, 'tensor')
        
        # Test basic tensor operation
        tensor = torch.tensor([1, 2, 3])
        assert tensor.sum().item() == 6
    
    def test_device_availability(self):
        """Check device availability."""
        import torch
        
        cpu_available = torch.cuda.is_available() or True  # CPU always available
        assert cpu_available
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert device is not None
    
    def test_chromadb_available(self):
        """Verify ChromaDB is available."""
        try:
            import chromadb
            assert chromadb is not None
        except ImportError:
            pytest.fail("ChromaDB not installed")
    
    def test_sentence_transformers_available(self):
        """Verify sentence-transformers is available."""
        try:
            from sentence_transformers import SentenceTransformer
            assert SentenceTransformer is not None
        except ImportError:
            pytest.fail("sentence-transformers not installed")


class TestWorkflowSanity:
    """Quick end-to-end workflow sanity check."""
    
    def test_minimal_workflow(self):
        """
        Test minimal workflow: Epic → RAG Setup → Generate
        
        This is a smoke test to verify the core pipeline works.
        """
        # 1. Create minimal epic
        epic = Epic(
            epic_id="EPIC-SANITY",
            title="Sanity Test Epic",
            user_stories=[
                UserStory(
                    id="US-SANITY",
                    summary="Sanity test story",
                    description="Minimal user story",
                    acceptance_criteria=[
                        AcceptanceCriteria(
                            criteria="Should work"
                        )
                    ]
                )
            ]
        )
        
        # 2. Verify epic structure
        assert epic is not None
        assert len(epic.user_stories) == 1
        assert len(epic.user_stories[0].acceptance_criteria) == 1
        
        # 3. Test embedding generation
        embedding_gen = EmbeddingGenerator()
        query = f"{epic.title} | {epic.user_stories[0].summary}"
        embedding = embedding_gen.encode(query)
        
        assert len(embedding) == 384
        
        # 4. Test vector store
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            vector_store = VectorStore(
                backend="chromadb",
                persist_directory=temp_dir
            )
            
            # Add a test document
            test_doc = [{"id": "sanity_1", "description": "Sanity test"}]
            test_emb = [embedding]
            vector_store.add_documents(test_doc, test_emb, ["sanity_1"])
            
            stats = vector_store.get_stats()
            assert stats["total_documents"] == 1
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=line"])
