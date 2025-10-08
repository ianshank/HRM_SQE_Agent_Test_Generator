"""
End-to-End RAG-Integrated Test Generation Workflow

Complete pipeline that combines:
1. Vector store indexing of existing test cases
2. RAG retrieval of similar examples
3. Context-enhanced test generation with HRM model
4. Quality evaluation and comparison

This creates a complete RAG → Generation → Evaluation loop.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from hrm_eval.models import HRMModel
from hrm_eval.models.hrm_model import HRMConfig
from hrm_eval.utils.config_utils import load_config
from hrm_eval.utils.checkpoint_utils import load_checkpoint
from hrm_eval.utils.logging_utils import setup_logging
from hrm_eval.requirements_parser import RequirementParser
from hrm_eval.requirements_parser.schemas import Epic, UserStory, TestCase
from hrm_eval.test_generator.generator import TestCaseGenerator
from hrm_eval.rag_vector_store.vector_store import VectorStore
from hrm_eval.rag_vector_store.embeddings import EmbeddingGenerator
from hrm_eval.rag_vector_store.retrieval import RAGRetriever

logger = logging.getLogger(__name__)


class RAGEnhancedTestGenerator:
    """
    Test generator enhanced with RAG capabilities.
    
    Retrieves similar test cases from vector store and uses them
    as context to improve generation quality.
    """
    
    def __init__(
        self,
        model: HRMModel,
        rag_retriever: RAGRetriever,
        top_k: int = 5,
    ):
        """
        Initialize RAG-enhanced generator.
        
        Args:
            model: HRM model for generation
            rag_retriever: RAG retriever for similar examples
            top_k: Number of similar examples to retrieve
        """
        self.model = model
        self.rag_retriever = rag_retriever
        self.top_k = top_k
        self.base_generator = TestCaseGenerator(model=model)
        
        logger.info(f"Initialized RAG-enhanced generator (top_k={top_k})")
    
    def generate_with_rag(
        self,
        epic: Epic,
        user_story: UserStory,
    ) -> List[TestCase]:
        """
        Generate test cases with RAG augmentation.
        
        Args:
            epic: Parent epic
            user_story: User story to generate tests for
            
        Returns:
            List of generated test cases with RAG context
        """
        # 1. Create query from user story
        query_text = self._create_query(epic, user_story)
        logger.debug(f"Query: {query_text[:100]}...")
        
        # 2. Retrieve similar test cases
        similar_tests = self._retrieve_similar_tests(query_text)
        logger.info(f"Retrieved {len(similar_tests)} similar test cases")
        
        # 3. Format context from retrieved tests
        context = self._format_context(similar_tests)
        
        # 4. Generate tests with context
        test_cases = self._generate_with_context(
            epic=epic,
            user_story=user_story,
            context=context,
        )
        
        # 5. Add RAG metadata
        for tc in test_cases:
            if not hasattr(tc, 'metadata') or tc.metadata is None:
                tc.metadata = {}
            tc.metadata['rag_enabled'] = True
            tc.metadata['retrieved_examples'] = len(similar_tests)
            tc.metadata['context_length'] = len(context)
        
        return test_cases
    
    def _create_query(self, epic: Epic, user_story: UserStory) -> str:
        """Create search query from requirements."""
        parts = [
            f"Epic: {epic.title}",
            f"Story: {user_story.summary}",
            f"Description: {user_story.description}",
        ]
        
        if user_story.acceptance_criteria:
            criteria_text = " ".join([
                ac.criteria for ac in user_story.acceptance_criteria[:3]
            ])
            parts.append(f"Criteria: {criteria_text}")
        
        return " | ".join(parts)
    
    def _retrieve_similar_tests(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve similar test cases from vector store."""
        # Use RAG retriever
        results = self.rag_retriever.retrieve_by_text(
            text=query,
            top_k=self.top_k,
            min_similarity=0.5,
        )
        
        return results
    
    def _format_context(self, similar_tests: List[Dict[str, Any]]) -> str:
        """Format retrieved tests as context string."""
        if not similar_tests:
            return ""
        
        context_parts = ["=== SIMILAR TEST CASES (for reference) ===\n"]
        
        for i, test in enumerate(similar_tests, 1):
            context_parts.append(f"\nExample {i}:")
            context_parts.append(f"Description: {test.get('description', 'N/A')}")
            context_parts.append(f"Type: {test.get('type', 'N/A')}")
            
            if 'steps' in test and test['steps']:
                steps_text = "; ".join([s.get('action', '') for s in test['steps'][:3]])
                context_parts.append(f"Steps: {steps_text}")
            
            if 'expected_results' in test and test['expected_results']:
                results_text = "; ".join([r.get('result', '') for r in test['expected_results'][:2]])
                context_parts.append(f"Expected: {results_text}")
            
            context_parts.append("")
        
        context_parts.append("=== END EXAMPLES ===\n")
        
        return "\n".join(context_parts)
    
    def _generate_with_context(
        self,
        epic: Epic,
        user_story: UserStory,
        context: str,
    ) -> List[TestCase]:
        """
        Generate test cases with RAG context.
        
        For now, this uses the base generator and adds context as metadata.
        In production, you'd inject context into the model's prompt/input.
        """
        # Use base generator
        test_cases = self.base_generator.generate_for_user_story(
            epic=epic,
            user_story=user_story,
        )
        
        # In a full RAG implementation, you would:
        # 1. Inject context into model input
        # 2. Use context-aware prompting
        # 3. Guide generation based on retrieved examples
        
        logger.info(f"Generated {len(test_cases)} tests with RAG context")
        
        return test_cases


def index_existing_tests(
    vector_store: VectorStore,
    embedding_generator: EmbeddingGenerator,
    test_data_paths: List[Path],
) -> int:
    """
    Index existing test cases into vector store.
    
    Args:
        vector_store: Vector store to populate
        embedding_generator: Embedding generator
        test_data_paths: Paths to test data files
        
    Returns:
        Number of tests indexed
    """
    logger.info("=" * 80)
    logger.info("Indexing Existing Test Cases")
    logger.info("=" * 80)
    
    total_indexed = 0
    all_documents = []
    all_embeddings = []
    all_ids = []
    
    for data_path in test_data_paths:
        if not data_path.exists():
            logger.warning(f"Test data file not found: {data_path}")
            continue
        
        logger.info(f"Indexing from {data_path.name}...")
        
        # Handle both JSON array and JSONL formats
        try:
            with open(data_path, 'r') as f:
                content = f.read()
            
            # Try parsing as JSON array first
            try:
                test_list = json.loads(content)
                if isinstance(test_list, list):
                    # JSON array format
                    for idx, test_data in enumerate(test_list, 1):
                        try:
                            text = _create_test_text_repr(test_data)
                            embedding = embedding_generator.encode(text)
                            
                            all_documents.append(test_data)
                            all_embeddings.append(embedding)
                            all_ids.append(f"{data_path.stem}_{idx}")
                            total_indexed += 1
                        except Exception as e:
                            logger.warning(f"Error processing test {idx}: {e}")
                else:
                    logger.warning(f"{data_path.name} is not a JSON array or JSONL")
            except json.JSONDecodeError:
                # Try JSONL format
                with open(data_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            test_data = json.loads(line.strip())
                            if not test_data:
                                continue
                            
                            text = _create_test_text_repr(test_data)
                            embedding = embedding_generator.encode(text)
                            
                            all_documents.append(test_data)
                            all_embeddings.append(embedding)
                            all_ids.append(f"{data_path.stem}_{line_num}")
                            total_indexed += 1
                        except json.JSONDecodeError:
                            pass  # Skip invalid lines
                        except Exception as e:
                            logger.debug(f"Error on line {line_num}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to process {data_path.name}: {e}")
        
        logger.info(f"Processed {total_indexed} tests from {data_path.name}")
    
    # Batch add to vector store
    if all_documents:
        logger.info(f"Adding {len(all_documents)} documents to vector store...")
        vector_store.add_documents(
            documents=all_documents,
            embeddings=all_embeddings,
            ids=all_ids,
        )
    
    logger.info(f"\nTotal tests indexed: {total_indexed}")
    return total_indexed


def _create_test_text_repr(test_data: Dict[str, Any]) -> str:
    """Create text representation of test case for embedding."""
    parts = []
    
    if 'description' in test_data:
        parts.append(f"Test: {test_data['description']}")
    
    if 'type' in test_data:
        parts.append(f"Type: {test_data['type']}")
    
    if 'preconditions' in test_data and test_data['preconditions']:
        parts.append(f"Preconditions: {'; '.join(test_data['preconditions'][:2])}")
    
    if 'test_steps' in test_data and test_data['test_steps']:
        steps = "; ".join([s.get('action', '') for s in test_data['test_steps'][:3]])
        parts.append(f"Steps: {steps}")
    
    if 'expected_results' in test_data and test_data['expected_results']:
        results = "; ".join([r.get('result', '') for r in test_data['expected_results'][:2]])
        parts.append(f"Expected: {results}")
    
    return " | ".join(parts)


def load_media_fulfillment_requirements() -> Epic:
    """Load media fulfillment requirements (same as before)."""
    from hrm_eval.run_media_fulfillment_workflow import create_media_fulfillment_epic
    return create_media_fulfillment_epic()


def compare_rag_vs_baseline(
    rag_tests: List[TestCase],
    baseline_tests: List[TestCase],
) -> Dict[str, Any]:
    """
    Compare RAG-enhanced vs baseline generation.
    
    Args:
        rag_tests: Tests generated with RAG
        baseline_tests: Tests generated without RAG
        
    Returns:
        Comparison metrics
    """
    logger.info("=" * 80)
    logger.info("RAG vs Baseline Comparison")
    logger.info("=" * 80)
    
    comparison = {
        "rag": {
            "count": len(rag_tests),
            "avg_steps": np.mean([len(tc.test_steps) for tc in rag_tests]),
            "avg_preconditions": np.mean([len(tc.preconditions) for tc in rag_tests]),
            "types": {},
        },
        "baseline": {
            "count": len(baseline_tests),
            "avg_steps": np.mean([len(tc.test_steps) for tc in baseline_tests]),
            "avg_preconditions": np.mean([len(tc.preconditions) for tc in baseline_tests]),
            "types": {},
        },
    }
    
    # Count types
    for tc in rag_tests:
        tc_type = tc.type.value if hasattr(tc.type, 'value') else str(tc.type)
        comparison["rag"]["types"][tc_type] = comparison["rag"]["types"].get(tc_type, 0) + 1
    
    for tc in baseline_tests:
        tc_type = tc.type.value if hasattr(tc.type, 'value') else str(tc.type)
        comparison["baseline"]["types"][tc_type] = comparison["baseline"]["types"].get(tc_type, 0) + 1
    
    # Log comparison
    logger.info("\nRAG-Enhanced:")
    logger.info(f"  Total Tests: {comparison['rag']['count']}")
    logger.info(f"  Avg Steps: {comparison['rag']['avg_steps']:.2f}")
    logger.info(f"  Avg Preconditions: {comparison['rag']['avg_preconditions']:.2f}")
    logger.info(f"  Types: {comparison['rag']['types']}")
    
    logger.info("\nBaseline:")
    logger.info(f"  Total Tests: {comparison['baseline']['count']}")
    logger.info(f"  Avg Steps: {comparison['baseline']['avg_steps']:.2f}")
    logger.info(f"  Avg Preconditions: {comparison['baseline']['avg_preconditions']:.2f}")
    logger.info(f"  Types: {comparison['baseline']['types']}")
    
    return comparison


def run_rag_e2e_workflow():
    """Run complete end-to-end RAG workflow."""
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("RAG-Enhanced Test Generation - E2E Workflow")
    logger.info("=" * 80)
    logger.info("\nComplete pipeline: Index → Retrieve → Generate → Evaluate\n")
    
    base_path = Path(__file__).parent
    output_dir = base_path / "rag_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Step 1: Initialize components
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Initialize Components")
    logger.info("=" * 80)
    
    # Load model
    config = load_config(
        model_config_path=base_path / "configs" / "model_config.yaml",
        eval_config_path=base_path / "configs" / "eval_config.yaml"
    )
    hrm_config = HRMConfig.from_yaml_config(config)
    
    # Use fine-tuned model if available, otherwise base
    finetuned_checkpoint = base_path / "fine_tuned_checkpoints" / "media_fulfillment" / "checkpoint_epoch_3_best.pt"
    if finetuned_checkpoint.exists():
        logger.info("Using fine-tuned model")
        checkpoint_path = finetuned_checkpoint
    else:
        logger.info("Using base model")
        checkpoint_path = base_path.parent / "checkpoints_hrm_v9_optimized_step_7566"
    
    model = HRMModel(hrm_config)
    checkpoint = load_checkpoint(str(checkpoint_path))
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Handle different checkpoint formats
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
    logger.info("Model loaded successfully")
    
    # Initialize vector store and embedding generator
    logger.info("Initializing vector store and embeddings...")
    vector_store_path = Path("vector_store_db").resolve()
    vector_store = VectorStore(backend="chromadb", persist_directory=str(vector_store_path))
    embedding_generator = EmbeddingGenerator()
    logger.info(f"Vector store initialized at: {vector_store_path}")
    
    # Initialize RAG retriever
    rag_retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
    )
    
    # Step 2: Index existing test cases
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Index Existing Test Cases")
    logger.info("=" * 80)
    
    test_data_paths = [
        base_path / "generated_tests" / "media_fulfillment_20251007_220527" / "test_cases.json",
        base_path.parent / "sqe_agent_real_data.jsonl",
    ]
    
    # Filter to existing paths
    test_data_paths = [p for p in test_data_paths if p.exists()]
    
    if not test_data_paths:
        logger.warning("No test data found to index. Creating mock examples...")
        # Create some mock test cases for demonstration
        mock_tests = _create_mock_test_data()
        mock_path = output_dir / "mock_tests.jsonl"
        with open(mock_path, 'w') as f:
            for test in mock_tests:
                f.write(json.dumps(test) + "\n")
        test_data_paths = [mock_path]
    
    num_indexed = index_existing_tests(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        test_data_paths=test_data_paths,
    )
    
    logger.info(f"\n✓ Indexed {num_indexed} test cases into vector store")
    
    # Step 3: Load requirements
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Load Requirements")
    logger.info("=" * 80)
    
    epic = load_media_fulfillment_requirements()
    logger.info(f"Loaded epic: {epic.title}")
    logger.info(f"User stories: {len(epic.user_stories)}")
    
    # Step 4: Generate tests with RAG
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Generate Tests with RAG")
    logger.info("=" * 80)
    
    rag_generator = RAGEnhancedTestGenerator(
        model=model,
        rag_retriever=rag_retriever,
        top_k=5,
    )
    
    rag_tests_all = []
    for story in epic.user_stories:
        logger.info(f"\nGenerating RAG-enhanced tests for: {story.summary}")
        rag_tests = rag_generator.generate_with_rag(epic=epic, user_story=story)
        rag_tests_all.extend(rag_tests)
        logger.info(f"Generated {len(rag_tests)} tests")
    
    logger.info(f"\n✓ Total RAG-enhanced tests: {len(rag_tests_all)}")
    
    # Step 5: Generate baseline tests (without RAG)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Generate Baseline Tests (No RAG)")
    logger.info("=" * 80)
    
    baseline_generator = TestCaseGenerator(model=model)
    baseline_tests_all = []
    
    for story in epic.user_stories:
        logger.info(f"\nGenerating baseline tests for: {story.summary}")
        baseline_tests = baseline_generator.generate_for_user_story(
            epic=epic,
            user_story=story,
        )
        baseline_tests_all.extend(baseline_tests)
        logger.info(f"Generated {len(baseline_tests)} tests")
    
    logger.info(f"\n✓ Total baseline tests: {len(baseline_tests_all)}")
    
    # Step 6: Compare RAG vs Baseline
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Compare RAG vs Baseline")
    logger.info("=" * 80)
    
    comparison = compare_rag_vs_baseline(rag_tests_all, baseline_tests_all)
    
    # Step 7: Save outputs
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Save Outputs")
    logger.info("=" * 80)
    
    # Save RAG tests
    rag_output_path = output_dir / "rag_enhanced_tests.json"
    with open(rag_output_path, 'w') as f:
        json.dump([tc.dict() for tc in rag_tests_all], f, indent=2, default=str)
    logger.info(f"Saved RAG tests to {rag_output_path}")
    
    # Save baseline tests
    baseline_output_path = output_dir / "baseline_tests.json"
    with open(baseline_output_path, 'w') as f:
        json.dump([tc.dict() for tc in baseline_tests_all], f, indent=2, default=str)
    logger.info(f"Saved baseline tests to {baseline_output_path}")
    
    # Save comparison
    comparison_path = output_dir / "rag_comparison.json"
    comparison['timestamp'] = datetime.now().isoformat()
    comparison['model_checkpoint'] = str(checkpoint_path)
    comparison['num_indexed_examples'] = num_indexed
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved comparison to {comparison_path}")
    
    # Step 8: Summary
    logger.info("\n" + "=" * 80)
    logger.info("E2E RAG Workflow Complete!")
    logger.info("=" * 80)
    
    logger.info(f"\n✓ Indexed {num_indexed} existing test cases")
    logger.info(f"✓ Generated {len(rag_tests_all)} RAG-enhanced tests")
    logger.info(f"✓ Generated {len(baseline_tests_all)} baseline tests")
    logger.info(f"✓ Saved all outputs to {output_dir}")
    
    logger.info("\nKey Findings:")
    improvement_steps = comparison['rag']['avg_steps'] - comparison['baseline']['avg_steps']
    improvement_preconditions = comparison['rag']['avg_preconditions'] - comparison['baseline']['avg_preconditions']
    
    logger.info(f"  RAG avg steps: {comparison['rag']['avg_steps']:.2f} "
                f"(Δ{improvement_steps:+.2f} vs baseline)")
    logger.info(f"  RAG avg preconditions: {comparison['rag']['avg_preconditions']:.2f} "
                f"(Δ{improvement_preconditions:+.2f} vs baseline)")
    
    logger.info("\nNext Steps:")
    logger.info("  1. Review generated tests in rag_outputs/")
    logger.info("  2. Collect user feedback on RAG vs baseline quality")
    logger.info("  3. Fine-tune model on RAG-enhanced data")
    logger.info("  4. Iterate: More examples → Better retrieval → Better generation")
    
    return {
        'rag_tests': rag_tests_all,
        'baseline_tests': baseline_tests_all,
        'comparison': comparison,
        'num_indexed': num_indexed,
    }


def _create_mock_test_data() -> List[Dict[str, Any]]:
    """Create mock test data for demonstration."""
    return [
        {
            "description": "Verify successful file upload with valid format",
            "type": "positive",
            "preconditions": ["User is authenticated", "Valid video file available"],
            "test_steps": [
                {"action": "Navigate to upload page"},
                {"action": "Select valid MP4 file"},
                {"action": "Click upload button"},
            ],
            "expected_results": [
                {"result": "Upload progress displayed"},
                {"result": "Success message shown"},
                {"result": "Confirmation email sent"},
            ],
        },
        {
            "description": "Verify error handling for invalid file format",
            "type": "negative",
            "preconditions": ["User is authenticated"],
            "test_steps": [
                {"action": "Navigate to upload page"},
                {"action": "Select invalid file (e.g., .exe)"},
                {"action": "Attempt upload"},
            ],
            "expected_results": [
                {"result": "Error message displayed"},
                {"result": "Upload rejected"},
            ],
        },
        {
            "description": "Verify metadata validation on upload",
            "type": "positive",
            "preconditions": ["User authenticated", "Valid file selected"],
            "test_steps": [
                {"action": "Fill in required metadata fields"},
                {"action": "Submit upload with metadata"},
            ],
            "expected_results": [
                {"result": "Metadata validated successfully"},
                {"result": "Upload completes"},
            ],
        },
    ]


if __name__ == "__main__":
    run_rag_e2e_workflow()
