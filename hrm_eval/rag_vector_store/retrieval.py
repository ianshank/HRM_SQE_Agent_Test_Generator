"""
RAG retrieval logic with context building.

Retrieves similar test cases from vector store and builds context
for test generation.
"""

from typing import List, Dict, Any, Optional
import logging

from .vector_store import VectorStore
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG-based retrieval for test case generation."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
    ):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        logger.info("RAGRetriever initialized")
    
    def retrieve_similar_test_cases(
        self,
        requirement: Dict[str, Any],
        top_k: int = 5,
        min_similarity: float = 0.7,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar test cases for a requirement.
        
        Args:
            requirement: Requirement dictionary
            top_k: Number of results to retrieve
            min_similarity: Minimum similarity threshold
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar test cases with metadata and similarity scores
        """
        logger.debug(f"Retrieving similar test cases for requirement")
        
        try:
            query_embedding = self.embedding_generator.encode_requirement(requirement)
            
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                filter_dict=filter_dict,
            )
            
            filtered_results = [
                r for r in results
                if r.get('similarity', 0) >= min_similarity
            ][:top_k]
            
            logger.info(
                f"Retrieved {len(filtered_results)} similar test cases "
                f"(from {len(results)} total results)"
            )
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return []
    
    def retrieve_by_text(
        self,
        text: str,
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar test cases by text query.
        
        Args:
            text: Query text
            top_k: Number of results
            min_similarity: Minimum similarity
            
        Returns:
            List of similar test cases
        """
        try:
            query_embedding = self.embedding_generator.encode(text)
            
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
            )
            
            filtered_results = [
                r for r in results
                if r.get('similarity', 0) >= min_similarity
            ][:top_k]
            
            logger.info(f"Retrieved {len(filtered_results)} test cases for text query")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Text retrieval failed: {e}", exc_info=True)
            return []
    
    def build_context(
        self,
        requirement: Dict[str, Any],
        retrieved_tests: List[Dict[str, Any]],
        include_metadata: bool = True,
    ) -> str:
        """
        Build context string from retrieved test cases.
        
        Args:
            requirement: Current requirement
            retrieved_tests: Retrieved similar test cases
            include_metadata: Include similarity scores and metadata
            
        Returns:
            Context string for model/agent
        """
        if not retrieved_tests:
            logger.debug("No retrieved tests to build context from")
            return ""
        
        context_parts = [
            "# Historical Test Case Examples",
            "",
            f"Based on similar requirements, here are {len(retrieved_tests)} relevant test case examples:",
            ""
        ]
        
        for idx, test in enumerate(retrieved_tests, 1):
            metadata = test.get('metadata', {})
            
            context_parts.append(f"## Example {idx}")
            
            if include_metadata:
                similarity = test.get('similarity', 0)
                context_parts.append(f"**Similarity Score:** {similarity:.2f}")
            
            test_type = metadata.get('type', 'N/A')
            context_parts.append(f"**Type:** {test_type}")
            
            description = metadata.get('description', test.get('document', 'N/A'))
            context_parts.append(f"**Description:** {description}")
            
            priority = metadata.get('priority', 'N/A')
            context_parts.append(f"**Priority:** {priority}")
            
            if 'labels' in metadata:
                labels = metadata['labels']
                if isinstance(labels, list):
                    context_parts.append(f"**Labels:** {', '.join(labels)}")
                elif isinstance(labels, str):
                    context_parts.append(f"**Labels:** {labels}")
            
            if 'preconditions' in metadata and metadata['preconditions']:
                context_parts.append("**Preconditions:**")
                preconditions = metadata['preconditions']
                if isinstance(preconditions, list):
                    for precond in preconditions[:3]:
                        context_parts.append(f"  - {precond}")
                else:
                    context_parts.append(f"  - {preconditions}")
            
            if 'test_steps' in metadata and metadata['test_steps']:
                context_parts.append("**Test Steps:**")
                steps = metadata['test_steps']
                if isinstance(steps, list):
                    for step in steps[:5]:
                        if isinstance(step, dict):
                            step_num = step.get('step', '?')
                            action = step.get('action', '')
                            context_parts.append(f"  {step_num}. {action}")
                        else:
                            context_parts.append(f"  - {step}")
            
            context_parts.append("")
        
        context_parts.extend([
            "---",
            "",
            "Use these examples as inspiration for generating similar test cases,",
            "but adapt them to the specific requirements provided.",
            ""
        ])
        
        context_string = "\n".join(context_parts)
        
        logger.debug(f"Built context with {len(retrieved_tests)} examples")
        
        return context_string
    
    def build_compact_context(
        self,
        retrieved_tests: List[Dict[str, Any]],
    ) -> str:
        """
        Build a compact context string (for token-limited scenarios).
        
        Args:
            retrieved_tests: Retrieved test cases
            
        Returns:
            Compact context string
        """
        if not retrieved_tests:
            return ""
        
        context_parts = ["Historical examples:"]
        
        for idx, test in enumerate(retrieved_tests[:3], 1):
            metadata = test.get('metadata', {})
            desc = metadata.get('description', '')[:100]
            test_type = metadata.get('type', 'N/A')
            priority = metadata.get('priority', 'N/A')
            
            context_parts.append(
                f"{idx}. [{test_type}/{priority}] {desc}..."
            )
        
        return " ".join(context_parts)
    
    def get_retrieval_stats(
        self,
        requirement: Dict[str, Any],
        retrieved_tests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get statistics about retrieval results.
        
        Args:
            requirement: Requirement used for retrieval
            retrieved_tests: Retrieved test cases
            
        Returns:
            Statistics dictionary
        """
        if not retrieved_tests:
            return {
                "num_retrieved": 0,
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
            }
        
        similarities = [t.get('similarity', 0) for t in retrieved_tests]
        
        types_count = {}
        priorities_count = {}
        
        for test in retrieved_tests:
            metadata = test.get('metadata', {})
            
            test_type = metadata.get('type', 'unknown')
            types_count[test_type] = types_count.get(test_type, 0) + 1
            
            priority = metadata.get('priority', 'unknown')
            priorities_count[priority] = priorities_count.get(priority, 0) + 1
        
        stats = {
            "num_retrieved": len(retrieved_tests),
            "avg_similarity": sum(similarities) / len(similarities),
            "max_similarity": max(similarities),
            "min_similarity": min(similarities),
            "types_distribution": types_count,
            "priorities_distribution": priorities_count,
        }
        
        return stats
