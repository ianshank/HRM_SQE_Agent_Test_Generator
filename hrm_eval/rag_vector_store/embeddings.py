"""
Embedding generation for test cases and requirements.

Uses sentence-transformers for generating embeddings.
"""

from typing import List, Union, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name (default: lightweight, fast)
                       - all-MiniLM-L6-v2: 384 dim, fast, good for short texts
                       - all-mpnet-base-v2: 768 dim, slower, better quality
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(
            f"Embedding model loaded: {model_name} "
            f"(dimension: {self.embedding_dim})"
        )
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Embeddings as list or list of lists
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        if not texts:
            logger.warning("No texts provided for encoding")
            return [] if not is_single else [0.0] * self.embedding_dim
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            
            embeddings_list = embeddings.tolist()
            
            return embeddings_list[0] if is_single else embeddings_list
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise
    
    def encode_test_case(self, test_case: Dict[str, Any]) -> List[float]:
        """
        Encode test case to embedding.
        
        Args:
            test_case: Test case dictionary
            
        Returns:
            Embedding vector
        """
        parts = []
        
        if 'description' in test_case and test_case['description']:
            parts.append(test_case['description'])
        
        if 'type' in test_case and test_case['type']:
            parts.append(f"Type: {test_case['type']}")
        
        if 'priority' in test_case and test_case['priority']:
            parts.append(f"Priority: {test_case['priority']}")
        
        if 'labels' in test_case and test_case['labels']:
            labels_str = ' '.join(test_case['labels'])
            parts.append(f"Labels: {labels_str}")
        
        if 'preconditions' in test_case and test_case['preconditions']:
            precond_str = ' '.join(test_case['preconditions'][:2])
            parts.append(f"Preconditions: {precond_str}")
        
        text = ' | '.join(parts) if parts else 'Test case'
        
        return self.encode(text)
    
    def encode_requirement(self, requirement: Dict[str, Any]) -> List[float]:
        """
        Encode requirement to embedding.
        
        Args:
            requirement: Requirement dictionary
            
        Returns:
            Embedding vector
        """
        parts = []
        
        if 'summary' in requirement and requirement['summary']:
            parts.append(requirement['summary'])
        
        if 'description' in requirement and requirement['description']:
            parts.append(requirement['description'])
        
        if 'acceptance_criteria' in requirement:
            criteria = requirement['acceptance_criteria']
            if isinstance(criteria, list) and criteria:
                criteria_str = ' '.join([
                    c.get('criteria', '') if isinstance(c, dict) else str(c)
                    for c in criteria[:3]
                ])
                parts.append(f"Criteria: {criteria_str}")
        
        if 'tech_stack' in requirement and requirement['tech_stack']:
            tech_str = ' '.join(requirement['tech_stack'][:5])
            parts.append(f"Tech: {tech_str}")
        
        text = ' | '.join(parts) if parts else 'Requirement'
        
        return self.encode(text)
    
    def encode_batch(
        self,
        items: List[Dict[str, Any]],
        item_type: str = "test_case",
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Encode batch of items efficiently.
        
        Args:
            items: List of test cases or requirements
            item_type: "test_case" or "requirement"
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
        """
        if item_type == "test_case":
            encode_func = self.encode_test_case
        elif item_type == "requirement":
            encode_func = self.encode_requirement
        else:
            raise ValueError(f"Unknown item_type: {item_type}")
        
        texts = []
        for item in items:
            try:
                if item_type == "test_case":
                    parts = []
                    if 'description' in item:
                        parts.append(item['description'])
                    if 'type' in item:
                        parts.append(f"Type: {item['type']}")
                    text = ' | '.join(parts) if parts else 'Test case'
                else:
                    parts = []
                    if 'summary' in item:
                        parts.append(item['summary'])
                    if 'description' in item:
                        parts.append(item['description'])
                    text = ' | '.join(parts) if parts else 'Requirement'
                
                texts.append(text)
            except Exception as e:
                logger.warning(f"Failed to encode item: {e}")
                texts.append("")
        
        return self.encode(texts, batch_size=batch_size)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim
