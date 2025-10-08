"""
Vector store implementation with multiple backend support.

Provides unified interface for ChromaDB and Pinecone vector databases.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStoreBackend(ABC):
    """Abstract base for vector store backends."""
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
    ):
        """
        Add documents with embeddings to store.
        
        Args:
            documents: List of document dictionaries
            embeddings: List of embedding vectors
            ids: Optional list of document IDs
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar documents with metadata and similarity scores
        """
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]):
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        pass


class ChromaDBBackend(VectorStoreBackend):
    """ChromaDB implementation (local, easy setup)."""
    
    def __init__(self, persist_directory: str = "vector_store_db", collection_name: str = "test_cases_requirements"):
        """
        Initialize ChromaDB backend.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
        
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Test cases and requirements knowledge base"}
        )
        
        logger.info(f"ChromaDB initialized at {persist_directory}, collection: {collection_name}")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
    ):
        """Add documents to ChromaDB."""
        if not documents:
            logger.warning("No documents to add")
            return
        
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) must have same length"
            )
        
        if ids is None:
            ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(documents)]
        
        if len(ids) != len(documents):
            raise ValueError(
                f"IDs ({len(ids)}) and documents ({len(documents)}) must have same length"
            )
        
        metadatas = []
        documents_text = []
        
        for doc in documents:
            metadata = {k: v for k, v in doc.items() if k != 'id' and isinstance(v, (str, int, float, bool))}
            metadatas.append(metadata)
            
            text = doc.get('description', '') or doc.get('summary', '') or str(doc)
            documents_text.append(text)
        
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents_text,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}", exc_info=True)
            raise
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in ChromaDB."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
            )
            
            similar_docs = []
            
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    doc = {
                        'id': doc_id,
                        'document': results['documents'][0][i] if results['documents'] else '',
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'similarity': 1 - results['distances'][0][i] if results['distances'] else 0.0,
                    }
                    similar_docs.append(doc)
            
            logger.debug(f"Found {len(similar_docs)} similar documents")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    def delete_documents(self, ids: List[str]):
        """Delete documents by IDs from ChromaDB."""
        if not ids:
            logger.warning("No IDs provided for deletion")
            return
        
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}", exc_info=True)
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "backend": "chromadb",
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}", exc_info=True)
            return {"error": str(e)}


class PineconeBackend(VectorStoreBackend):
    """Pinecone implementation (cloud, scalable)."""
    
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str = "test-cases",
        dimension: int = 384,
    ):
        """
        Initialize Pinecone backend.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the index
            dimension: Embedding dimension
        """
        try:
            import pinecone
        except ImportError:
            raise ImportError(
                "Pinecone not installed. Install with: pip install pinecone-client"
            )
        
        pinecone.init(api_key=api_key, environment=environment)
        
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            logger.info(f"Created Pinecone index: {index_name}")
        
        self.index = pinecone.Index(index_name)
        self.index_name = index_name
        
        logger.info(f"Pinecone initialized, index: {index_name}")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
    ):
        """Add documents to Pinecone."""
        if not documents:
            return
        
        if ids is None:
            ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(documents)]
        
        vectors = []
        for doc_id, embedding, doc in zip(ids, embeddings, documents):
            metadata = {k: v for k, v in doc.items() if k != 'id'}
            vectors.append((doc_id, embedding, metadata))
        
        try:
            self.index.upsert(vectors=vectors)
            logger.info(f"Added {len(vectors)} documents to Pinecone")
        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}", exc_info=True)
            raise
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search Pinecone index."""
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True,
            )
            
            similar_docs = []
            for match in results['matches']:
                doc = {
                    'id': match['id'],
                    'metadata': match.get('metadata', {}),
                    'score': match['score'],
                    'similarity': match['score'],
                }
                similar_docs.append(doc)
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}", exc_info=True)
            return []
    
    def delete_documents(self, ids: List[str]):
        """Delete documents from Pinecone."""
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from Pinecone")
        except Exception as e:
            logger.error(f"Failed to delete from Pinecone: {e}", exc_info=True)
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_documents": stats.get('total_vector_count', 0),
                "index_name": self.index_name,
                "backend": "pinecone",
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}", exc_info=True)
            return {"error": str(e)}


class VectorStore:
    """Unified vector store interface."""
    
    def __init__(self, backend: str = "chromadb", **config):
        """
        Initialize vector store with specified backend.
        
        Args:
            backend: Backend type ("chromadb" or "pinecone")
            **config: Backend-specific configuration
        """
        if backend == "chromadb":
            self.backend = ChromaDBBackend(**config)
        elif backend == "pinecone":
            self.backend = PineconeBackend(**config)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'chromadb' or 'pinecone'")
        
        self.backend_type = backend
        logger.info(f"VectorStore initialized with {backend} backend")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
    ):
        """Add documents with embeddings."""
        return self.backend.add_documents(documents, embeddings, ids)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        return self.backend.search(query_embedding, top_k, filter_dict)
    
    def delete_documents(self, ids: List[str]):
        """Delete documents by IDs."""
        return self.backend.delete_documents(ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.backend.get_collection_stats()
