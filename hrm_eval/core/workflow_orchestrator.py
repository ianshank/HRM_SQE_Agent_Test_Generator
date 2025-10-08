"""
Workflow Orchestrator Module.

Provides reusable workflow components to eliminate duplication and ensure
consistent setup across different workflows.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, NamedTuple
from datetime import datetime

from ..rag_vector_store.vector_store import VectorStore
from ..rag_vector_store.embeddings import EmbeddingGenerator
from ..rag_vector_store.retrieval import RAGRetriever
from ..utils.unified_config import SystemConfig, create_output_directory

logger = logging.getLogger(__name__)


class RAGComponents(NamedTuple):
    """Container for RAG components."""
    vector_store: VectorStore
    embedding_generator: EmbeddingGenerator
    retriever: RAGRetriever


class PipelineContext(NamedTuple):
    """Context for initialized pipeline."""
    output_dir: Path
    config: SystemConfig
    device: str
    rag_components: Optional[RAGComponents] = None


class WorkflowOrchestrator:
    """
    Orchestrates common workflow patterns.
    
    Handles:
    - Pipeline initialization
    - RAG component setup
    - Output directory creation
    - Result saving
    - Consistent logging and error handling
    
    Example:
        >>> orchestrator = WorkflowOrchestrator(config)
        >>> context = orchestrator.initialize_pipeline("test_generation")
        >>> rag = orchestrator.setup_rag_components()
        >>> orchestrator.save_workflow_results(results, context.output_dir)
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize workflow orchestrator.
        
        Args:
            config: System configuration
        """
        self.config = config
        logger.info("WorkflowOrchestrator initialized")
    
    def initialize_pipeline(
        self,
        workflow_name: str,
        base_dir: Optional[Path] = None,
        enable_rag: Optional[bool] = None,
    ) -> PipelineContext:
        """
        Initialize a workflow pipeline.
        
        Args:
            workflow_name: Name of workflow for output directory
            base_dir: Base directory for outputs (uses config if None)
            enable_rag: Enable RAG components (uses config feature flag if None)
            
        Returns:
            PipelineContext with initialized components
            
        Example:
            >>> orchestrator = WorkflowOrchestrator(config)
            >>> context = orchestrator.initialize_pipeline("test_generation")
            >>> print(f"Output: {context.output_dir}")
        """
        logger.info(f"Initializing pipeline: {workflow_name}")
        
        output_dir = create_output_directory(self.config, workflow_name, base_dir)
        logger.info(f"Output directory: {output_dir}")
        
        device = self._determine_device()
        logger.info(f"Device: {device}")
        
        rag_components = None
        if enable_rag is None:
            enable_rag = self.config.features.enable_rag
        
        if enable_rag:
            logger.info("Setting up RAG components...")
            rag_components = self.setup_rag_components()
        
        context = PipelineContext(
            output_dir=output_dir,
            config=self.config,
            device=device,
            rag_components=rag_components,
        )
        
        logger.info("Pipeline initialization complete")
        return context
    
    def setup_rag_components(
        self,
        vector_store_dir: Optional[Path] = None,
        backend: Optional[str] = None,
    ) -> RAGComponents:
        """
        Setup RAG components (vector store, embeddings, retriever).
        
        Args:
            vector_store_dir: Directory for vector store (uses config if None)
            backend: Vector store backend (uses config if None)
            
        Returns:
            RAGComponents with initialized components
            
        Example:
            >>> orchestrator = WorkflowOrchestrator(config)
            >>> rag = orchestrator.setup_rag_components()
            >>> results = rag.retriever.retrieve_by_text("search query")
        """
        logger.info("Setting up RAG components")
        
        if vector_store_dir is None:
            vector_store_dir = Path.cwd() / self.config.paths.vector_store_dir
        vector_store_dir = Path(vector_store_dir).resolve()
        
        if backend is None:
            backend = self.config.rag.backend
        
        logger.debug(f"Vector store: {backend} at {vector_store_dir}")
        
        vector_store = VectorStore(
            backend=backend,
            persist_directory=str(vector_store_dir),
            collection_name=self.config.rag.collection_name,
        )
        
        embedding_generator = EmbeddingGenerator(
            model_name=self.config.rag.embedding_model,
        )
        
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            top_k=self.config.rag.top_k_retrieval,
            min_similarity=self.config.rag.min_similarity,
        )
        
        logger.info("RAG components initialized successfully")
        
        return RAGComponents(
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            retriever=retriever,
        )
    
    def _determine_device(self) -> str:
        """Determine device based on configuration."""
        import torch
        
        if self.config.device.auto_select:
            if torch.cuda.is_available():
                return f"cuda:{self.config.device.device_id}"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return self.config.device.preferred_device
    
    def save_workflow_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        include_metadata: Optional[bool] = None,
    ):
        """
        Save workflow results in configured formats.
        
        Args:
            results: Results dictionary to save
            output_dir: Output directory
            include_metadata: Include metadata (uses config if None)
            
        Example:
            >>> orchestrator = WorkflowOrchestrator(config)
            >>> results = {"test_cases": [...], "stats": {...}}
            >>> orchestrator.save_workflow_results(results, output_dir)
        """
        import json
        
        if include_metadata is None:
            include_metadata = self.config.output.include_metadata
        
        if include_metadata:
            results = self._add_metadata(results)
        
        for format_name in self.config.output.report_formats:
            if format_name == "json":
                self._save_json(results, output_dir)
            elif format_name == "markdown":
                self._save_markdown(results, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _add_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to results."""
        import copy
        results = copy.deepcopy(results)
        
        if "metadata" not in results:
            results["metadata"] = {}
        
        results["metadata"].update({
            "timestamp": datetime.now().isoformat(),
            "config_version": "1.0",
            "system_config": {
                "rag_enabled": self.config.features.enable_rag,
                "debug_enabled": self.config.debug.enabled,
                "device": self.config.device.preferred_device,
            }
        })
        
        return results
    
    def _save_json(self, results: Dict[str, Any], output_dir: Path):
        """Save results as JSON."""
        import json
        
        json_path = output_dir / self.config.paths.test_cases_filename
        
        with open(json_path, "w") as f:
            json.dump(results, f, indent=self.config.output.indent_spaces, default=str)
        
        logger.debug(f"Saved JSON: {json_path}")
    
    def _save_markdown(self, results: Dict[str, Any], output_dir: Path):
        """Save results as Markdown."""
        md_path = output_dir / self.config.paths.report_filename
        
        with open(md_path, "w") as f:
            f.write(self._generate_markdown_report(results))
        
        logger.debug(f"Saved Markdown: {md_path}")
    
    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report from results."""
        lines = []
        width = self.config.output.formatting_width
        sep = self.config.output.separator_char * width
        
        lines.append("# Workflow Results Report")
        lines.append("")
        lines.append(sep)
        lines.append("")
        
        if "metadata" in results:
            lines.append("## Metadata")
            lines.append("")
            for key, value in results["metadata"].items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
            lines.append(sep)
            lines.append("")
        
        if "statistics" in results:
            lines.append("## Statistics")
            lines.append("")
            for key, value in results["statistics"].items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")
            lines.append(sep)
            lines.append("")
        
        if "test_cases" in results:
            lines.append("## Test Cases")
            lines.append("")
            test_cases = results["test_cases"]
            if isinstance(test_cases, list):
                lines.append(f"Generated {len(test_cases)} test cases")
            lines.append("")
        
        return "\n".join(lines)
    
    def create_checkpoint(
        self,
        stage_name: str,
        state: Dict[str, Any],
        output_dir: Path,
    ):
        """
        Create a checkpoint for debugging.
        
        Args:
            stage_name: Name of the pipeline stage
            state: State dictionary to checkpoint
            output_dir: Output directory
            
        Example:
            >>> orchestrator = WorkflowOrchestrator(config)
            >>> orchestrator.create_checkpoint("parsing", {"epic": epic}, output_dir)
        """
        import json
        
        if stage_name not in self.config.debug.checkpoint_stages:
            return
        
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{stage_name}.json"
        
        with open(checkpoint_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.debug(f"Created checkpoint: {checkpoint_path}")
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate workflow inputs.
        
        Args:
            inputs: Input dictionary to validate
            
        Returns:
            True if valid, False otherwise
            
        Example:
            >>> orchestrator = WorkflowOrchestrator(config)
            >>> if orchestrator.validate_inputs({"epic": epic}):
            ...     # proceed with workflow
        """
        if not self.config.workflow.validate_inputs:
            return True
        
        required_keys = ["requirements", "epic", "input_data"]
        
        for key in required_keys:
            if key in inputs and inputs[key] is not None:
                return True
        
        logger.error(f"Invalid inputs: missing one of {required_keys}")
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        return f"WorkflowOrchestrator(rag_enabled={self.config.features.enable_rag})"


__all__ = ["WorkflowOrchestrator", "RAGComponents", "PipelineContext"]

