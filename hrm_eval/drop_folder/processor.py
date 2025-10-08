"""
Drop Folder Processor.

Core component that processes requirement files and generates test cases.
"""

import torch
import yaml
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

from hrm_eval.models import HRMModel
from hrm_eval.models.hrm_model import HRMConfig
from hrm_eval.utils.config_utils import load_config
from hrm_eval.utils.checkpoint_utils import load_checkpoint
from hrm_eval.requirements_parser.nl_parser import parse_natural_language_requirements
from hrm_eval.requirements_parser.schemas import Epic, TestCase
from hrm_eval.test_generator.generator import TestCaseGenerator
from hrm_eval.rag_vector_store.vector_store import VectorStore
from hrm_eval.rag_vector_store.embeddings import EmbeddingGenerator
from hrm_eval.rag_vector_store.retrieval import RAGRetriever
from hrm_eval.utils.security import PathValidator, SecurityAuditor

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a requirements file."""
    
    success: bool
    input_file: str
    output_dir: Optional[str]
    test_count: int
    processing_time: float
    error_message: Optional[str] = None
    rag_enabled: bool = False
    retrieved_examples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DropFolderProcessor:
    """
    Processes requirement files from drop folder.
    
    Handles:
    - File validation and security checks
    - Natural language parsing
    - RAG-enhanced test generation
    - Output formatting and archiving
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize processor with configuration.
        
        Args:
            config_path: Path to drop_folder_config.yaml (optional)
        """
        logger.info("Initializing DropFolderProcessor")
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.base_path = Path(self.config['base_path'])
        
        # Initialize paths
        self.input_dir = self.base_path / self.config['input_dir']
        self.output_dir = self.base_path / self.config['output_dir']
        self.archive_dir = self.base_path / self.config['archive_dir']
        self.processing_dir = self.base_path / self.config['processing_dir']
        self.errors_dir = self.base_path / self.config['errors_dir']
        
        # Security components
        self.path_validator = PathValidator(base_dir=str(self.base_path))
        self.security_auditor = SecurityAuditor()
        
        # Processing state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.test_generator = None
        self.rag_retriever = None
        
        # Rate limiting
        self.processing_times: List[float] = []
        self.max_per_minute = self.config.get('rate_limit_per_minute', 10)
        
        logger.info(f"Processor initialized with base path: {self.base_path}")
        logger.info(f"Device: {self.device}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load drop folder configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "drop_folder_config.yaml"
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        config = full_config.get('drop_folder', {})
        logger.debug(f"Loaded configuration from {config_path}")
        return config
    
    def initialize_models(self):
        """
        Initialize HRM model and RAG components.
        
        Lazy initialization - only loads when first needed.
        """
        if self.model is not None:
            logger.debug("Models already initialized")
            return
        
        logger.info("Initializing models (this may take a moment)...")
        
        try:
            # Load model configuration
            base_path = Path(__file__).parent.parent
            config = load_config(
                model_config_path=base_path / "configs" / "model_config.yaml",
                eval_config_path=base_path / "configs" / "eval_config.yaml"
            )
            hrm_config = HRMConfig.from_yaml_config(config)
            
            # Determine checkpoint path
            if self.config.get('use_fine_tuned', True):
                checkpoint_path = base_path / self.config['model_checkpoint']
                if not checkpoint_path.exists():
                    logger.warning(f"Fine-tuned checkpoint not found: {checkpoint_path}")
                    checkpoint_path = base_path.parent / self.config['fallback_checkpoint']
            else:
                checkpoint_path = base_path.parent / self.config['fallback_checkpoint']
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
            logger.info(f"Loading model from: {checkpoint_path}")
            
            # Load model
            self.model = HRMModel(hrm_config)
            checkpoint = load_checkpoint(str(checkpoint_path))
            
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Clean state dict
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for prefix in ["model.inner.", "model.", "module."]:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                if new_k == "embedding_weight":
                    new_k = "weight"
                new_state_dict[new_k] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize test generator
            self.test_generator = TestCaseGenerator(
                model=self.model,
                device=self.device,
                config=config
            )
            
            # Initialize RAG components if enabled
            if self.config.get('use_rag', True):
                logger.info("Initializing RAG components...")
                vector_store = VectorStore(backend="chromadb", persist_directory="vector_store_db")
                embedding_generator = EmbeddingGenerator()
                self.rag_retriever = RAGRetriever(
                    vector_store=vector_store,
                    embedding_generator=embedding_generator
                )
                logger.info("RAG components initialized")
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}", exc_info=True)
            raise
    
    def process_file(self, filepath: Path) -> ProcessingResult:
        """
        Process a single requirements file.
        
        Args:
            filepath: Path to requirements file
            
        Returns:
            ProcessingResult with status and metadata
        """
        start_time = datetime.now()
        logger.info(f"Processing file: {filepath.name}")
        
        try:
            # Validate file
            self._validate_file(filepath)
            
            # Move to processing directory
            processing_path = self.processing_dir / filepath.name
            shutil.move(str(filepath), str(processing_path))
            logger.debug(f"Moved to processing: {processing_path}")
            
            # Read file content
            with open(processing_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.debug(f"Read {len(content)} characters from file")
            
            # Parse requirements
            epic = parse_natural_language_requirements(content, filepath.name)
            logger.info(f"Parsed epic '{epic.title}' with {len(epic.user_stories)} stories")
            
            # Initialize models if not done
            self.initialize_models()
            
            # Generate test cases
            test_cases, rag_stats = self._generate_tests(epic)
            logger.info(f"Generated {len(test_cases)} test cases")
            
            # Save outputs
            output_path = self._save_outputs(
                filepath.name,
                epic,
                test_cases,
                rag_stats,
                start_time
            )
            
            # Archive processed file
            archive_path = self.archive_dir / filepath.name
            shutil.move(str(processing_path), str(archive_path))
            logger.info(f"Archived to: {archive_path}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Security audit
            self.security_auditor.log_security_event(
                event_type="file_processed",
                description=f"Successfully processed {filepath.name}",
                severity="INFO"
            )
            
            result = ProcessingResult(
                success=True,
                input_file=str(filepath),
                output_dir=str(output_path),
                test_count=len(test_cases),
                processing_time=processing_time,
                rag_enabled=rag_stats.get('rag_enabled', False),
                retrieved_examples=rag_stats.get('retrieved_examples', 0)
            )
            
            logger.info(f"Successfully processed {filepath.name} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}", exc_info=True)
            
            # Move to errors directory
            try:
                if processing_path.exists():
                    error_path = self.errors_dir / filepath.name
                    shutil.move(str(processing_path), str(error_path))
                    
                    # Save error log
                    error_log = error_path.with_suffix(error_path.suffix + '.error.log')
                    with open(error_log, 'w') as f:
                        f.write(f"Error processing {filepath.name}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"Error: {str(e)}\n\n")
                        f.write(traceback.format_exc())
                    
                    logger.info(f"Moved to errors: {error_path}")
            except Exception as move_error:
                logger.error(f"Failed to move error file: {move_error}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=False,
                input_file=str(filepath),
                output_dir=None,
                test_count=0,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _validate_file(self, filepath: Path):
        """Validate file before processing."""
        # Check file exists
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Check file extension
        allowed_extensions = self.config.get('file_extensions', ['.txt', '.md'])
        if filepath.suffix not in allowed_extensions:
            raise ValueError(f"Invalid file extension: {filepath.suffix}")
        
        # Check file size
        max_size_mb = self.config.get('max_file_size_mb', 10)
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)")
        
        # Path validation (security)
        self.path_validator.validate_path(str(filepath))
        
        logger.debug(f"File validation passed: {filepath.name}")
    
    def _generate_tests(self, epic: Epic) -> tuple[List[TestCase], Dict[str, Any]]:
        """Generate test cases for epic."""
        test_cases = []
        rag_stats = {
            'rag_enabled': False,
            'retrieved_examples': 0,
            'total_retrievals': 0
        }
        
        for user_story in epic.user_stories:
            logger.debug(f"Generating tests for story: {user_story.summary[:50]}...")
            
            # Generate with RAG if enabled
            if self.rag_retriever and self.config.get('use_rag', True):
                # Retrieve similar tests
                query = f"{epic.title} | {user_story.summary}"
                similar_tests = self.rag_retriever.retrieve_by_text(
                    text=query,
                    top_k=self.config.get('top_k_similar', 5),
                    min_similarity=self.config.get('min_similarity', 0.5)
                )
                
                rag_stats['rag_enabled'] = True
                rag_stats['retrieved_examples'] += len(similar_tests)
                rag_stats['total_retrievals'] += 1
                
                logger.debug(f"Retrieved {len(similar_tests)} similar tests via RAG")
            
            # Generate tests using HRM model
            epic_context = {
                'epic_id': epic.epic_id,
                'title': epic.title,
                'description': getattr(epic, 'description', ''),
                'business_value': getattr(epic, 'business_value', ''),
                'target_release': getattr(epic, 'target_release', '')
            }
            story_tests = self.test_generator.generate_for_user_story(
                story=user_story,
                epic_context=epic_context
            )
            
            test_cases.extend(story_tests)
        
        return test_cases, rag_stats
    
    def _save_outputs(
        self,
        filename: str,
        epic: Epic,
        test_cases: List[TestCase],
        rag_stats: Dict[str, Any],
        start_time: datetime
    ) -> Path:
        """Save generated outputs to organized directory."""
        # Create timestamped output directory
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        stem = Path(filename).stem
        output_name = f"{timestamp}_{stem}"
        output_path = self.output_dir / output_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Saving outputs to: {output_path}")
        
        # Import formatter
        from .formatter import OutputFormatter
        formatter = OutputFormatter()
        
        # Save JSON if configured
        if self.config.get('save_json', True):
            json_path = output_path / "test_cases.json"
            formatter.save_json(test_cases, json_path)
            logger.debug(f"Saved JSON: {json_path}")
        
        # Save Markdown if configured
        if self.config.get('save_markdown', True):
            md_path = output_path / "test_cases.md"
            formatter.save_markdown(test_cases, epic, md_path)
            logger.debug(f"Saved Markdown: {md_path}")
        
        # Generate report if configured
        if self.config.get('generate_report', True):
            report_path = output_path / "generation_report.md"
            processing_time = (datetime.now() - start_time).total_seconds()
            formatter.save_report(
                epic, test_cases, rag_stats, processing_time, report_path
            )
            logger.debug(f"Saved report: {report_path}")
        
        # Save metadata
        if self.config.get('include_metadata', True):
            metadata_path = output_path / "metadata.json"
            metadata = {
                'input_file': filename,
                'timestamp': start_time.isoformat(),
                'epic_title': epic.title,
                'user_stories_count': len(epic.user_stories),
                'test_cases_count': len(test_cases),
                'processing_time_seconds': processing_time,
                'rag_enabled': rag_stats.get('rag_enabled', False),
                'retrieved_examples': rag_stats.get('retrieved_examples', 0),
                'model_checkpoint': self.config.get('model_checkpoint', 'unknown')
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata: {metadata_path}")
        
        return output_path
    
    def process_all_pending(self) -> List[ProcessingResult]:
        """
        Process all pending files in input directory.
        
        Returns:
            List of ProcessingResult for each file
        """
        logger.info("Processing all pending files")
        
        # Find all pending files
        pending_files = []
        for ext in self.config.get('file_extensions', ['.txt', '.md']):
            pending_files.extend(self.input_dir.glob(f"*{ext}"))
        
        if not pending_files:
            logger.info("No pending files found")
            return []
        
        logger.info(f"Found {len(pending_files)} pending files")
        
        results = []
        for filepath in pending_files:
            result = self.process_file(filepath)
            results.append(result)
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        return results
