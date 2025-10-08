"""
Test Generation Pipeline Module.

Provides modular pipeline stages for test generation, enabling reusability
and clear separation of concerns.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..requirements_parser import RequirementParser
from ..requirements_parser.schemas import Epic, UserStory, TestCase, TestContext
from ..test_generator.generator import TestCaseGenerator
from ..test_generator.coverage_analyzer import CoverageAnalyzer
from ..rag_vector_store.retrieval import RAGRetriever
from ..utils.unified_config import SystemConfig
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class TestGenerationPipeline:
    """
    Reusable test generation pipeline with modular stages.
    
    Stages:
    1. Requirements parsing
    2. Context generation
    3. RAG retrieval (optional)
    4. Test generation
    5. Validation
    6. Output formatting
    
    Example:
        >>> pipeline = TestGenerationPipeline(model_manager, config)
        >>> epic = pipeline.parse_requirements(input_data)
        >>> contexts = pipeline.generate_contexts(epic)
        >>> tests = pipeline.generate_tests(contexts)
        >>> validated = pipeline.validate_tests(tests)
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        config: SystemConfig,
        rag_retriever: Optional[RAGRetriever] = None,
    ):
        """
        Initialize test generation pipeline.
        
        Args:
            model_manager: Model manager for loading HRM model
            config: System configuration
            rag_retriever: Optional RAG retriever for enhanced generation
        """
        self.model_manager = model_manager
        self.config = config
        self.rag_retriever = rag_retriever
        
        self.requirement_parser = RequirementParser()
        self.coverage_analyzer = CoverageAnalyzer()
        
        self.test_generator = None
        
        logger.info("TestGenerationPipeline initialized")
        logger.info(f"RAG enabled: {rag_retriever is not None}")
    
    def parse_requirements(
        self,
        input_data: Any,
        source_format: str = "epic",
    ) -> Epic:
        """
        Parse requirements into Epic structure.
        
        Args:
            input_data: Input requirements (Epic object, dict, or file path)
            source_format: Format of input data (epic, dict, json, yaml)
            
        Returns:
            Parsed Epic object
            
        Raises:
            ValueError: If parsing fails
            
        Example:
            >>> pipeline = TestGenerationPipeline(model_manager, config)
            >>> epic = pipeline.parse_requirements(epic_dict, "dict")
        """
        logger.info("Parsing requirements...")
        
        if isinstance(input_data, Epic):
            logger.debug("Input is already an Epic object")
            return input_data
        
        if isinstance(input_data, dict):
            logger.debug("Parsing from dictionary")
            return Epic(**input_data)
        
        if isinstance(input_data, (str, Path)):
            logger.debug(f"Parsing from file: {input_data}")
            return self.requirement_parser.parse_from_file(str(input_data))
        
        raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def generate_contexts(
        self,
        epic: Epic,
        filter_stories: Optional[List[str]] = None,
    ) -> List[TestContext]:
        """
        Generate test contexts from epic.
        
        Args:
            epic: Parsed epic with user stories
            filter_stories: Optional list of story IDs to process
            
        Returns:
            List of test contexts ready for generation
            
        Example:
            >>> pipeline = TestGenerationPipeline(model_manager, config)
            >>> contexts = pipeline.generate_contexts(epic)
            >>> print(f"Generated {len(contexts)} contexts")
        """
        logger.info(f"Generating test contexts for epic: {epic.title}")
        
        contexts = []
        user_stories = epic.user_stories
        
        if filter_stories:
            user_stories = [s for s in user_stories if s.id in filter_stories]
            logger.debug(f"Filtered to {len(user_stories)} stories")
        
        for story in user_stories:
            story_contexts = self.requirement_parser.parse_requirement(epic, story)
            contexts.extend(story_contexts)
            logger.debug(f"Generated {len(story_contexts)} contexts for story {story.id}")
        
        logger.info(f"Total contexts generated: {len(contexts)}")
        return contexts
    
    def retrieve_rag_examples(
        self,
        contexts: List[TestContext],
        max_examples_per_context: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve similar examples using RAG.
        
        Args:
            contexts: Test contexts to retrieve examples for
            max_examples_per_context: Max examples per context (uses config if None)
            
        Returns:
            Dictionary mapping context ID to retrieved examples
            
        Example:
            >>> pipeline = TestGenerationPipeline(model_manager, config, rag_retriever)
            >>> rag_examples = pipeline.retrieve_rag_examples(contexts)
        """
        if self.rag_retriever is None:
            logger.warning("RAG retriever not available, skipping retrieval")
            return {}
        
        logger.info(f"Retrieving RAG examples for {len(contexts)} contexts")
        
        if max_examples_per_context is None:
            max_examples_per_context = self.config.rag.top_k_retrieval
        
        rag_examples = {}
        
        for context in contexts:
            query_text = self._create_rag_query(context)
            
            try:
                results = self.rag_retriever.retrieve_by_text(
                    query_text=query_text,
                    top_k=max_examples_per_context,
                )
                
                rag_examples[context.story_id] = results
                logger.debug(f"Retrieved {len(results)} examples for {context.story_id}")
                
            except Exception as e:
                logger.error(f"RAG retrieval failed for {context.story_id}: {e}")
                rag_examples[context.story_id] = []
        
        total_retrieved = sum(len(examples) for examples in rag_examples.values())
        logger.info(f"Total RAG examples retrieved: {total_retrieved}")
        
        return rag_examples
    
    def _create_rag_query(self, context: TestContext) -> str:
        """Create RAG query from test context."""
        from .common_utils import create_rag_query_from_context
        return create_rag_query_from_context(context, self.config)
    
    def generate_tests(
        self,
        contexts: List[TestContext],
        rag_examples: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        checkpoint_name: Optional[str] = None,
    ) -> List[TestCase]:
        """
        Generate test cases from contexts.
        
        Args:
            contexts: Test contexts to generate from
            rag_examples: Optional RAG examples for enhancement
            checkpoint_name: Checkpoint to load (uses config default if None)
            
        Returns:
            List of generated test cases
            
        Example:
            >>> pipeline = TestGenerationPipeline(model_manager, config)
            >>> tests = pipeline.generate_tests(contexts)
            >>> print(f"Generated {len(tests)} test cases")
        """
        logger.info(f"Generating tests from {len(contexts)} contexts")
        
        if self.test_generator is None:
            logger.debug("Initializing test generator...")
            model_info = self._load_model(checkpoint_name)
            
            self.test_generator = TestCaseGenerator(
                model=model_info.model,
                device=model_info.device,
                config=self.config,
            )
        
        test_cases = self.test_generator.generate_test_cases(
            test_contexts=contexts,
        )
        
        logger.info(f"Generated {len(test_cases)} test cases")
        
        return test_cases
    
    def _load_model(self, checkpoint_name: Optional[str] = None):
        """Load HRM model via model manager."""
        if checkpoint_name is None:
            checkpoint_name = self.config.model.default_checkpoint
            logger.debug(f"Using default checkpoint from config: {checkpoint_name}")
        
        return self.model_manager.load_model(checkpoint_name, use_cache=True)
    
    def validate_tests(
        self,
        test_cases: List[TestCase],
        epic: Optional[Epic] = None,
    ) -> Dict[str, Any]:
        """
        Validate generated test cases.
        
        Args:
            test_cases: Test cases to validate
            epic: Optional epic for coverage analysis
            
        Returns:
            Validation results dictionary
            
        Example:
            >>> pipeline = TestGenerationPipeline(model_manager, config)
            >>> results = pipeline.validate_tests(test_cases, epic)
            >>> print(f"Valid: {results['all_valid']}")
        """
        logger.info(f"Validating {len(test_cases)} test cases")
        
        validation_results = {
            "total_tests": len(test_cases),
            "valid_tests": 0,
            "invalid_tests": 0,
            "errors": [],
            "warnings": [],
            "all_valid": True,
        }
        
        for i, test_case in enumerate(test_cases):
            try:
                if not test_case.description:
                    validation_results["errors"].append(f"Test {i}: Missing description")
                    validation_results["invalid_tests"] += 1
                    validation_results["all_valid"] = False
                    continue
                
                if not test_case.test_steps:
                    validation_results["warnings"].append(f"Test {i}: No test steps")
                
                validation_results["valid_tests"] += 1
                
            except Exception as e:
                validation_results["errors"].append(f"Test {i}: {str(e)}")
                validation_results["invalid_tests"] += 1
                validation_results["all_valid"] = False
        
        if epic:
            coverage = self.coverage_analyzer.analyze_coverage(test_cases, epic)
            validation_results["coverage"] = coverage
        
        logger.info(f"Validation complete: {validation_results['valid_tests']}/{validation_results['total_tests']} valid")
        
        return validation_results
    
    def format_output(
        self,
        test_cases: List[TestCase],
        format_type: str = "dict",
    ) -> Any:
        """
        Format test cases for output.
        
        Args:
            test_cases: Test cases to format
            format_type: Output format (dict, json, markdown)
            
        Returns:
            Formatted output
            
        Example:
            >>> pipeline = TestGenerationPipeline(model_manager, config)
            >>> output = pipeline.format_output(test_cases, "dict")
        """
        logger.debug(f"Formatting {len(test_cases)} test cases as {format_type}")
        
        if format_type == "dict":
            return [tc.dict() for tc in test_cases]
        
        elif format_type == "json":
            import json
            return json.dumps([tc.dict() for tc in test_cases], indent=2, default=str)
        
        elif format_type == "markdown":
            return self._format_markdown(test_cases)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _format_markdown(self, test_cases: List[TestCase]) -> str:
        """Format test cases as markdown."""
        lines = ["# Test Cases", ""]
        
        for i, tc in enumerate(test_cases, 1):
            lines.append(f"## Test Case {i}: {tc.description}")
            lines.append("")
            lines.append(f"**ID:** {tc.id}")
            lines.append(f"**Type:** {tc.type}")
            lines.append(f"**Priority:** {tc.priority}")
            lines.append("")
            
            if tc.test_steps:
                lines.append("### Steps")
                for step in tc.test_steps:
                    lines.append(f"{step.step_number}. {step.action}")
                lines.append("")
            
            if tc.expected_results:
                lines.append("### Expected Results")
                for result in tc.expected_results:
                    lines.append(f"- {result.result}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def run_complete_pipeline(
        self,
        input_data: Any,
        checkpoint_name: Optional[str] = None,
        output_format: str = "dict",
    ) -> Dict[str, Any]:
        """
        Run complete pipeline from requirements to tests.
        
        Args:
            input_data: Input requirements
            checkpoint_name: Model checkpoint to use
            output_format: Output format
            
        Returns:
            Dictionary with epic, contexts, tests, and validation results
            
        Example:
            >>> pipeline = TestGenerationPipeline(model_manager, config)
            >>> results = pipeline.run_complete_pipeline(epic_data)
            >>> print(f"Generated {len(results['test_cases'])} tests")
        """
        logger.info("Running complete test generation pipeline")
        
        epic = self.parse_requirements(input_data)
        logger.info(f"Parsed epic: {epic.title}")
        
        contexts = self.generate_contexts(epic)
        logger.info(f"Generated {len(contexts)} contexts")
        
        rag_examples = None
        if self.rag_retriever:
            rag_examples = self.retrieve_rag_examples(contexts)
            logger.info(f"Retrieved RAG examples")
        
        test_cases = self.generate_tests(contexts, rag_examples, checkpoint_name)
        logger.info(f"Generated {len(test_cases)} test cases")
        
        validation = self.validate_tests(test_cases, epic)
        logger.info(f"Validation: {validation['valid_tests']}/{validation['total_tests']} valid")
        
        formatted_tests = self.format_output(test_cases, output_format)
        
        results = {
            "epic": epic,
            "contexts": contexts,
            "test_cases": formatted_tests,
            "validation": validation,
            "statistics": {
                "total_user_stories": len(epic.user_stories),
                "total_contexts": len(contexts),
                "total_tests": len(test_cases),
                "valid_tests": validation["valid_tests"],
                "rag_enabled": rag_examples is not None,
            }
        }
        
        logger.info("Pipeline complete")
        return results
    
    def __repr__(self) -> str:
        """String representation."""
        return f"TestGenerationPipeline(rag_enabled={self.rag_retriever is not None})"


__all__ = ["TestGenerationPipeline"]

