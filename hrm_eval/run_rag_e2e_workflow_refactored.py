"""
End-to-End RAG-Integrated Test Generation Workflow (Refactored)

Complete pipeline using new core modules:
- ModelManager for model loading
- WorkflowOrchestrator for setup and output
- TestGenerationPipeline for modular generation
- SystemConfig for all configuration
- DebugManager for profiling

BEFORE: 642 lines with hard-coded values and duplication
AFTER: ~150 lines using reusable components
"""

import logging
from pathlib import Path
from typing import Dict, Any

from hrm_eval.core import ModelManager, WorkflowOrchestrator, TestGenerationPipeline
from hrm_eval.utils import load_system_config, setup_logging, DebugManager
from hrm_eval.requirements_parser import RequirementParser
from hrm_eval.requirements_parser.schemas import Epic, UserStory

logger = logging.getLogger(__name__)


def create_media_fulfillment_epic() -> Epic:
    """Create the media fulfillment requirements epic."""
    return Epic(
        epic_id="MEDIA_FULFILLMENT_001",
        title="Media Fulfillment System",
        user_stories=[
            UserStory(
                id="US-001",
                summary="Content Ingestion",
                description="As an external producer, I want to upload completed media files and metadata",
                acceptance_criteria=[
                    {"criteria": "Must accept batch uploads of media files"},
                    {"criteria": "Metadata input forms must enforce required fields"},
                    {"criteria": "Upon submission, confirmation email is sent"},
                ]
            ),
            UserStory(
                id="US-002",
                summary="Metadata Validation and Enrichment",
                description="As a metadata specialist, I want to review and enrich submitted metadata",
                acceptance_criteria=[
                    {"criteria": "Automatic checks for required metadata fields"},
                    {"criteria": "UI for editors to add or correct metadata"},
                    {"criteria": "Traceable change log for metadata updates"},
                ]
            ),
            UserStory(
                id="US-003",
                summary="Content Quality Control",
                description="As a QC operator, I want to perform technical and editorial quality checks",
                acceptance_criteria=[
                    {"criteria": "Checklist for audio/video quality, rights compliance"},
                    {"criteria": "Option to reject, accept, or request re-upload"},
                    {"criteria": "QC results visible to production stakeholders"},
                ]
            ),
            UserStory(
                id="US-004",
                summary="Packaging and Rights Management",
                description="As a packaging manager, I want to package approved content",
                acceptance_criteria=[
                    {"criteria": "Select delivery profiles (OTT, broadcast, VOD)"},
                    {"criteria": "Populate rights windows and restrictions"},
                    {"criteria": "Trigger packaging and encryption jobs"},
                ]
            ),
            UserStory(
                id="US-005",
                summary="Delivery and Confirmation",
                description="As a fulfillment coordinator, I want to track delivery status",
                acceptance_criteria=[
                    {"criteria": "Display delivery progress and error alerts"},
                    {"criteria": "Automated notifications when content is delivered"},
                    {"criteria": "Generate and store delivery confirmation receipts"},
                ]
            ),
        ]
    )


def run_rag_e2e_workflow():
    """
    Execute complete RAG-enhanced test generation workflow.
    
    Uses new core modules for clean, maintainable code:
    - SystemConfig: All configuration values
    - ModelManager: Model loading with caching
    - WorkflowOrchestrator: RAG setup and output handling
    - TestGenerationPipeline: End-to-end generation
    - DebugManager: Performance profiling
    """
    # ==================== Configuration ====================
    config = load_system_config()
    setup_logging(config.logging.level, config.logging.format)
    
    logger.info("=" * 80)
    logger.info("RAG-Enhanced Test Generation Workflow (Refactored)")
    logger.info("=" * 80)
    
    # ==================== Debug Setup ====================
    debug = DebugManager(config)
    
    with debug.profile_section("total_workflow"):
        
        # ==================== Model Loading ====================
        with debug.profile_section("model_loading"):
            logger.info("\n📦 Loading HRM Model...")
            model_manager = ModelManager(config)
            
            model_info = model_manager.load_model(
                checkpoint_name=config.model.default_checkpoint,
                validate=True,
            )
            
            logger.info(f"✓ Model loaded from: {model_info.checkpoint_path.name}")
            logger.info(f"✓ Checkpoint step: {model_info.checkpoint_step}")
            logger.info(f"✓ Device: {model_info.device}")
        
        # ==================== Workflow Orchestration ====================
        with debug.profile_section("workflow_setup"):
            logger.info("\n🔧 Setting up workflow components...")
            orchestrator = WorkflowOrchestrator(config)
            
            # Create output directory
            output_dir = orchestrator.create_output_directory("rag_e2e_workflow")
            logger.info(f"✓ Output directory: {output_dir}")
            
            # Setup RAG components
            rag_components = orchestrator.setup_rag_components()
            logger.info("✓ RAG components initialized")
            logger.info(f"  - Vector store: {rag_components['vector_store'].backend}")
            logger.info(f"  - Embeddings: {rag_components['embeddings'].model_name}")
        
        # ==================== Requirements ====================
        with debug.profile_section("requirements_parsing"):
            logger.info("\n📝 Loading requirements...")
            epic = create_media_fulfillment_epic()
            
            logger.info(f"✓ Epic: {epic.title}")
            logger.info(f"✓ User stories: {len(epic.user_stories)}")
            logger.info(f"✓ Total acceptance criteria: {sum(len(us.acceptance_criteria) for us in epic.user_stories)}")
        
        # ==================== RAG Indexing ====================
        with debug.profile_section("rag_indexing"):
            logger.info("\n🔍 Indexing existing test cases...")
            
            # Load and index existing tests
            existing_tests_path = Path("hrm_eval/generated_tests/media_fulfillment_20251008_161418/test_cases.json")
            if existing_tests_path.exists():
                num_indexed = orchestrator.index_test_cases(
                    test_cases_path=existing_tests_path,
                    rag_components=rag_components,
                )
                logger.info(f"✓ Indexed {num_indexed} test cases for RAG retrieval")
            else:
                logger.warning("⚠ No existing tests found for indexing")
        
        # ==================== Test Generation Pipeline ====================
        with debug.profile_section("test_generation"):
            logger.info("\n🎯 Generating test cases...")
            
            pipeline = TestGenerationPipeline(
                model_manager=model_manager,
                config=config,
                rag_retriever=rag_components.get('retriever'),
            )
            
            # Run end-to-end pipeline
            results = pipeline.run_end_to_end(
                epic=epic,
                use_rag=True,
            )
            
            logger.info(f"✓ Generated {len(results['test_cases'])} test cases")
            logger.info(f"✓ Validation: {results['validation']['valid_count']}/{results['validation']['total_count']} valid")
            logger.info(f"✓ RAG examples used: {results['rag_stats']['total_retrieved']}")
        
        # ==================== Save Results ====================
        with debug.profile_section("save_results"):
            logger.info("\n💾 Saving results...")
            
            orchestrator.save_workflow_results(
                results=results,
                output_dir=output_dir,
                formats=config.output.report_formats,
            )
            
            logger.info(f"✓ Results saved to: {output_dir}")
        
        # ==================== Debug Checkpoint ====================
        if config.debug.enabled:
            with debug.debug_checkpoint("workflow_complete"):
                debug.dump_intermediate_state(
                    state={
                        "epic": epic.title,
                        "test_cases_generated": len(results['test_cases']),
                        "validation_rate": results['validation']['valid_count'] / results['validation']['total_count'],
                        "output_dir": str(output_dir),
                    },
                    stage="workflow_completion",
                    save_to_file=True,
                )
    
    # ==================== Performance Report ====================
    logger.info("\n" + "=" * 80)
    logger.info("Performance Summary")
    logger.info("=" * 80)
    
    perf_report = debug.get_performance_report()
    
    if "sections" in perf_report:
        for section_name, metrics in perf_report["sections"].items():
            logger.info(f"{section_name}:")
            logger.info(f"  Time: {metrics['elapsed_seconds']:.2f}s")
            if 'memory_delta_mb' in metrics:
                logger.info(f"  Memory: {metrics['memory_delta_mb']:+.1f}MB")
    
    logger.info(f"\nTotal workflow time: {perf_report.get('total_time', 0):.2f}s")
    
    # Save debug report
    if config.debug.enabled:
        debug.save_report(filename="workflow_performance.json")
        logger.info(f"✓ Debug report saved: {debug.output_dir}/workflow_performance.json")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Workflow complete!")
    logger.info("=" * 80)
    
    return results, output_dir


if __name__ == "__main__":
    try:
        results, output_dir = run_rag_e2e_workflow()
        print(f"\n✅ Success! Results saved to: {output_dir}")
        print(f"   Generated {len(results['test_cases'])} test cases")
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}", exc_info=True)
        raise

