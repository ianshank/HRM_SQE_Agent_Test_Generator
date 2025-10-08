"""
FastAPI application for Requirements to Test Cases API.

Provides REST API endpoints for generating test cases from requirements
using the HRM v9 Optimized model.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import torch
import logging
from pathlib import Path
import time
from typing import Optional

from .models import (
    GenerationOptions,
    TestCaseResponse,
    BatchGenerationRequest,
    BatchTestCaseResponse,
    HealthCheckResponse,
    HealthStatus,
    ErrorResponse,
)
from .rag_sqe_models import (
    RAGGenerationRequest,
    RAGGenerationResponse,
    RAGGenerationOptions,
    RAGMetadata,
    IndexTestCasesRequest,
    IndexTestCasesResponse,
    IndexingStats,
    SimilarTestSearchRequest,
    SimilarTestSearchResponse,
    SimilarTest,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    WorkflowStep,
    ExtendedHealthCheckResponse,
    RAGHealthCheck,
)
from .middleware import (
    LoggingMiddleware,
    RateLimitMiddleware,
    AuthenticationMiddleware,
    setup_cors,
)
from ..models import HRMModel
from ..test_generator import TestCaseGenerator, CoverageAnalyzer
from ..requirements_parser import RequirementParser, Epic
from ..requirements_parser.schemas import GenerationMetadata, CoverageReport
from ..utils import load_config, load_checkpoint

# RAG + SQE imports
from ..rag_vector_store import VectorStore, RAGRetriever, EmbeddingGenerator, VectorIndexer
from ..agents import SQEAgent
from ..orchestration import HybridTestGenerator, WorkflowManager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Requirements to Test Cases API",
    version="1.0.0",
    description="Generate comprehensive test cases from structured requirements using HRM model",
    docs_url="/docs",
    redoc_url="/redoc",
)

_generator: Optional[TestCaseGenerator] = None
_model: Optional[HRMModel] = None
_device: Optional[torch.device] = None
_checkpoint_path: Optional[str] = None
_start_time = time.time()

# RAG + SQE components
_vector_store: Optional[VectorStore] = None
_rag_retriever: Optional[RAGRetriever] = None
_embedding_generator: Optional[EmbeddingGenerator] = None
_vector_indexer: Optional[VectorIndexer] = None
_sqe_agent: Optional[SQEAgent] = None
_hybrid_generator: Optional[HybridTestGenerator] = None
_workflow_manager: Optional[WorkflowManager] = None


def get_generator() -> TestCaseGenerator:
    """
    Dependency to get generator instance.
    
    Returns:
        TestCaseGenerator instance
        
    Raises:
        HTTPException: If generator not initialized
    """
    if _generator is None:
        raise HTTPException(
            status_code=503,
            detail="Generator not initialized. Call /initialize endpoint first."
        )
    return _generator


def get_hybrid_generator() -> HybridTestGenerator:
    """
    Dependency to get hybrid generator instance.
    
    Returns:
        HybridTestGenerator instance
        
    Raises:
        HTTPException: If hybrid generator not initialized
    """
    if _hybrid_generator is None:
        raise HTTPException(
            status_code=503,
            detail="Hybrid generator not initialized. Call /initialize-rag endpoint first."
        )
    return _hybrid_generator


def get_rag_retriever() -> RAGRetriever:
    """
    Dependency to get RAG retriever instance.
    
    Returns:
        RAGRetriever instance
        
    Raises:
        HTTPException: If RAG retriever not initialized
    """
    if _rag_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="RAG retriever not initialized. Call /initialize-rag endpoint first."
        )
    return _rag_retriever


def get_vector_indexer() -> VectorIndexer:
    """
    Dependency to get vector indexer instance.
    
    Returns:
        VectorIndexer instance
        
    Raises:
        HTTPException: If vector indexer not initialized
    """
    if _vector_indexer is None:
        raise HTTPException(
            status_code=503,
            detail="Vector indexer not initialized. Call /initialize-rag endpoint first."
        )
    return _vector_indexer


def get_workflow_manager() -> WorkflowManager:
    """
    Dependency to get workflow manager instance.
    
    Returns:
        WorkflowManager instance
        
    Raises:
        HTTPException: If workflow manager not initialized
    """
    if _workflow_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Workflow manager not initialized. Call /initialize-rag endpoint first."
        )
    return _workflow_manager


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Requirements to Test Cases API...")
    
    setup_cors(app)
    
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)
    app.add_middleware(AuthenticationMiddleware, require_auth=False)
    
    logger.info("Middleware configured")
    
    try:
        await initialize_model()
    except Exception as e:
        logger.error(f"Failed to initialize model on startup: {e}")
        logger.info("Model can be initialized later via /initialize endpoint")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Requirements to Test Cases API...")
    
    global _generator, _model
    _generator = None
    _model = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Shutdown complete")


async def initialize_model(checkpoint_path: Optional[str] = None):
    """
    Initialize HRM model and generator.
    
    Args:
        checkpoint_path: Path to model checkpoint (uses default if None)
    """
    global _generator, _model, _device, _checkpoint_path
    
    if checkpoint_path is None:
        base_path = Path(__file__).parent.parent.parent
        checkpoint_path = str(base_path / "checkpoints_hrm_v9_optimized_step_7566")
    
    logger.info(f"Initializing model from {checkpoint_path}")
    
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {_device}")
    
    config = load_config(Path(__file__).parent.parent / "configs" / "model_config.yaml")
    
    _model = HRMModel(
        vocab_size=config["model"]["vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        num_h_layers=config["model"]["num_h_layers"],
        num_l_layers=config["model"]["num_l_layers"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        dropout=config["model"]["dropout"],
        puzzle_vocab_size=config["model"]["puzzle_vocab_size"],
        q_head_actions=config["model"]["q_head_actions"],
    )
    
    checkpoint = load_checkpoint(checkpoint_path)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(_device)
    _model.eval()
    
    logger.info("Model loaded successfully")
    
    gen_config = load_config(
        Path(__file__).parent.parent / "configs" / "test_generation_config.yaml"
    )
    
    _generator = TestCaseGenerator(
        model=_model,
        device=_device,
        config=gen_config,
    )
    
    _checkpoint_path = checkpoint_path
    
    logger.info("Generator initialized successfully")


@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status information
    """
    uptime = time.time() - _start_time
    
    if _model is None or _generator is None:
        status = HealthStatus.DEGRADED
        model_loaded = False
    else:
        status = HealthStatus.HEALTHY
        model_loaded = True
    
    return HealthCheckResponse(
        status=status,
        model_loaded=model_loaded,
        model_checkpoint=_checkpoint_path,
        device=str(_device) if _device else "none",
        uptime_seconds=uptime,
        version="1.0.0",
    )


@app.post("/api/v1/initialize")
async def initialize_endpoint(checkpoint_path: Optional[str] = None):
    """
    Initialize or reinitialize the model.
    
    Args:
        checkpoint_path: Optional path to checkpoint
        
    Returns:
        Initialization status
    """
    try:
        await initialize_model(checkpoint_path)
        
        return {
            "status": "success",
            "message": "Model initialized successfully",
            "checkpoint": _checkpoint_path,
            "device": str(_device),
        }
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize model: {str(e)}"
        )


@app.post("/api/v1/generate-tests", response_model=TestCaseResponse)
async def generate_test_cases(
    epic: Epic,
    options: Optional[GenerationOptions] = None,
    generator: TestCaseGenerator = Depends(get_generator),
):
    """
    Generate test cases from an epic.
    
    Args:
        epic: Epic containing user stories
        options: Optional generation options
        generator: Generator instance (injected)
        
    Returns:
        Generated test cases with metadata
    """
    start_time = time.time()
    
    logger.info(f"Generating test cases for epic '{epic.epic_id}'")
    
    try:
        parser = RequirementParser()
        
        test_contexts = parser.extract_test_contexts(epic)
        
        logger.info(f"Extracted {len(test_contexts)} test contexts")
        
        test_types = options.test_types if options else None
        
        test_cases = generator.generate_test_cases(
            test_contexts=test_contexts,
            test_types=test_types,
        )
        
        generation_time = time.time() - start_time
        
        analyzer = CoverageAnalyzer()
        coverage_report_dict = analyzer.analyze_coverage(test_cases, test_contexts)
        
        coverage_report = CoverageReport(
            positive_tests=coverage_report_dict["positive_tests"],
            negative_tests=coverage_report_dict["negative_tests"],
            edge_tests=coverage_report_dict["edge_tests"],
            acceptance_criteria_covered=len(coverage_report_dict["stories_covered"]),
            total_acceptance_criteria=coverage_report_dict["total_stories"],
            coverage_percentage=coverage_report_dict["coverage_percentage"],
            missing_scenarios=coverage_report_dict["missing_test_types"],
        )
        
        metadata = GenerationMetadata(
            model_checkpoint=_checkpoint_path or "unknown",
            generation_time_seconds=generation_time,
            num_test_cases=len(test_cases),
            coverage_score=coverage_report_dict["coverage_percentage"] / 100.0,
        )
        
        recommendations = analyzer.get_recommendations(coverage_report_dict)
        
        logger.info(
            f"Generated {len(test_cases)} test cases in {generation_time:.2f}s "
            f"(coverage: {coverage_report.coverage_percentage:.1f}%)"
        )
        
        return TestCaseResponse(
            test_cases=test_cases,
            metadata=metadata,
            coverage_report=coverage_report,
            recommendations=recommendations,
        )
        
    except Exception as e:
        logger.error(f"Test case generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Test case generation failed: {str(e)}"
        )


@app.post("/api/v1/batch-generate", response_model=BatchTestCaseResponse)
async def batch_generate(
    request: BatchGenerationRequest,
    generator: TestCaseGenerator = Depends(get_generator),
):
    """
    Generate test cases for multiple epics in batch.
    
    Args:
        request: Batch generation request
        generator: Generator instance (injected)
        
    Returns:
        Batch generation results
    """
    start_time = time.time()
    
    logger.info(f"Batch generating test cases for {len(request.epics)} epics")
    
    results = []
    success_count = 0
    failure_count = 0
    total_test_cases = 0
    
    for epic in request.epics:
        try:
            response = await generate_test_cases(
                epic=epic,
                options=request.options,
                generator=generator,
            )
            
            results.append(response)
            success_count += 1
            total_test_cases += len(response.test_cases)
            
        except Exception as e:
            logger.error(f"Failed to generate for epic '{epic.epic_id}': {e}")
            failure_count += 1
    
    total_time = time.time() - start_time
    
    logger.info(
        f"Batch generation complete: {success_count} succeeded, {failure_count} failed, "
        f"{total_test_cases} total test cases in {total_time:.2f}s"
    )
    
    return BatchTestCaseResponse(
        results=results,
        total_test_cases=total_test_cases,
        total_time_seconds=total_time,
        success_count=success_count,
        failure_count=failure_count,
    )


@app.post("/api/v1/initialize-rag")
async def initialize_rag(
    backend: str = "chromadb",
    enable_sqe: bool = True,
    llm_provider: str = "openai",
):
    """
    Initialize RAG + SQE components.
    
    Args:
        backend: Vector store backend (chromadb or pinecone)
        enable_sqe: Enable SQE agent
        llm_provider: LLM provider (openai or anthropic)
        
    Returns:
        Initialization status
    """
    global _vector_store, _rag_retriever, _embedding_generator, _vector_indexer
    global _sqe_agent, _hybrid_generator, _workflow_manager
    
    try:
        logger.info(f"Initializing RAG components with {backend} backend...")
        
        # Load RAG configuration
        rag_config = load_config(
            Path(__file__).parent.parent / "configs" / "rag_sqe_config.yaml"
        )
        
        # Initialize vector store
        _vector_store = VectorStore(
            backend=backend,
            persist_directory=rag_config["rag"].get("persist_directory", "vector_store_db")
        )
        logger.info(f"Vector store initialized: {backend}")
        
        # Initialize embedding generator
        embedding_model = rag_config["rag"].get("embedding_model", "all-MiniLM-L6-v2")
        _embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        logger.info(f"Embedding generator initialized: {embedding_model}")
        
        # Initialize RAG retriever
        _rag_retriever = RAGRetriever(_vector_store, _embedding_generator)
        logger.info("RAG retriever initialized")
        
        # Initialize vector indexer
        _vector_indexer = VectorIndexer(_vector_store, _embedding_generator)
        logger.info("Vector indexer initialized")
        
        # Initialize SQE agent if enabled
        if enable_sqe:
            try:
                from langchain_openai import ChatOpenAI
                from langchain_anthropic import ChatAnthropic
                
                if llm_provider == "openai":
                    llm = ChatOpenAI(
                        model=rag_config["sqe_agent"].get("model", "gpt-4"),
                        temperature=rag_config["sqe_agent"].get("temperature", 0.7)
                    )
                else:
                    llm = ChatAnthropic(
                        model=rag_config["sqe_agent"].get("model", "claude-3-opus-20240229"),
                        temperature=rag_config["sqe_agent"].get("temperature", 0.7)
                    )
                
                _sqe_agent = SQEAgent(
                    llm=llm,
                    rag_retriever=_rag_retriever,
                    hrm_generator=_generator,
                    enable_rag=rag_config["sqe_agent"].get("enable_rag", True),
                    enable_hrm=rag_config["sqe_agent"].get("enable_hrm", True),
                )
                logger.info(f"SQE agent initialized with {llm_provider}")
                
            except Exception as e:
                logger.warning(f"SQE agent initialization failed: {e}")
                _sqe_agent = None
        
        # Initialize hybrid generator
        if _generator:
            _hybrid_generator = HybridTestGenerator(
                hrm_generator=_generator,
                sqe_agent=_sqe_agent,
                rag_retriever=_rag_retriever,
                mode=rag_config["hybrid_generation"].get("mode", "hybrid"),
                merge_strategy=rag_config["hybrid_generation"].get("merge_strategy", "weighted"),
                hrm_weight=rag_config["hybrid_generation"].get("hrm_weight", 0.6),
                sqe_weight=rag_config["hybrid_generation"].get("sqe_weight", 0.4),
            )
            logger.info("Hybrid generator initialized")
            
            # Initialize workflow manager
            _workflow_manager = WorkflowManager(
                hybrid_generator=_hybrid_generator,
                vector_indexer=_vector_indexer,
                auto_index=rag_config["hybrid_generation"].get("auto_index", True),
            )
            logger.info("Workflow manager initialized")
        
        return {
            "status": "success",
            "message": "RAG + SQE components initialized successfully",
            "components": {
                "vector_store": backend,
                "embedding_model": embedding_model,
                "rag_retriever": _rag_retriever is not None,
                "sqe_agent": _sqe_agent is not None,
                "hybrid_generator": _hybrid_generator is not None,
                "workflow_manager": _workflow_manager is not None,
            }
        }
        
    except Exception as e:
        logger.error(f"RAG initialization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RAG components: {str(e)}"
        )


@app.post("/api/v1/generate-tests-rag", response_model=RAGGenerationResponse)
async def generate_tests_rag(
    request: RAGGenerationRequest,
    hybrid_generator: HybridTestGenerator = Depends(get_hybrid_generator),
):
    """
    Generate test cases with RAG + SQE enhancement.
    
    Args:
        request: RAG generation request
        hybrid_generator: Hybrid generator instance (injected)
        
    Returns:
        RAG-enhanced test generation response
    """
    start_time = time.time()
    
    logger.info(f"RAG generation for epic '{request.epic.epic_id}'")
    
    try:
        options = request.options or RAGGenerationOptions()
        
        result = hybrid_generator.generate(
            requirements=request.epic.dict(),
            context=None,
        )
        
        generation_time = time.time() - start_time
        
        metadata = RAGMetadata(
            generation_mode=result["metadata"]["mode"],
            hrm_generated=result["metadata"].get("hrm_generated", 0),
            sqe_generated=result["metadata"].get("sqe_generated", 0),
            merged_count=len(result["test_cases"]),
            rag_context_used=result["metadata"].get("rag_context_used", False),
            rag_similar_count=result["metadata"].get("rag_similar_count", 0),
            sqe_enhanced=result["metadata"].get("sqe_enhanced", False),
            merge_strategy=result["metadata"].get("merge_strategy", ""),
            generation_time_seconds=generation_time,
        )
        
        logger.info(
            f"RAG generation complete: {metadata.merged_count} test cases "
            f"(HRM: {metadata.hrm_generated}, SQE: {metadata.sqe_generated}) "
            f"in {generation_time:.2f}s"
        )
        
        return RAGGenerationResponse(
            test_cases=result["test_cases"],
            metadata=metadata,
            coverage_analysis=result["metadata"].get("coverage_analysis", {}),
            recommendations=[],
            status="success",
        )
        
    except Exception as e:
        logger.error(f"RAG test generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG test generation failed: {str(e)}"
        )


@app.post("/api/v1/index-test-cases", response_model=IndexTestCasesResponse)
async def index_test_cases(
    request: IndexTestCasesRequest,
    indexer: VectorIndexer = Depends(get_vector_indexer),
):
    """
    Index test cases into vector store.
    
    Args:
        request: Indexing request
        indexer: Vector indexer instance (injected)
        
    Returns:
        Indexing response
    """
    start_time = time.time()
    
    logger.info(f"Indexing {len(request.test_cases)} test cases from {request.source}")
    
    try:
        test_cases_dicts = [tc.dict() for tc in request.test_cases]
        
        indexer.index_test_cases(
            test_cases=test_cases_dicts,
            batch_size=request.batch_size,
        )
        
        indexing_time = time.time() - start_time
        
        stats = IndexingStats(
            total_indexed=len(request.test_cases),
            batch_count=(len(request.test_cases) + request.batch_size - 1) // request.batch_size,
            indexing_time_seconds=indexing_time,
            errors=0,
        )
        
        logger.info(f"Indexing complete: {stats.total_indexed} test cases in {indexing_time:.2f}s")
        
        return IndexTestCasesResponse(
            status="success",
            stats=stats,
            message=f"Successfully indexed {stats.total_indexed} test cases",
        )
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Test case indexing failed: {str(e)}"
        )


@app.post("/api/v1/search-similar", response_model=SimilarTestSearchResponse)
async def search_similar_tests(
    request: SimilarTestSearchRequest,
    retriever: RAGRetriever = Depends(get_rag_retriever),
):
    """
    Search for similar test cases.
    
    Args:
        request: Search request
        retriever: RAG retriever instance (injected)
        
    Returns:
        Similar test cases
    """
    start_time = time.time()
    
    logger.info(f"Searching for similar tests (top_k={request.top_k})")
    
    try:
        if request.requirement:
            similar_tests = retriever.retrieve_similar_test_cases(
                requirement=request.requirement,
                top_k=request.top_k,
                min_similarity=request.min_similarity,
            )
        elif request.query:
            similar_tests = retriever.retrieve_similar_test_cases(
                requirement={"description": request.query},
                top_k=request.top_k,
                min_similarity=request.min_similarity,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'query' or 'requirement' must be provided"
            )
        
        search_time = time.time() - start_time
        
        results = [
            SimilarTest(
                test_case=test,
                similarity_score=test.get("similarity", 0.0),
                source=test.get("source", "unknown"),
            )
            for test in similar_tests
        ]
        
        logger.info(f"Found {len(results)} similar tests in {search_time:.2f}s")
        
        return SimilarTestSearchResponse(
            results=results,
            query_info={
                "top_k": request.top_k,
                "min_similarity": request.min_similarity,
                "query": request.query,
            },
            search_time_seconds=search_time,
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Similar test search failed: {str(e)}"
        )


@app.post("/api/v1/execute-workflow", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    request: WorkflowExecutionRequest,
    workflow_manager: WorkflowManager = Depends(get_workflow_manager),
):
    """
    Execute complete workflow (validate → generate → analyze → index).
    
    Args:
        request: Workflow execution request
        workflow_manager: Workflow manager instance (injected)
        
    Returns:
        Workflow execution response
    """
    start_time = time.time()
    
    logger.info(f"Executing {request.workflow_type} workflow for epic '{request.epic.epic_id}'")
    
    try:
        result = workflow_manager.execute_workflow(
            requirements=request.epic.dict(),
            workflow_type=request.workflow_type,
        )
        
        total_time = time.time() - start_time
        
        steps = [
            WorkflowStep(
                step=step["step"],
                status=step["status"],
                result=step.get("result", {}),
                duration_seconds=0.0,
            )
            for step in result.get("steps", [])
        ]
        
        logger.info(
            f"Workflow complete: {len(result.get('test_cases', []))} test cases "
            f"in {total_time:.2f}s"
        )
        
        return WorkflowExecutionResponse(
            workflow_type=result["workflow_type"],
            steps=steps,
            test_cases=result.get("test_cases", []),
            status=result["status"],
            total_time_seconds=total_time,
        )
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(e)}"
        )


@app.get("/api/v1/health-extended", response_model=ExtendedHealthCheckResponse)
async def health_check_extended():
    """
    Extended health check including RAG components.
    
    Returns:
        Extended health status
    """
    uptime = time.time() - _start_time
    
    if _model is None or _generator is None:
        status = "degraded"
        model_loaded = False
    else:
        status = "healthy"
        model_loaded = True
    
    rag_health = None
    if _vector_store or _rag_retriever or _sqe_agent:
        try:
            indexed_count = 0
            if _vector_store:
                try:
                    indexed_count = _vector_store.get_collection_count()
                except:
                    pass
            
            rag_health = RAGHealthCheck(
                vector_store_connected=_vector_store is not None,
                vector_store_backend="chromadb" if _vector_store else "none",
                indexed_test_count=indexed_count,
                sqe_agent_initialized=_sqe_agent is not None,
                hybrid_generator_available=_hybrid_generator is not None,
                rag_retriever_available=_rag_retriever is not None,
            )
        except Exception as e:
            logger.warning(f"Failed to get RAG health: {e}")
    
    return ExtendedHealthCheckResponse(
        status=status,
        model_loaded=model_loaded,
        model_checkpoint=_checkpoint_path,
        device=str(_device) if _device else "none",
        uptime_seconds=uptime,
        version="1.0.0",
        rag_health=rag_health,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
        }
    )

