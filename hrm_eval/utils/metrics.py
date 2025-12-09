"""
Prometheus metrics module for HRM SQE Agent Test Generator.

Provides standardized metrics for:
- Request latency and throughput
- Model inference timing
- RAG retrieval performance
- Error rates and status

NO HARDCODED VALUES - all labels and configurations are parameterized.
"""

import time
from functools import wraps
from typing import Callable, Optional, Any, Dict
from contextlib import contextmanager
import logging

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
        multiprocess,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def inc(self, *args, **kwargs): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def observe(self, *args, **kwargs): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def observe(self, *args, **kwargs): pass
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Definitions
# =============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    "hrm_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "hrm_api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

REQUEST_IN_PROGRESS = Gauge(
    "hrm_api_requests_in_progress",
    "Number of API requests in progress",
    ["method", "endpoint"]
)

# Generation metrics
GENERATION_COUNT = Counter(
    "hrm_generation_total",
    "Total number of test generation requests",
    ["mode", "status"]
)

GENERATION_LATENCY = Histogram(
    "hrm_generation_latency_seconds",
    "Test generation latency in seconds",
    ["mode"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
)

TEST_CASES_GENERATED = Counter(
    "hrm_test_cases_generated_total",
    "Total number of test cases generated",
    ["mode", "type"]
)

# Model inference metrics
MODEL_INFERENCE_LATENCY = Histogram(
    "hrm_model_inference_latency_seconds",
    "HRM model inference latency in seconds",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

MODEL_BATCH_SIZE = Histogram(
    "hrm_model_batch_size",
    "Model inference batch sizes",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

# RAG metrics
RAG_QUERY_COUNT = Counter(
    "hrm_rag_queries_total",
    "Total number of RAG queries",
    ["operation", "status"]
)

RAG_QUERY_LATENCY = Histogram(
    "hrm_rag_query_latency_seconds",
    "RAG query latency in seconds",
    ["operation"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

RAG_RESULTS_COUNT = Histogram(
    "hrm_rag_results_count",
    "Number of results returned by RAG queries",
    buckets=[0, 1, 2, 5, 10, 20, 50, 100]
)

RAG_SIMILARITY_SCORE = Histogram(
    "hrm_rag_similarity_score",
    "Similarity scores from RAG retrieval",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

RAG_INDEX_SIZE = Gauge(
    "hrm_rag_index_size",
    "Number of documents in RAG index"
)

# Error metrics
ERROR_COUNT = Counter(
    "hrm_errors_total",
    "Total number of errors",
    ["error_type", "component"]
)

# Resource metrics
MEMORY_USAGE = Gauge(
    "hrm_memory_usage_bytes",
    "Memory usage in bytes",
    ["type"]
)

GPU_MEMORY_USAGE = Gauge(
    "hrm_gpu_memory_usage_bytes",
    "GPU memory usage in bytes",
    ["device"]
)

# Application info
APP_INFO = Info(
    "hrm_app",
    "Application information"
)


# =============================================================================
# Metric Collection Functions
# =============================================================================

def record_request(
    method: str,
    endpoint: str,
    status_code: int,
    latency: float
) -> None:
    """
    Record API request metrics.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        status_code: HTTP response status code
        latency: Request latency in seconds
    """
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).inc()

    REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint
    ).observe(latency)


def record_generation(
    mode: str,
    status: str,
    latency: float,
    test_count: int,
    test_types: Optional[Dict[str, int]] = None
) -> None:
    """
    Record test generation metrics.

    Args:
        mode: Generation mode (hrm_only, sqe_only, hybrid)
        status: Generation status (success, error)
        latency: Generation latency in seconds
        test_count: Number of tests generated
        test_types: Optional dict of test types and counts
    """
    GENERATION_COUNT.labels(mode=mode, status=status).inc()
    GENERATION_LATENCY.labels(mode=mode).observe(latency)

    if test_types:
        for test_type, count in test_types.items():
            TEST_CASES_GENERATED.labels(
                mode=mode,
                type=test_type
            ).inc(count)
    else:
        TEST_CASES_GENERATED.labels(
            mode=mode,
            type="unknown"
        ).inc(test_count)


def record_model_inference(operation: str, latency: float, batch_size: int = 1) -> None:
    """
    Record model inference metrics.

    Args:
        operation: Type of inference operation
        latency: Inference latency in seconds
        batch_size: Batch size for the operation
    """
    MODEL_INFERENCE_LATENCY.labels(operation=operation).observe(latency)
    MODEL_BATCH_SIZE.observe(batch_size)


def record_rag_query(
    operation: str,
    status: str,
    latency: float,
    results_count: int = 0,
    avg_similarity: float = 0.0
) -> None:
    """
    Record RAG query metrics.

    Args:
        operation: Type of RAG operation (retrieve, index, search)
        status: Query status (success, error)
        latency: Query latency in seconds
        results_count: Number of results returned
        avg_similarity: Average similarity score
    """
    RAG_QUERY_COUNT.labels(operation=operation, status=status).inc()
    RAG_QUERY_LATENCY.labels(operation=operation).observe(latency)
    RAG_RESULTS_COUNT.observe(results_count)

    if avg_similarity > 0:
        RAG_SIMILARITY_SCORE.observe(avg_similarity)


def record_error(error_type: str, component: str) -> None:
    """
    Record error metrics.

    Args:
        error_type: Type of error (e.g., ValidationError, TimeoutError)
        component: Component where error occurred
    """
    ERROR_COUNT.labels(error_type=error_type, component=component).inc()


def update_rag_index_size(size: int) -> None:
    """Update RAG index size gauge."""
    RAG_INDEX_SIZE.set(size)


def update_memory_metrics() -> None:
    """Update memory usage metrics."""
    try:
        import psutil
        process = psutil.Process()
        MEMORY_USAGE.labels(type="rss").set(process.memory_info().rss)
        MEMORY_USAGE.labels(type="vms").set(process.memory_info().vms)
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                GPU_MEMORY_USAGE.labels(device=f"cuda:{i}").set(allocated)
    except ImportError:
        pass


def set_app_info(version: str, environment: str, model_checkpoint: str) -> None:
    """
    Set application info metrics.

    Args:
        version: Application version
        environment: Deployment environment
        model_checkpoint: Model checkpoint being used
    """
    APP_INFO.info({
        "version": version,
        "environment": environment,
        "model_checkpoint": model_checkpoint,
    })


# =============================================================================
# Decorators for Automatic Metric Collection
# =============================================================================

def track_request(endpoint: str):
    """
    Decorator to track API request metrics.

    Args:
        endpoint: Endpoint name for labeling

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            method = "POST"  # Default for API routes
            REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
            start_time = time.time()
            status_code = 200

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                record_error(type(e).__name__, endpoint)
                raise
            finally:
                latency = time.time() - start_time
                REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
                record_request(method, endpoint, status_code, latency)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            method = "POST"
            REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
            start_time = time.time()
            status_code = 200

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                record_error(type(e).__name__, endpoint)
                raise
            finally:
                latency = time.time() - start_time
                REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
                record_request(method, endpoint, status_code, latency)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def track_generation(mode: str):
    """
    Decorator to track generation metrics.

    Args:
        mode: Generation mode for labeling

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            test_count = 0

            try:
                result = func(*args, **kwargs)
                if isinstance(result, dict):
                    test_count = len(result.get("test_cases", []))
                return result
            except Exception as e:
                status = "error"
                record_error(type(e).__name__, "generation")
                raise
            finally:
                latency = time.time() - start_time
                record_generation(mode, status, latency, test_count)

        return wrapper
    return decorator


def track_model_inference(operation: str):
    """
    Decorator to track model inference metrics.

    Args:
        operation: Operation name for labeling

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start_time
                record_model_inference(operation, latency)

        return wrapper
    return decorator


@contextmanager
def track_rag_operation(operation: str):
    """
    Context manager to track RAG operation metrics.

    Args:
        operation: Operation name for labeling

    Yields:
        Dict to store results count and similarity
    """
    start_time = time.time()
    context = {"results_count": 0, "avg_similarity": 0.0}
    status = "success"

    try:
        yield context
    except Exception as e:
        status = "error"
        record_error(type(e).__name__, f"rag_{operation}")
        raise
    finally:
        latency = time.time() - start_time
        record_rag_query(
            operation,
            status,
            latency,
            context.get("results_count", 0),
            context.get("avg_similarity", 0.0)
        )


# =============================================================================
# Metrics Export
# =============================================================================

def get_metrics() -> bytes:
    """
    Get current metrics in Prometheus format.

    Returns:
        Metrics in Prometheus exposition format
    """
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not available\n"

    # Update memory metrics before export
    update_memory_metrics()

    return generate_latest(REGISTRY)


def get_metrics_content_type() -> str:
    """Get content type for metrics response."""
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    return CONTENT_TYPE_LATEST


# =============================================================================
# Health Check Metrics
# =============================================================================

def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of key metrics for health checks.

    Returns:
        Dict with metric summaries
    """
    # This would typically query the metrics registry
    # For now, return a basic structure
    return {
        "metrics_available": PROMETHEUS_AVAILABLE,
        "status": "healthy" if PROMETHEUS_AVAILABLE else "degraded",
    }
