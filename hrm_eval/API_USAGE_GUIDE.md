# RAG + SQE API Usage Guide

## Overview

The Requirements to Test Cases API now includes RAG (Retrieval-Augmented Generation) and SQE (Software Quality Engineering) agent capabilities for enhanced test generation.

## Base URL

```
http://localhost:8000
```

## API Endpoints

### 1. Initialize RAG + SQE Components

**POST** `/api/v1/initialize-rag`

Initialize RAG vector store, embeddings, and SQE agent components.

```bash
curl -X POST "http://localhost:8000/api/v1/initialize-rag?backend=chromadb&enable_sqe=true&llm_provider=openai"
```

**Parameters:**
- `backend` (string): Vector store backend - `chromadb` or `pinecone` (default: `chromadb`)
- `enable_sqe` (boolean): Enable SQE agent (default: `true`)
- `llm_provider` (string): LLM provider - `openai` or `anthropic` (default: `openai`)

**Response:**
```json
{
  "status": "success",
  "message": "RAG + SQE components initialized successfully",
  "components": {
    "vector_store": "chromadb",
    "embedding_model": "all-MiniLM-L6-v2",
    "rag_retriever": true,
    "sqe_agent": true,
    "hybrid_generator": true,
    "workflow_manager": true
  }
}
```

---

### 2. Generate Test Cases with RAG + SQE

**POST** `/api/v1/generate-tests-rag`

Generate test cases using hybrid approach (HRM + RAG + SQE).

```bash
curl -X POST "http://localhost:8000/api/v1/generate-tests-rag" \
  -H "Content-Type: application/json" \
  -d '{
    "epic": {
      "epic_id": "EPIC-001",
      "title": "User Authentication",
      "user_stories": [
        {
          "id": "US-001",
          "summary": "User login with credentials",
          "description": "As a user, I want to login with email and password",
          "acceptance_criteria": [
            {"criteria": "Valid credentials allow login"},
            {"criteria": "Invalid credentials show error"}
          ],
          "tech_stack": ["FastAPI", "PostgreSQL"]
        }
      ],
      "tech_stack": ["FastAPI", "PostgreSQL", "Redis"],
      "architecture": "Microservice Architecture"
    },
    "options": {
      "use_rag": true,
      "use_sqe": true,
      "generation_mode": "hybrid",
      "merge_strategy": "weighted",
      "top_k_similar": 5,
      "min_similarity": 0.7,
      "hrm_weight": 0.6,
      "sqe_weight": 0.4,
      "auto_index": true
    }
  }'
```

**Response:**
```json
{
  "test_cases": [
    {
      "id": "TC-001",
      "description": "Verify user login with valid credentials",
      "type": "positive",
      "priority": "P1",
      "labels": ["authentication", "api"],
      "test_steps": [
        {"step": 1, "action": "Navigate to login page"},
        {"step": 2, "action": "Enter valid email"},
        {"step": 3, "action": "Enter valid password"},
        {"step": 4, "action": "Click login button"}
      ],
      "expected_results": [
        {"result": "User is successfully logged in"},
        {"result": "Dashboard is displayed"}
      ]
    }
  ],
  "metadata": {
    "generation_mode": "hybrid",
    "hrm_generated": 10,
    "sqe_generated": 5,
    "merged_count": 12,
    "rag_context_used": true,
    "rag_similar_count": 5,
    "sqe_enhanced": true,
    "merge_strategy": "weighted",
    "generation_time_seconds": 3.45
  },
  "coverage_analysis": {
    "coverage_percentage": 85.5,
    "positive_tests": 6,
    "negative_tests": 4,
    "edge_tests": 2
  },
  "recommendations": [],
  "status": "success"
}
```

---

### 3. Index Test Cases

**POST** `/api/v1/index-test-cases`

Index generated test cases into the vector store for future RAG retrieval.

```bash
curl -X POST "http://localhost:8000/api/v1/index-test-cases" \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "id": "TC-001",
        "description": "Test user login",
        "type": "positive",
        "priority": "P1",
        "labels": ["login", "authentication"]
      }
    ],
    "source": "manual_upload",
    "batch_size": 100
  }'
```

**Response:**
```json
{
  "status": "success",
  "stats": {
    "total_indexed": 1,
    "batch_count": 1,
    "indexing_time_seconds": 0.25,
    "errors": 0
  },
  "message": "Successfully indexed 1 test cases"
}
```

---

### 4. Search Similar Test Cases

**POST** `/api/v1/search-similar`

Search for similar test cases using RAG retrieval.

```bash
curl -X POST "http://localhost:8000/api/v1/search-similar" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "user authentication with password",
    "top_k": 5,
    "min_similarity": 0.7
  }'
```

**Response:**
```json
{
  "results": [
    {
      "test_case": {
        "id": "TC-042",
        "description": "Verify user login with valid password",
        "type": "positive",
        "priority": "P1"
      },
      "similarity_score": 0.92,
      "source": "previous_project"
    },
    {
      "test_case": {
        "id": "TC-087",
        "description": "Test authentication failure with incorrect password",
        "type": "negative",
        "priority": "P2"
      },
      "similarity_score": 0.85,
      "source": "historical"
    }
  ],
  "query_info": {
    "top_k": 5,
    "min_similarity": 0.7,
    "query": "user authentication with password"
  },
  "search_time_seconds": 0.15
}
```

---

### 5. Execute Complete Workflow

**POST** `/api/v1/execute-workflow`

Execute complete workflow: validate → generate → analyze → index.

```bash
curl -X POST "http://localhost:8000/api/v1/execute-workflow" \
  -H "Content-Type: application/json" \
  -d '{
    "epic": {
      "epic_id": "EPIC-001",
      "title": "User Authentication",
      "user_stories": [...]
    },
    "workflow_type": "full",
    "rag_options": {
      "use_rag": true,
      "use_sqe": true,
      "generation_mode": "hybrid"
    }
  }'
```

**Workflow Types:**
- `full`: Complete workflow (validate → generate → analyze → index)
- `generate_only`: Generation only
- `validate_only`: Validation only

**Response:**
```json
{
  "workflow_type": "full",
  "steps": [
    {
      "step": "validation",
      "status": "complete",
      "result": {"is_valid": true},
      "duration_seconds": 0.1
    },
    {
      "step": "generation",
      "status": "complete",
      "result": {"test_count": 12},
      "duration_seconds": 3.2
    },
    {
      "step": "indexing",
      "status": "complete",
      "result": {"indexed_count": 12},
      "duration_seconds": 0.3
    }
  ],
  "test_cases": [...],
  "status": "complete",
  "total_time_seconds": 3.75
}
```

---

### 6. Extended Health Check

**GET** `/api/v1/health-extended`

Get extended health status including RAG components.

```bash
curl "http://localhost:8000/api/v1/health-extended"
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_checkpoint": "/path/to/checkpoint",
  "device": "cuda",
  "uptime_seconds": 3600.5,
  "version": "1.0.0",
  "rag_health": {
    "vector_store_connected": true,
    "vector_store_backend": "chromadb",
    "indexed_test_count": 1250,
    "sqe_agent_initialized": true,
    "hybrid_generator_available": true,
    "rag_retriever_available": true
  }
}
```

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# 1. Initialize RAG + SQE
init_response = requests.post(
    f"{BASE_URL}/initialize-rag",
    params={
        "backend": "chromadb",
        "enable_sqe": True,
        "llm_provider": "openai"
    }
)
print(f"Initialization: {init_response.json()}")

# 2. Generate test cases with RAG
epic_data = {
    "epic": {
        "epic_id": "EPIC-001",
        "title": "User Authentication",
        "user_stories": [
            {
                "id": "US-001",
                "summary": "User login",
                "description": "As a user, I want to login",
                "acceptance_criteria": [
                    {"criteria": "Valid credentials allow login"},
                    {"criteria": "Invalid credentials show error"}
                ],
                "tech_stack": ["FastAPI", "PostgreSQL"]
            }
        ],
        "tech_stack": ["FastAPI", "PostgreSQL"],
        "architecture": "Microservice"
    },
    "options": {
        "use_rag": True,
        "use_sqe": True,
        "generation_mode": "hybrid"
    }
}

gen_response = requests.post(
    f"{BASE_URL}/generate-tests-rag",
    json=epic_data
)
result = gen_response.json()

print(f"Generated {result['metadata']['merged_count']} test cases")
print(f"HRM: {result['metadata']['hrm_generated']}, SQE: {result['metadata']['sqe_generated']}")
print(f"RAG context used: {result['metadata']['rag_context_used']}")

# 3. Search similar tests
search_response = requests.post(
    f"{BASE_URL}/search-similar",
    json={
        "query": "user authentication",
        "top_k": 5,
        "min_similarity": 0.7
    }
)
similar = search_response.json()
print(f"Found {len(similar['results'])} similar test cases")

# 4. Execute complete workflow
workflow_response = requests.post(
    f"{BASE_URL}/execute-workflow",
    json={
        "epic": epic_data["epic"],
        "workflow_type": "full"
    }
)
workflow = workflow_response.json()
print(f"Workflow status: {workflow['status']}")
print(f"Steps: {len(workflow['steps'])}")
```

---

## Generation Modes

### HRM Only
- Uses only HRM model for generation
- Fast, deterministic
- No RAG context or SQE orchestration

```json
{
  "generation_mode": "hrm_only"
}
```

### SQE Only
- Uses SQE agent with LangGraph workflow
- Intelligent orchestration
- RAG context if available

```json
{
  "generation_mode": "sqe_only"
}
```

### Hybrid (Recommended)
- Combines HRM model + SQE agent + RAG
- Best quality and coverage
- Configurable merge strategies

```json
{
  "generation_mode": "hybrid",
  "merge_strategy": "weighted",
  "hrm_weight": 0.6,
  "sqe_weight": 0.4
}
```

---

## Merge Strategies

### Weighted
- Combines results with configurable weights
- Default: HRM 60%, SQE 40%
- Balances speed and quality

### Union
- Combines all unique test cases from both
- Maximum coverage
- May include duplicates

### Intersection
- Only test cases present in both HRM and SQE
- High confidence results
- Lower total count

---

## Environment Variables

```bash
# OpenAI API Key (required for SQE agent with OpenAI)
export OPENAI_API_KEY="sk-..."

# Anthropic API Key (required for SQE agent with Anthropic)
export ANTHROPIC_API_KEY="sk-ant-..."

# Pinecone API Key (required for Pinecone backend)
export PINECONE_API_KEY="..."
export PINECONE_ENVIRONMENT="us-west1-gcp"
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Successful operation
- `400 Bad Request`: Invalid input
- `503 Service Unavailable`: Components not initialized
- `500 Internal Server Error`: Server error

Example error response:
```json
{
  "detail": "Hybrid generator not initialized. Call /initialize-rag endpoint first."
}
```

---

## Performance Tips

1. **Initialize once**: Call `/initialize-rag` once at startup
2. **Use hybrid mode**: Best balance of quality and speed
3. **Index regularly**: Keep vector store updated with new tests
4. **Adjust weights**: Tune `hrm_weight` and `sqe_weight` for your use case
5. **Monitor health**: Check `/health-extended` regularly

---

## Interactive Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Next Steps

1. Start the API server:
   ```bash
   cd hrm_eval
   uvicorn api_service.main:app --reload --port 8000
   ```

2. Initialize RAG components:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/initialize-rag"
   ```

3. Generate test cases:
   ```bash
   # Use the example requests above
   ```

4. Monitor health:
   ```bash
   curl "http://localhost:8000/api/v1/health-extended"
   ```
