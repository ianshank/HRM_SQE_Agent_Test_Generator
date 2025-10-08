# Quick Start Guide

**Get up and running with RAG + SQE + HRM Test Generation in 5 minutes!**

---

## Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt

# Key dependencies:
# - torch>=2.0.0
# - fastapi>=0.100.0
# - chromadb>=0.4.0
# - sentence-transformers>=2.2.0
# - langchain>=0.1.0
# - langgraph>=0.0.20
# - pydantic>=2.0.0
```

---

##  Quick Start (5 Steps)

### Step 1: Initialize the System

```python
from orchestration.hybrid_generator import HybridTestGenerator
from rag_vector_store.vector_store import VectorStore
from rag_vector_store.embeddings import EmbeddingGenerator
from rag_vector_store.retrieval import RAGRetriever
from agents.sqe_agent import SQEAgent
from test_generator.generator import TestCaseGenerator

# Initialize components
vector_store = VectorStore(backend="chromadb", persist_directory="./vector_db")
embedding_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
rag_retriever = RAGRetriever(vector_store, embedding_gen)
hrm_generator = TestCaseGenerator(checkpoint_path="path/to/checkpoint")
sqe_agent = SQEAgent(llm=your_llm, rag_retriever=rag_retriever, hrm_generator=hrm_generator)

# Create hybrid generator
hybrid_gen = HybridTestGenerator(
    hrm_generator=hrm_generator,
    sqe_agent=sqe_agent,
    rag_retriever=rag_retriever,
    mode="hybrid"  # or "hrm_only", "sqe_only"
)
```

### Step 2: Prepare Your Requirements

```python
from requirements_parser.schemas import Epic, UserStory, AcceptanceCriteria

# Create an Epic
epic = Epic(
    epic_id="EPIC-001",
    title="User Authentication System",
    user_stories=[
        UserStory(
            id="US-001",
            summary="User Login with Email",
            description="As a user, I want to log in with email and password...",
            acceptance_criteria=[
                AcceptanceCriteria(
                    criteria="User can enter valid email and password"
                ),
                AcceptanceCriteria(
                    criteria="System validates credentials against database"
                ),
                AcceptanceCriteria(
                    criteria="User is redirected to dashboard on success"
                ),
            ]
        )
    ]
)
```

### Step 3: Generate Test Cases

```python
# Generate using hybrid approach
result = hybrid_gen.generate(
    requirements=epic.dict(),
    context="Authentication system with JWT tokens"
)

# Access generated test cases
test_cases = result["test_cases"]
metadata = result["metadata"]

print(f"Generated {len(test_cases)} test cases")
print(f"HRM generated: {metadata['hrm_generated']}")
print(f"SQE enhanced: {metadata['sqe_enhanced']}")
print(f"RAG context used: {metadata['rag_context_used']}")
```

### Step 4: Index Generated Tests (for RAG)

```python
from rag_vector_store.indexing import VectorIndexer

# Initialize indexer
indexer = VectorIndexer(vector_store, embedding_gen)

# Index generated test cases
indexer.index_from_generated_results(
    test_cases=test_cases,
    source="auth_system_v1"
)

print("Test cases indexed successfully!")
```

### Step 5: Start the API Server

```bash
# Start FastAPI server
cd hrm_eval
uvicorn api_service.main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
# Open http://localhost:8000/docs
```

---

## 🌐 API Usage

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Initialize RAG + SQE

```bash
curl -X POST http://localhost:8000/api/v1/initialize-rag \
  -H "Content-Type: application/json" \
  -d '{
    "vector_store_backend": "chromadb",
    "embedding_model": "all-MiniLM-L6-v2"
  }'
```

### Generate Test Cases (Hybrid Mode)

```bash
curl -X POST http://localhost:8000/api/v1/generate-tests-rag \
  -H "Content-Type: application/json" \
  -d '{
    "epic": {
      "epic_id": "EPIC-001",
      "title": "User Authentication",
      "user_stories": [
        {
          "id": "US-001",
          "summary": "User Login",
          "description": "As a user, I want to log in...",
          "acceptance_criteria": [
            {"criteria": "User can enter credentials"},
            {"criteria": "System validates credentials"}
          ]
        }
      ]
    },
    "options": {
      "use_rag": true,
      "use_sqe": true,
      "generation_mode": "hybrid",
      "merge_strategy": "weighted",
      "hrm_weight": 0.6,
      "sqe_weight": 0.4
    }
  }'
```

### Search Similar Test Cases

```bash
curl -X POST http://localhost:8000/api/v1/search-similar \
  -H "Content-Type: application/json" \
  -d '{
    "query": "user login authentication",
    "top_k": 5,
    "min_similarity": 0.7
  }'
```

### Execute Full Workflow

```bash
curl -X POST http://localhost:8000/api/v1/execute-workflow \
  -H "Content-Type: application/json" \
  -d '{
    "epic": { ... },
    "workflow_type": "full",
    "auto_index": true
  }'
```

---

##  End-to-End Workflow

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Input Requirements (Epic with User Stories)    │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│ Step 2: Parse & Validate Requirements                   │
│   • RequirementParser.parse_epic()                      │
│   • RequirementValidator.validate()                     │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│ Step 3: RAG Context Retrieval (if enabled)              │
│   • Embed requirement text                              │
│   • Search vector store for similar tests               │
│   • Build context from top-k results                    │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼─────────┐           ┌────────▼───────────┐
│ Step 4a:        │           │ Step 4b:           │
│ HRM Generation  │           │ SQE Orchestration  │
│                 │           │                    │
│ • Tokenize      │           │ • LangGraph flow   │
│ • Model infer   │           │ • Agent reasoning  │
│ • Post-process  │           │ • Tool usage       │
└───────┬─────────┘           └────────┬───────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│ Step 5: Merge & Deduplicate                             │
│   • Weighted merge (default)                            │
│   • Union / Intersection (optional)                     │
│   • Remove duplicates                                   │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│ Step 6: Coverage Analysis                               │
│   • Map tests to acceptance criteria                    │
│   • Calculate coverage percentage                       │
│   • Identify gaps                                       │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│ Step 7: Auto-Index (if enabled)                         │
│   • Embed generated tests                               │
│   • Store in vector DB                                  │
│   • Update knowledge base                               │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│ Step 8: Return Results                                  │
│   • test_cases[]                                        │
│   • metadata                                            │
│   • coverage_analysis                                   │
│   • recommendations                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🎛️ Configuration Options

### Generation Modes

```python
# HRM Only: Fast, model-based
mode="hrm_only"

# SQE Only: Agent-orchestrated, reasoning-focused
mode="sqe_only"

# Hybrid: Best of both worlds (recommended)
mode="hybrid"
```

### Merge Strategies

```python
# Weighted: Combine with configurable weights
merge_strategy="weighted"
hrm_weight=0.6
sqe_weight=0.4

# Union: All tests from both sources
merge_strategy="union"

# Intersection: Only common/overlapping tests
merge_strategy="intersection"
```

### RAG Configuration

```python
rag_top_k=5              # Number of similar tests to retrieve
min_similarity_score=0.7 # Minimum similarity threshold
use_rag=True             # Enable RAG context
```

---

##  Example Output

```json
{
  "test_cases": [
    {
      "id": "TC-AUTH-001",
      "type": "positive",
      "priority": "P1",
      "description": "Verify user can login with valid credentials",
      "preconditions": [
        "User account exists in database",
        "User has valid email and password"
      ],
      "test_steps": [
        {
          "step_number": 1,
          "action": "Navigate to login page",
          "expected_result": "Login form is displayed"
        },
        {
          "step_number": 2,
          "action": "Enter valid email: user@example.com",
          "expected_result": "Email field accepts input"
        },
        {
          "step_number": 3,
          "action": "Enter valid password",
          "expected_result": "Password field accepts input (masked)"
        },
        {
          "step_number": 4,
          "action": "Click 'Login' button",
          "expected_result": "User is authenticated and redirected to dashboard"
        }
      ],
      "expected_results": [
        {
          "result": "User successfully logged in",
          "success_criteria": "Dashboard page loads with user data"
        }
      ],
      "labels": ["auth", "login", "positive"],
      "automation_level": "automated"
    }
  ],
  "metadata": {
    "generation_mode": "hybrid",
    "hrm_generated": 15,
    "sqe_generated": 12,
    "merged_count": 20,
    "rag_context_used": true,
    "rag_similar_count": 5,
    "generation_time_seconds": 2.3
  },
  "coverage_analysis": {
    "total_criteria": 3,
    "covered_criteria": 3,
    "coverage_percentage": 100.0,
    "gaps": []
  }
}
```

---

## 🔧 Troubleshooting

### Issue: Vector Store Connection Failed

```python
# Check ChromaDB installation
pip install chromadb --upgrade

# Verify directory permissions
ls -la vector_store_db/

# Initialize with explicit path
vector_store = VectorStore(
    backend="chromadb",
    persist_directory="/absolute/path/to/vector_db"
)
```

### Issue: HRM Model Not Loading

```python
# Verify checkpoint path
import os
assert os.path.exists("path/to/checkpoint/data.pkl")

# Check PyTorch version
import torch
print(torch.__version__)  # Should be 2.0+

# Load with explicit device
generator = TestCaseGenerator(
    checkpoint_path="path/to/checkpoint",
    device="cpu"  # or "cuda"
)
```

### Issue: SQE Agent Errors

```python
# Verify LangChain installation
pip install langchain langchain-core langgraph --upgrade

# Check LLM configuration
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Initialize with error handling
try:
    agent = SQEAgent(llm=llm)
except Exception as e:
    print(f"SQE initialization failed: {e}")
```

---

## 📚 Additional Resources

### Documentation
- **Full Implementation Guide:** `FINAL_IMPLEMENTATION_SUMMARY.md`
- **API Documentation:** `API_USAGE_GUIDE.md`
- **Test Results:** `REAL_REQUIREMENTS_TEST_REPORT.md`
- **Future Enhancements:** `FUTURE_ENHANCEMENTS.md`

### Code Examples
- **Test Script:** `test_real_requirements.py`
- **Integration Tests:** `tests/test_integration_rag_sqe.py`
- **API Tests:** `tests/test_api_integration.py`

### Configuration
- **RAG Config:** `configs/rag_sqe_config.yaml`
- **Model Config:** `configs/model_config.yaml`
- **Evaluation Config:** `configs/evaluation_config.yaml`

---

##  Common Use Cases

### Use Case 1: Quick Test Generation

```python
# For fast prototyping
result = hybrid_gen.generate(
    requirements=epic.dict(),
    mode="hrm_only"  # Fastest mode
)
```

### Use Case 2: High-Quality Test Generation

```python
# For production deployment
result = hybrid_gen.generate(
    requirements=epic.dict(),
    mode="hybrid",  # Best quality
    merge_strategy="weighted",
    hrm_weight=0.6,
    sqe_weight=0.4
)
```

### Use Case 3: Knowledge Base Building

```python
# Generate and index continuously
for epic in epic_backlog:
    result = hybrid_gen.generate(requirements=epic.dict())
    indexer.index_from_generated_results(result["test_cases"])
```

### Use Case 4: Similar Test Discovery

```python
# Find existing tests before generating new ones
similar = retriever.retrieve_similar_test_cases(
    requirement=new_story.dict(),
    top_k=10,
    min_similarity=0.8
)
```

---

## 🚦 Performance Tips

### Tip 1: Batch Processing

```python
# Process multiple epics efficiently
for epic in epics_batch:
    result = hybrid_gen.generate(requirements=epic.dict())
    all_results.append(result)
```

### Tip 2: Caching

```python
# Enable caching for RAG retrieval
rag_retriever = RAGRetriever(
    vector_store=vector_store,
    embedding_generator=embedding_gen,
    cache_embeddings=True  # Cache for faster repeated queries
)
```

### Tip 3: Async Operations

```python
# Use async for API calls
import asyncio

async def generate_async(epic):
    return await hybrid_gen.generate_async(requirements=epic.dict())

results = await asyncio.gather(*[generate_async(e) for e in epics])
```

---

## [DONE] Next Steps

1. **Test with Your Data**
   - Replace sample epic with your requirements
   - Generate test cases
   - Review and validate output

2. **Fine-Tune Configuration**
   - Adjust merge strategies
   - Tune weights
   - Configure RAG parameters

3. **Build Knowledge Base**
   - Index existing test cases
   - Generate new tests
   - Auto-index results

4. **Deploy to Production**
   - Set up monitoring
   - Configure auto-scaling
   - Enable security features

---

## 🆘 Support

- **Documentation:** See all `*.md` files in project root
- **API Docs:** http://localhost:8000/docs
- **Test Examples:** `tests/` directory
- **Config Files:** `configs/` directory

---

**Ready to generate test cases? Let's go! **
