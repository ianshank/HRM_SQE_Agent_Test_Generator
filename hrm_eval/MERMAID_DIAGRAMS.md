# HRM Test Generation System - Comprehensive Mermaid Diagrams

Complete architectural documentation using Mermaid diagrams.

---

## Table of Contents
1. [Overall System Architecture](#1-overall-system-architecture)
2. [Component Architecture](#2-component-architecture)
3. [Data Flow Diagram](#3-data-flow-diagram)
4. [API Workflow](#4-api-workflow)
5. [Neural Network Architecture](#5-neural-network-architecture)
6. [RAG Integration](#6-rag-integration)
7. [SQE Agent Workflow](#7-sqe-agent-workflow)
8. [Test Generation Pipeline](#8-test-generation-pipeline)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Class Diagrams](#10-class-diagrams)

---

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        CLI[CLI Interface]
        API_CLIENT[API Client/Postman]
        WEB[Web Dashboard]
    end
    
    subgraph "API Gateway"
        FASTAPI[FastAPI Server<br/>10 REST Endpoints]
        AUTH[Authentication]
        RATE[Rate Limiter]
    end
    
    subgraph "Orchestration Layer"
        HYBRID[Hybrid Generator<br/>3 Modes, 3 Strategies]
        WORKFLOW[Workflow Manager<br/>Multi-Agent Coordination]
        CONTEXT[Context Builder<br/>Enrichment Engine]
    end
    
    subgraph "Core Processing Components"
        HRM[HRM Model<br/>PyTorch Transformer<br/>Step 7566]
        SQE[SQE Agent<br/>LangGraph<br/>5 Nodes, 4 Tools]
        RAG[RAG Retriever<br/>ChromaDB<br/>384-dim Embeddings]
    end
    
    subgraph "Supporting Services"
        REQ_PARSER[Requirements Parser<br/>Pydantic Schemas]
        COVERAGE[Coverage Analyzer<br/>Metrics Engine]
        POST_PROC[Post Processor<br/>Template Engine]
    end
    
    subgraph "Data Layer"
        VECTOR_DB[(Vector Store<br/>ChromaDB/Pinecone)]
        CONFIG[(Configuration<br/>YAML Files)]
        RESULTS[(Test Results<br/>JSON/CSV)]
    end
    
    CLI --> FASTAPI
    API_CLIENT --> FASTAPI
    WEB --> FASTAPI
    
    FASTAPI --> AUTH
    FASTAPI --> RATE
    AUTH --> HYBRID
    RATE --> HYBRID
    
    HYBRID --> HRM
    HYBRID --> SQE
    HYBRID --> RAG
    
    WORKFLOW --> HYBRID
    CONTEXT --> HYBRID
    
    HRM --> POST_PROC
    SQE --> POST_PROC
    RAG --> VECTOR_DB
    
    POST_PROC --> COVERAGE
    COVERAGE --> RESULTS
    
    REQ_PARSER --> HYBRID
    CONFIG --> HYBRID
    
    style FASTAPI fill:#4CAF50
    style HYBRID fill:#2196F3
    style HRM fill:#FF9800
    style SQE fill:#9C27B0
    style RAG fill:#00BCD4
    style VECTOR_DB fill:#607D8B
```

---

## 2. Component Architecture

### 2.1 FastAPI Service Architecture

```mermaid
graph LR
    subgraph "FastAPI Application"
        MAIN[main.py<br/>Application Entry]
        MODELS[models.py<br/>Pydantic Models]
        MIDDLEWARE[middleware.py<br/>Logging & Rate Limit]
        RAG_MODELS[rag_sqe_models.py<br/>Extended Models]
        
        MAIN --> MODELS
        MAIN --> MIDDLEWARE
        MAIN --> RAG_MODELS
    end
    
    subgraph "API Endpoints"
        E1[POST /initialize]
        E2[POST /generate-tests]
        E3[POST /initialize-rag]
        E4[POST /generate-tests-rag]
        E5[POST /index-test-cases]
        E6[POST /search-similar]
        E7[POST /execute-workflow]
        E8[GET /health]
        E9[GET /health-extended]
        E10[POST /batch-generate]
    end
    
    MAIN --> E1
    MAIN --> E2
    MAIN --> E3
    MAIN --> E4
    MAIN --> E5
    MAIN --> E6
    MAIN --> E7
    MAIN --> E8
    MAIN --> E9
    MAIN --> E10
    
    style MAIN fill:#4CAF50
    style MIDDLEWARE fill:#FF9800
```

### 2.2 Orchestration Components

```mermaid
graph TB
    subgraph "Hybrid Generator"
        HG_INIT[Initialize<br/>HRM, SQE, RAG]
        HG_MODE{Generation Mode?}
        HG_HRM[HRM Only Path]
        HG_SQE[SQE Only Path]
        HG_HYBRID[Hybrid Path]
        HG_MERGE[Merge Strategy<br/>Weighted/Union/Intersection]
        HG_OUTPUT[Merged Results]
        
        HG_INIT --> HG_MODE
        HG_MODE -->|hrm_only| HG_HRM
        HG_MODE -->|sqe_only| HG_SQE
        HG_MODE -->|hybrid| HG_HYBRID
        
        HG_HRM --> HG_MERGE
        HG_SQE --> HG_MERGE
        HG_HYBRID --> HG_MERGE
        HG_MERGE --> HG_OUTPUT
    end
    
    subgraph "Workflow Manager"
        WM_VALIDATE[Validate Requirements]
        WM_GENERATE[Generate Tests]
        WM_ANALYZE[Analyze Coverage]
        WM_INDEX[Auto-Index Results]
        WM_STATS[Collect Statistics]
        
        WM_VALIDATE --> WM_GENERATE
        WM_GENERATE --> WM_ANALYZE
        WM_ANALYZE --> WM_INDEX
        WM_INDEX --> WM_STATS
    end
    
    subgraph "Context Builder"
        CB_EXTRACT[Extract Context]
        CB_RAG[RAG Retrieval]
        CB_ENRICH[Enrich with History]
        CB_FORMAT[Format for Models]
        
        CB_EXTRACT --> CB_RAG
        CB_RAG --> CB_ENRICH
        CB_ENRICH --> CB_FORMAT
    end
    
    style HG_INIT fill:#2196F3
    style WM_VALIDATE fill:#9C27B0
    style CB_EXTRACT fill:#00BCD4
```

### 2.3 Core Models Architecture

```mermaid
graph TB
    subgraph "HRM Model (PyTorch)"
        HRM_LOAD[Load Checkpoint<br/>Step 7566]
        HRM_TOKENIZE[Tokenizer<br/>Vocab: 12]
        HRM_EMBED[Puzzle Embedding<br/>dim: 128]
        HRM_TRANS[Transformer Stack<br/>6 Layers, 8 Heads]
        HRM_ACTION[Action Head<br/>Token Probs]
        HRM_Q[Q-Head<br/>RL Values]
        
        HRM_LOAD --> HRM_TOKENIZE
        HRM_TOKENIZE --> HRM_EMBED
        HRM_EMBED --> HRM_TRANS
        HRM_TRANS --> HRM_ACTION
        HRM_TRANS --> HRM_Q
    end
    
    subgraph "SQE Agent (LangGraph)"
        SQE_STATE[Agent State<br/>TypedDict]
        SQE_TOOLS[4 Custom Tools<br/>Parse, Generate, Validate, Enhance]
        SQE_GRAPH[LangGraph Workflow<br/>5 Nodes]
        SQE_LLM[LLM Integration<br/>GPT-4/Claude]
        
        SQE_STATE --> SQE_GRAPH
        SQE_TOOLS --> SQE_GRAPH
        SQE_LLM --> SQE_GRAPH
    end
    
    subgraph "RAG System"
        RAG_EMBED[Sentence Transformers<br/>384-dim]
        RAG_STORE[ChromaDB/Pinecone<br/>Vector Store]
        RAG_RETRIEVE[Top-K Retrieval<br/>Similarity Search]
        RAG_CONTEXT[Context Builder<br/>Format Results]
        
        RAG_EMBED --> RAG_STORE
        RAG_STORE --> RAG_RETRIEVE
        RAG_RETRIEVE --> RAG_CONTEXT
    end
    
    style HRM_LOAD fill:#FF9800
    style SQE_STATE fill:#9C27B0
    style RAG_EMBED fill:#00BCD4
```

---

## 3. Data Flow Diagram

```mermaid
flowchart TD
    START([User Submits<br/>Requirements])
    
    subgraph "Input Processing"
        PARSE[Parse Epic<br/>Extract Stories & Criteria]
        VALIDATE[Validate Schema<br/>Pydantic Models]
        EXTRACT[Extract Test Contexts<br/>Map to Test Scenarios]
    end
    
    subgraph "Context Enrichment"
        RAG_QUERY[Generate Embeddings<br/>for Requirements]
        RAG_SEARCH[Search Vector DB<br/>Find Similar Tests]
        RAG_BUILD[Build Context<br/>Top-K Results]
    end
    
    subgraph "Test Generation"
        HRM_GEN[HRM Generation<br/>Token Prediction]
        SQE_GEN[SQE Generation<br/>Agent Reasoning]
        MERGE[Merge Results<br/>Apply Strategy]
    end
    
    subgraph "Post Processing"
        FORMAT[Format Test Cases<br/>Structure Output]
        ANALYZE[Coverage Analysis<br/>Map to Criteria]
        LABEL[Priority & Labels<br/>Classify Tests]
    end
    
    subgraph "Output & Storage"
        INDEX[Index to Vector DB<br/>Store for Future]
        EXPORT[Export Results<br/>JSON/CSV]
        STATS[Generate Statistics<br/>Metrics Report]
    end
    
    END([Return Results<br/>to User])
    
    START --> PARSE
    PARSE --> VALIDATE
    VALIDATE --> EXTRACT
    
    EXTRACT --> RAG_QUERY
    RAG_QUERY --> RAG_SEARCH
    RAG_SEARCH --> RAG_BUILD
    
    RAG_BUILD --> HRM_GEN
    RAG_BUILD --> SQE_GEN
    
    HRM_GEN --> MERGE
    SQE_GEN --> MERGE
    
    MERGE --> FORMAT
    FORMAT --> ANALYZE
    ANALYZE --> LABEL
    
    LABEL --> INDEX
    INDEX --> EXPORT
    EXPORT --> STATS
    
    STATS --> END
    
    style START fill:#4CAF50
    style MERGE fill:#2196F3
    style END fill:#4CAF50
```

---

## 4. API Workflow

### 4.1 Test Generation Workflow (RAG-Enhanced)

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant HybridGen
    participant RAG
    participant HRM
    participant SQE
    participant VectorDB
    participant Results
    
    Client->>FastAPI: POST /generate-tests-rag
    FastAPI->>FastAPI: Validate Request
    FastAPI->>HybridGen: Initialize Components
    
    HybridGen->>RAG: Retrieve Similar Tests
    RAG->>VectorDB: Query Embeddings
    VectorDB-->>RAG: Top-K Results
    RAG-->>HybridGen: Context Built
    
    par Parallel Generation
        HybridGen->>HRM: Generate with Context
        HRM-->>HybridGen: HRM Test Cases
    and
        HybridGen->>SQE: Orchestrate with Context
        SQE-->>HybridGen: SQE Test Cases
    end
    
    HybridGen->>HybridGen: Merge Results (Weighted)
    HybridGen->>HybridGen: Analyze Coverage
    
    opt Auto-Index
        HybridGen->>VectorDB: Index New Tests
    end
    
    HybridGen->>Results: Save Results
    HybridGen-->>FastAPI: Return Response
    FastAPI-->>Client: JSON Response
```

### 4.2 Complete Workflow Execution

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Workflow
    participant Parser
    participant Generator
    participant Analyzer
    participant Indexer
    
    Client->>API: POST /execute-workflow
    API->>Workflow: Start Full Workflow
    
    rect rgb(200, 220, 255)
    note right of Workflow: Step 1: Validate
    Workflow->>Parser: Validate Requirements
    Parser-->>Workflow: Validation Result
    end
    
    rect rgb(255, 220, 200)
    note right of Workflow: Step 2: Generate
    Workflow->>Generator: Generate Test Cases
    Generator-->>Workflow: Generated Tests
    end
    
    rect rgb(220, 255, 200)
    note right of Workflow: Step 3: Analyze
    Workflow->>Analyzer: Analyze Coverage
    Analyzer-->>Workflow: Coverage Report
    end
    
    rect rgb(255, 240, 200)
    note right of Workflow: Step 4: Index
    Workflow->>Indexer: Index Test Cases
    Indexer-->>Workflow: Indexing Stats
    end
    
    Workflow-->>API: Workflow Complete
    API-->>Client: Full Results + Stats
```

---

## 5. Neural Network Architecture

```mermaid
graph TD
    subgraph "Input Layer"
        INPUT[Tokenized Input<br/>Vocab Size: 12]
    end
    
    subgraph "Embedding Layer"
        EMBED[Puzzle Embedding<br/>Dim: 128]
        POS[Positional Encoding<br/>Sinusoidal]
        
        INPUT --> EMBED
        EMBED --> POS
    end
    
    subgraph "Transformer Stack"
        direction TB
        TRANS1[Transformer Block 1<br/>8 Heads, d_k=64]
        TRANS2[Transformer Block 2<br/>8 Heads, d_k=64]
        TRANS3[Transformer Block 3<br/>8 Heads, d_k=64]
        TRANS4[Transformer Block 4<br/>8 Heads, d_k=64]
        TRANS5[Transformer Block 5<br/>8 Heads, d_k=64]
        TRANS6[Transformer Block 6<br/>8 Heads, d_k=64]
        
        POS --> TRANS1
        TRANS1 --> TRANS2
        TRANS2 --> TRANS3
        TRANS3 --> TRANS4
        TRANS4 --> TRANS5
        TRANS5 --> TRANS6
    end
    
    subgraph "Output Layer"
        NORM[Layer Normalization]
        DENSE[Dense Layer<br/>256 units, ReLU]
        
        TRANS6 --> NORM
        NORM --> DENSE
    end
    
    subgraph "Dual Heads"
        ACTION[Action Head<br/>Softmax, Vocab: 12]
        Q_HEAD[Q-Value Head<br/>Linear, Actions: 2]
        
        DENSE --> ACTION
        DENSE --> Q_HEAD
    end
    
    subgraph "Output"
        TOKEN_PROBS[Token Probabilities]
        Q_VALUES[Q-Values for RL]
        
        ACTION --> TOKEN_PROBS
        Q_HEAD --> Q_VALUES
    end
    
    style INPUT fill:#E3F2FD
    style EMBED fill:#BBDEFB
    style TRANS1 fill:#90CAF9
    style TRANS6 fill:#42A5F5
    style ACTION fill:#4CAF50
    style Q_HEAD fill:#FF9800
```

### Transformer Block Detail

```mermaid
graph TB
    subgraph "Single Transformer Block"
        INPUT_TB[Input from Previous Layer]
        
        subgraph "Multi-Head Attention"
            Q[Query]
            K[Key]
            V[Value]
            HEADS[8 Attention Heads<br/>d_k=64, d_v=64]
            CONCAT[Concatenate Heads]
            
            INPUT_TB --> Q
            INPUT_TB --> K
            INPUT_TB --> V
            Q --> HEADS
            K --> HEADS
            V --> HEADS
            HEADS --> CONCAT
        end
        
        ADD1[Add & Norm<br/>Residual Connection]
        
        subgraph "Feed-Forward Network"
            FF1[Linear<br/>dim → 512]
            RELU[ReLU Activation]
            DROP1[Dropout: 0.1]
            FF2[Linear<br/>512 → dim]
            
            FF1 --> RELU
            RELU --> DROP1
            DROP1 --> FF2
        end
        
        ADD2[Add & Norm<br/>Residual Connection]
        OUTPUT_TB[Output to Next Layer]
        
        CONCAT --> ADD1
        INPUT_TB -.residual.-> ADD1
        ADD1 --> FF1
        FF2 --> ADD2
        ADD1 -.residual.-> ADD2
        ADD2 --> OUTPUT_TB
    end
    
    style INPUT_TB fill:#E3F2FD
    style HEADS fill:#90CAF9
    style ADD1 fill:#64B5F6
    style ADD2 fill:#64B5F6
    style OUTPUT_TB fill:#42A5F5
```

---

## 6. RAG Integration

```mermaid
graph TB
    subgraph "RAG Vector Store System"
        direction LR
        
        subgraph "Embedding Generation"
            TEXT[Input Text<br/>Requirements/Tests]
            TOKENIZER[Tokenizer<br/>Sentence Transformers]
            ENCODER[Encoder<br/>all-MiniLM-L6-v2]
            EMBED_VEC[384-dim Embedding Vector]
            
            TEXT --> TOKENIZER
            TOKENIZER --> ENCODER
            ENCODER --> EMBED_VEC
        end
        
        subgraph "Vector Store"
            CHROMA[(ChromaDB)]
            PINE[(Pinecone)]
            CHOICE{Backend?}
            
            EMBED_VEC --> CHOICE
            CHOICE -->|local| CHROMA
            CHOICE -->|cloud| PINE
        end
        
        subgraph "Retrieval"
            QUERY[Query Vector]
            SIMILARITY[Cosine Similarity<br/>Top-K Search]
            FILTER[Filter by Threshold<br/>min_similarity: 0.7]
            RESULTS[Retrieved Documents]
            
            QUERY --> SIMILARITY
            CHROMA --> SIMILARITY
            PINE --> SIMILARITY
            SIMILARITY --> FILTER
            FILTER --> RESULTS
        end
        
        subgraph "Context Building"
            FORMAT[Format Results]
            METADATA[Add Metadata<br/>Similarity Scores]
            CONTEXT[Final Context String]
            
            RESULTS --> FORMAT
            FORMAT --> METADATA
            METADATA --> CONTEXT
        end
    end
    
    subgraph "Indexing Process"
        NEW_TESTS[New Test Cases]
        BATCH[Batch Processing<br/>Size: 100]
        EMBED_NEW[Generate Embeddings]
        STORE[Store in Vector DB]
        
        NEW_TESTS --> BATCH
        BATCH --> EMBED_NEW
        EMBED_NEW --> STORE
        STORE --> CHROMA
        STORE --> PINE
    end
    
    CONTEXT --> GENERATOR[Test Generator]
    
    style TEXT fill:#E3F2FD
    style CHROMA fill:#00BCD4
    style SIMILARITY fill:#4CAF50
    style CONTEXT fill:#FF9800
```

---

## 7. SQE Agent Workflow

```mermaid
stateDiagram-v2
    [*] --> Initialize
    
    state Initialize {
        [*] --> LoadState
        LoadState --> LoadTools
        LoadTools --> BuildGraph
        BuildGraph --> [*]
    }
    
    Initialize --> ParseRequirements
    
    state ParseRequirements {
        [*] --> ValidateInput
        ValidateInput --> ExtractStructure
        ExtractStructure --> [*]
    }
    
    ParseRequirements --> RetrieveContext
    
    state RetrieveContext {
        [*] --> QueryRAG
        QueryRAG --> BuildContext
        BuildContext --> [*]
    }
    
    RetrieveContext --> PlanTests
    
    state PlanTests {
        [*] --> AnalyzeRequirements
        AnalyzeRequirements --> GenerateStrategy
        GenerateStrategy --> DefineScenarios
        DefineScenarios --> [*]
    }
    
    PlanTests --> GenerateTests
    
    state GenerateTests {
        [*] --> UseHRMTool
        UseHRMTool --> EnhanceWithSQE
        EnhanceWithSQE --> [*]
    }
    
    GenerateTests --> ValidateResults
    
    state ValidateResults {
        [*] --> CheckCompleteness
        CheckCompleteness --> VerifyCoverage
        VerifyCoverage --> [*]
    }
    
    ValidateResults --> Decision
    
    state Decision <<choice>>
    Decision --> GenerateTests : Incomplete
    Decision --> Finalize : Complete
    
    state Finalize {
        [*] --> FormatOutput
        FormatOutput --> AddMetadata
        AddMetadata --> [*]
    }
    
    Finalize --> [*]
```

### SQE Agent Tools

```mermaid
graph LR
    subgraph "SQE Agent Tools"
        TOOL1[RequirementParserTool<br/>Parse Epic & Stories]
        TOOL2[TestCaseGeneratorTool<br/>Generate via HRM]
        TOOL3[CoverageAnalyzerTool<br/>Analyze Coverage]
        TOOL4[TestEnhancerTool<br/>Enhance Quality]
        
        STATE[Agent State<br/>Context & History]
        
        STATE --> TOOL1
        STATE --> TOOL2
        STATE --> TOOL3
        STATE --> TOOL4
        
        TOOL1 --> STATE
        TOOL2 --> STATE
        TOOL3 --> STATE
        TOOL4 --> STATE
    end
    
    subgraph "External Integrations"
        HRM[HRM Model]
        RAG[RAG Retriever]
        PARSER[Requirement Parser]
        
        TOOL1 --> PARSER
        TOOL2 --> HRM
        TOOL2 --> RAG
        TOOL3 --> PARSER
    end
    
    style STATE fill:#9C27B0
    style TOOL2 fill:#FF9800
```

---

## 8. Test Generation Pipeline

```mermaid
flowchart TD
    START([Requirements Input])
    
    subgraph "Phase 1: Parsing"
        P1[Load Epic JSON]
        P2[Validate Schema]
        P3[Extract User Stories]
        P4[Extract Acceptance Criteria]
        
        P1 --> P2
        P2 --> P3
        P3 --> P4
    end
    
    subgraph "Phase 2: Context Extraction"
        C1[Create Test Contexts]
        C2[Map to Test Scenarios]
        C3[Identify Test Types]
        
        P4 --> C1
        C1 --> C2
        C2 --> C3
    end
    
    subgraph "Phase 3: RAG Retrieval"
        R1[Generate Embeddings]
        R2[Search Vector DB]
        R3[Retrieve Top-K]
        R4[Build Context]
        
        C3 --> R1
        R1 --> R2
        R2 --> R3
        R3 --> R4
    end
    
    subgraph "Phase 4: Generation"
        MODE{Mode?}
        
        HRM_PATH[HRM Generation<br/>Tokenize → Infer → Decode]
        SQE_PATH[SQE Generation<br/>Plan → Generate → Validate]
        HYBRID_PATH[Hybrid<br/>Both HRM + SQE]
        
        R4 --> MODE
        MODE -->|hrm_only| HRM_PATH
        MODE -->|sqe_only| SQE_PATH
        MODE -->|hybrid| HYBRID_PATH
    end
    
    subgraph "Phase 5: Merging"
        STRATEGY{Strategy?}
        
        WEIGHTED[Weighted Merge<br/>60% HRM, 40% SQE]
        UNION[Union<br/>All Unique Tests]
        INTERSECTION[Intersection<br/>Common Tests]
        
        HRM_PATH --> STRATEGY
        SQE_PATH --> STRATEGY
        HYBRID_PATH --> STRATEGY
        
        STRATEGY -->|weighted| WEIGHTED
        STRATEGY -->|union| UNION
        STRATEGY -->|intersection| INTERSECTION
    end
    
    subgraph "Phase 6: Post-Processing"
        PP1[Format Test Cases]
        PP2[Add Priorities]
        PP3[Add Labels]
        PP4[Estimate Duration]
        
        WEIGHTED --> PP1
        UNION --> PP1
        INTERSECTION --> PP1
        PP1 --> PP2
        PP2 --> PP3
        PP3 --> PP4
    end
    
    subgraph "Phase 7: Analysis"
        A1[Coverage Analysis]
        A2[Gap Identification]
        A3[Generate Recommendations]
        
        PP4 --> A1
        A1 --> A2
        A2 --> A3
    end
    
    subgraph "Phase 8: Output"
        O1[Generate Report]
        O2[Export JSON/CSV]
        O3[Index to Vector DB]
        
        A3 --> O1
        O1 --> O2
        O2 --> O3
    end
    
    END([Test Cases<br/>+ Metadata])
    
    O3 --> END
    
    style START fill:#4CAF50
    style MODE fill:#2196F3
    style STRATEGY fill:#9C27B0
    style END fill:#4CAF50
```

---

## 9. Deployment Architecture

```mermaid
graph TB
    subgraph "Production Environment"
        LB[Load Balancer<br/>Nginx/AWS ALB]
        
        subgraph "API Cluster"
            API1[FastAPI Instance 1]
            API2[FastAPI Instance 2]
            API3[FastAPI Instance N]
        end
        
        subgraph "Worker Pool"
            W1[Worker 1<br/>HRM + SQE]
            W2[Worker 2<br/>HRM + SQE]
            W3[Worker N<br/>HRM + SQE]
        end
        
        subgraph "Data Layer"
            REDIS[(Redis Cache<br/>Session & Results)]
            POSTGRES[(PostgreSQL<br/>Metadata & Logs)]
            CHROMA[(ChromaDB<br/>Vector Store)]
        end
        
        subgraph "Monitoring"
            PROM[Prometheus<br/>Metrics]
            GRAFANA[Grafana<br/>Dashboards]
            LOGS[ELK Stack<br/>Centralized Logs]
        end
        
        subgraph "ML Model Storage"
            S3[(S3/MinIO<br/>Model Checkpoints)]
        end
    end
    
    USERS[Users/Clients] --> LB
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> REDIS
    API2 --> REDIS
    API3 --> REDIS
    
    API1 --> W1
    API2 --> W2
    API3 --> W3
    
    W1 --> CHROMA
    W2 --> CHROMA
    W3 --> CHROMA
    
    W1 --> S3
    W2 --> S3
    W3 --> S3
    
    API1 --> POSTGRES
    API1 --> PROM
    PROM --> GRAFANA
    API1 --> LOGS
    
    style LB fill:#4CAF50
    style REDIS fill:#FF9800
    style CHROMA fill:#00BCD4
    style PROM fill:#E91E63
```

---

## 10. Class Diagrams

### 10.1 Core Models

```mermaid
classDiagram
    class HRMModel {
        +checkpoint_path: str
        +device: str
        +vocab_size: int
        +embedding_dim: int
        +num_layers: int
        +num_heads: int
        +load_checkpoint()
        +tokenize(text)
        +generate(input_ids)
        +decode(tokens)
    }
    
    class SQEAgent {
        +llm: BaseChatModel
        +rag_retriever: RAGRetriever
        +hrm_generator: TestCaseGenerator
        +state: AgentState
        +build_workflow()
        +execute(requirements)
        +get_tools()
    }
    
    class RAGRetriever {
        +vector_store: VectorStore
        +embedding_gen: EmbeddingGenerator
        +top_k: int
        +min_similarity: float
        +retrieve_similar(query)
        +build_context(results)
    }
    
    class HybridTestGenerator {
        +hrm_generator: TestCaseGenerator
        +sqe_agent: SQEAgent
        +rag_retriever: RAGRetriever
        +mode: str
        +merge_strategy: str
        +generate(requirements)
        +merge_results(hrm, sqe)
    }
    
    class WorkflowManager {
        +hybrid_gen: HybridTestGenerator
        +req_parser: RequirementParser
        +coverage_analyzer: CoverageAnalyzer
        +execute_workflow(epic)
        +validate_step(epic)
        +generate_step(contexts)
        +analyze_step(tests)
    }
    
    HybridTestGenerator --> HRMModel
    HybridTestGenerator --> SQEAgent
    HybridTestGenerator --> RAGRetriever
    WorkflowManager --> HybridTestGenerator
    SQEAgent --> RAGRetriever
```

### 10.2 Data Models

```mermaid
classDiagram
    class Epic {
        +epic_id: str
        +title: str
        +user_stories: List[UserStory]
        +tech_stack: List[str]
        +architecture: str
    }
    
    class UserStory {
        +id: str
        +summary: str
        +description: str
        +acceptance_criteria: List[AcceptanceCriteria]
        +tech_stack: List[str]
    }
    
    class AcceptanceCriteria {
        +criteria: str
        +priority: str
    }
    
    class TestCase {
        +id: str
        +type: TestType
        +priority: TestPriority
        +description: str
        +preconditions: List[str]
        +test_steps: List[TestStep]
        +expected_results: List[ExpectedResult]
        +labels: List[str]
        +automation_level: AutomationLevel
    }
    
    class TestStep {
        +step_number: int
        +action: str
        +expected_result: str
    }
    
    class GenerationMetadata {
        +generation_mode: str
        +hrm_generated: int
        +sqe_generated: int
        +merged_count: int
        +rag_context_used: bool
        +generation_time_seconds: float
    }
    
    Epic "1" --> "*" UserStory
    UserStory "1" --> "*" AcceptanceCriteria
    TestCase "1" --> "*" TestStep
```

---

## Usage

To render these diagrams:

1. **GitHub/GitLab**: Copy directly to README.md (automatic rendering)
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Use [Mermaid Live Editor](https://mermaid.live/)
4. **Documentation Sites**: Works with Docusaurus, MkDocs, etc.

---

## Notes

- All diagrams are based on actual implementation
- Color coding: Green (API), Blue (Orchestration), Orange (HRM), Purple (SQE), Cyan (RAG)
- Diagrams are maintained alongside code changes
- Use Mermaid v9.0+ for best compatibility

---

**Last Updated:** October 7, 2025  
**HRM Version:** v9 Optimized (Step 7566)  
**Status:** Production Ready (85%)
