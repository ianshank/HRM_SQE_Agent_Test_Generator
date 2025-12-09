"""
Load testing for HRM SQE Agent Test Generator API.

Uses Locust for load testing with realistic user scenarios.

Usage:
    locust -f locustfile.py --host http://localhost:8000

Or run headless:
    locust -f locustfile.py --host http://localhost:8000 \
        --users 10 --spawn-rate 2 --run-time 60s --headless
"""

import json
import random
from locust import HttpUser, task, between, tag
from typing import Dict, Any, List


# Sample test data - NO HARDCODED in production, use fixtures
SAMPLE_EPICS: List[Dict[str, Any]] = [
    {
        "epic_id": "EPIC-LOAD-001",
        "title": "User Authentication System",
        "description": "Implement secure user authentication",
        "user_stories": [
            {
                "id": "US-001",
                "summary": "User login with email and password",
                "description": "As a user, I want to login with email and password",
                "acceptance_criteria": [
                    {"criteria": "Valid credentials allow login"},
                    {"criteria": "Invalid credentials show error"},
                    {"criteria": "Account locks after failed attempts"},
                ],
                "tech_stack": ["FastAPI", "PostgreSQL"],
            },
            {
                "id": "US-002",
                "summary": "Password reset",
                "description": "As a user, I want to reset my password",
                "acceptance_criteria": [
                    {"criteria": "Reset email is sent"},
                    {"criteria": "Link expires after 1 hour"},
                ],
                "tech_stack": ["FastAPI", "SendGrid"],
            },
        ],
        "tech_stack": ["FastAPI", "PostgreSQL", "Redis"],
        "architecture": "Microservices",
    },
    {
        "epic_id": "EPIC-LOAD-002",
        "title": "Product Catalog Management",
        "description": "Manage product catalog with search",
        "user_stories": [
            {
                "id": "US-003",
                "summary": "Search products by name",
                "description": "As a user, I want to search products",
                "acceptance_criteria": [
                    {"criteria": "Search returns matching products"},
                    {"criteria": "Search is case-insensitive"},
                ],
                "tech_stack": ["Elasticsearch", "FastAPI"],
            },
        ],
        "tech_stack": ["FastAPI", "Elasticsearch", "PostgreSQL"],
        "architecture": "Microservices",
    },
    {
        "epic_id": "EPIC-LOAD-003",
        "title": "Order Processing System",
        "description": "Handle customer orders",
        "user_stories": [
            {
                "id": "US-004",
                "summary": "Create new order",
                "description": "As a user, I want to place an order",
                "acceptance_criteria": [
                    {"criteria": "Order is created with items"},
                    {"criteria": "Order total is calculated correctly"},
                    {"criteria": "Confirmation email is sent"},
                ],
                "tech_stack": ["FastAPI", "PostgreSQL", "RabbitMQ"],
            },
        ],
        "tech_stack": ["FastAPI", "PostgreSQL", "RabbitMQ"],
        "architecture": "Event-Driven",
    },
]


class TestGenerationUser(HttpUser):
    """
    Simulated user for load testing test generation API.

    Represents typical usage patterns:
    - Health checks (frequent)
    - Test generation (primary use case)
    - Similar test search (supporting feature)
    """

    # Wait between requests (simulates real user behavior)
    wait_time = between(1, 5)

    def on_start(self):
        """Initialize user session."""
        # Check API health on start
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception("API health check failed")

    @task(10)
    @tag("health")
    def health_check(self):
        """Check API health (high frequency)."""
        self.client.get("/health")

    @task(5)
    @tag("health")
    def health_extended(self):
        """Check extended health status."""
        self.client.get("/health-extended")

    @task(3)
    @tag("generation")
    def generate_tests_basic(self):
        """Generate test cases from a random epic."""
        epic = random.choice(SAMPLE_EPICS)

        payload = {
            "epic": epic,
            "options": {
                "include_edge_cases": True,
                "include_negative_tests": True,
                "max_tests_per_story": 5,
            },
        }

        with self.client.post(
            "/api/v1/generate-tests",
            json=payload,
            catch_response=True,
            name="/api/v1/generate-tests",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "test_cases" in data:
                    response.success()
                else:
                    response.failure("Missing test_cases in response")
            elif response.status_code == 503:
                # Service not initialized - mark as expected failure
                response.failure("Service not initialized")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    @tag("generation", "rag")
    def generate_tests_with_rag(self):
        """Generate test cases with RAG context."""
        epic = random.choice(SAMPLE_EPICS)

        payload = {
            "epic": epic,
            "options": {
                "mode": "hybrid",
                "include_rag_context": True,
                "top_k_similar": 5,
            },
        }

        with self.client.post(
            "/api/v1/generate-tests-rag",
            json=payload,
            catch_response=True,
            name="/api/v1/generate-tests-rag",
        ) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(2)
    @tag("search")
    def search_similar_tests(self):
        """Search for similar test cases."""
        queries = [
            "user authentication login",
            "password reset email",
            "product search filter",
            "order creation checkout",
            "payment processing",
        ]

        payload = {
            "query": random.choice(queries),
            "top_k": 5,
        }

        with self.client.post(
            "/api/v1/search-similar",
            json=payload,
            catch_response=True,
            name="/api/v1/search-similar",
        ) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)
    @tag("batch")
    def batch_generation(self):
        """Batch generation (less frequent, heavier)."""
        epics = random.sample(SAMPLE_EPICS, min(2, len(SAMPLE_EPICS)))

        payload = {
            "epics": epics,
            "options": {
                "include_edge_cases": True,
            },
        }

        with self.client.post(
            "/api/v1/batch-generate",
            json=payload,
            catch_response=True,
            name="/api/v1/batch-generate",
            timeout=60,  # Longer timeout for batch
        ) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class AdminUser(HttpUser):
    """
    Admin user for management operations.

    Less frequent, administrative tasks:
    - Index management
    - Workflow execution
    """

    wait_time = between(5, 15)
    weight = 1  # Less common than regular users

    @task(3)
    @tag("admin")
    def check_extended_health(self):
        """Check detailed system health."""
        self.client.get("/health-extended")

    @task(1)
    @tag("admin", "index")
    def index_test_cases(self):
        """Index sample test cases."""
        test_cases = [
            {
                "id": f"TC-LOAD-{i}",
                "description": f"Test case {i} for load testing",
                "type": random.choice(["positive", "negative", "edge"]),
                "priority": random.choice(["P1", "P2", "P3"]),
            }
            for i in range(5)
        ]

        payload = {
            "test_cases": test_cases,
            "source": "load_test",
        }

        with self.client.post(
            "/api/v1/index-test-cases",
            json=payload,
            catch_response=True,
            name="/api/v1/index-test-cases",
        ) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class QuickHealthCheck(HttpUser):
    """
    Minimal user for health check monitoring.

    Simulates monitoring systems doing frequent health checks.
    """

    wait_time = between(0.5, 2)
    weight = 2  # More common

    @task
    def health(self):
        """Simple health check."""
        self.client.get("/health")


# Load test configuration profiles
class StressTestUser(TestGenerationUser):
    """Aggressive user for stress testing."""
    wait_time = between(0.1, 0.5)


class SoakTestUser(TestGenerationUser):
    """Steady user for soak testing."""
    wait_time = between(2, 8)
