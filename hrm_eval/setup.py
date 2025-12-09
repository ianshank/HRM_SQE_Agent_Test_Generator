"""Setup script for HRM Evaluation Framework.

This setup.py is maintained for backward compatibility.
For modern installations, use pyproject.toml with pip install -e .
"""

from setuptools import setup, find_packages

# Core dependencies required for the application to run
CORE_DEPENDENCIES = [
    # ML Core
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    # Embeddings & RAG
    "sentence-transformers>=2.2.0",
    "chromadb>=0.4.0",
    "pinecone-client>=2.2.0",
    # LLM & Agents
    "langchain>=0.1.0",
    "langchain-core>=0.1.0",
    "langgraph>=0.0.20",
    "openai>=1.0.0",
    "anthropic>=0.7.0",
    # Web Framework
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "httpx>=0.25.0",
    # Monitoring & Logging
    "prometheus-client>=0.17.0",
    "loguru>=0.7.0",
    "structlog>=23.1.0",
    # Utilities
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "tqdm>=4.65.0",
    "watchdog>=3.0.0",
    "tenacity>=8.2.0",
    # Experiment Tracking
    "tensorboard>=2.13.0",
    "wandb>=0.15.0",
]

# Development dependencies
DEV_DEPENDENCIES = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "pytest-asyncio>=0.21.0",
    "pytest-xdist>=3.3.0",
    "hypothesis>=6.82.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.3.0",
    "pre-commit>=3.3.0",
    "types-PyYAML>=6.0.0",
    "types-requests>=2.31.0",
]

# Testing dependencies
TEST_DEPENDENCIES = [
    "pytest>=7.3.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "pytest-asyncio>=0.21.0",
    "pytest-xdist>=3.3.0",
    "hypothesis>=6.82.0",
    "locust>=2.15.0",
]

# Documentation dependencies
DOCS_DEPENDENCIES = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
    "mkdocstrings[python]>=0.22.0",
]

setup(
    name="hrm_eval",
    version="1.0.0",
    description="HRM Evaluation & Test Generation System - AI-powered test case generation from requirements",
    long_description=open("../README.md", encoding="utf-8").read() if __name__ != "__main__" else "",
    long_description_content_type="text/markdown",
    author="Ian Cruickshank",
    author_email="ianshank@gmail.com",
    url="https://github.com/ianshank/HRM_SQE_Agent_Test_Generator",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "hrm_eval": ["configs/*.yaml", "py.typed"],
    },
    include_package_data=True,
    install_requires=CORE_DEPENDENCIES,
    extras_require={
        "dev": DEV_DEPENDENCIES,
        "test": TEST_DEPENDENCIES,
        "docs": DOCS_DEPENDENCIES,
        "all": DEV_DEPENDENCIES + TEST_DEPENDENCIES + DOCS_DEPENDENCIES,
    },
    entry_points={
        "console_scripts": [
            "hrm-eval=hrm_eval.cli:main",
            "hrm-api=hrm_eval.api_service.main:run_server",
            "hrm-drop-folder=hrm_eval.drop_folder.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
        "Typing :: Typed",
    ],
    keywords=[
        "machine-learning",
        "test-generation",
        "nlp",
        "transformer",
        "rag",
        "langchain",
        "fastapi",
    ],
)

