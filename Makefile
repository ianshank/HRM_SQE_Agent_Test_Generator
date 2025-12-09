# Makefile for HRM SQE Agent Test Generator
# Common development tasks and commands

.PHONY: help install install-dev test lint format type-check security \
        docker-build docker-up docker-down clean docs serve watch

# Default target
help:
	@echo "HRM SQE Agent Test Generator - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make setup          Complete development setup"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-int       Run integration tests only"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linters (ruff)"
	@echo "  make format         Format code (black, isort)"
	@echo "  make type-check     Run type checking (mypy)"
	@echo "  make security       Run security scans (bandit)"
	@echo "  make check          Run all quality checks"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start services with docker-compose"
	@echo "  make docker-down    Stop docker-compose services"
	@echo "  make docker-logs    View docker-compose logs"
	@echo ""
	@echo "Development:"
	@echo "  make serve          Start API server (development mode)"
	@echo "  make watch          Start drop folder watcher"
	@echo "  make clean          Clean build artifacts"
	@echo ""

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test]"
	pre-commit install

setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation."

# =============================================================================
# Testing
# =============================================================================

test:
	pytest hrm_eval/tests/ -v

test-unit:
	pytest hrm_eval/tests/ -v -m "not integration and not slow"

test-int:
	pytest hrm_eval/tests/ -v -m "integration"

test-cov:
	pytest hrm_eval/tests/ -v --cov=hrm_eval --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

test-fast:
	pytest hrm_eval/tests/ -v -x --ff

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check hrm_eval/

lint-fix:
	ruff check hrm_eval/ --fix

format:
	black hrm_eval/
	isort hrm_eval/

format-check:
	black hrm_eval/ --check
	isort hrm_eval/ --check

type-check:
	mypy hrm_eval/ --ignore-missing-imports

security:
	bandit -r hrm_eval/ -ll -ii --exclude hrm_eval/tests/

check: format-check lint type-check security
	@echo "All quality checks passed!"

pre-commit:
	pre-commit run --all-files

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t hrm-eval:latest .

docker-build-dev:
	docker build -t hrm-eval:dev --target development .

docker-build-gpu:
	docker build -t hrm-eval:gpu --target gpu .

docker-up:
	docker-compose up -d

docker-up-dev:
	docker-compose --profile dev up -d

docker-up-full:
	docker-compose --profile full up -d

docker-up-monitoring:
	docker-compose --profile monitoring up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v --rmi local

# =============================================================================
# Development
# =============================================================================

serve:
	uvicorn hrm_eval.api_service.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	gunicorn hrm_eval.api_service.main:app \
		--bind 0.0.0.0:8000 \
		--workers 4 \
		--worker-class uvicorn.workers.UvicornWorker

watch:
	python -m hrm_eval.drop_folder.cli watch

# =============================================================================
# Documentation
# =============================================================================

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage 2>/dev/null || true
	@echo "Cleaned build artifacts"

clean-docker:
	docker system prune -f

clean-all: clean clean-docker
	rm -rf vector_store_db/ logs/ 2>/dev/null || true

# =============================================================================
# Release
# =============================================================================

build:
	python -m build

publish-test:
	python -m twine upload --repository testpypi dist/*

publish:
	python -m twine upload dist/*
