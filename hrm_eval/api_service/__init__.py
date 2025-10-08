"""FastAPI service for requirements to test cases generation."""

from .main import app, get_generator

__all__ = ["app", "get_generator"]

