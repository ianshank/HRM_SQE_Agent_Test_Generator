"""
Drop Folder System for Automated Test Generation.

Monitors a folder for requirement files and automatically generates
test cases using the RAG+HRM hybrid workflow.
"""

from .processor import DropFolderProcessor, ProcessingResult
from .watcher import DropFolderWatcher
from .formatter import OutputFormatter

__all__ = [
    'DropFolderProcessor',
    'ProcessingResult',
    'DropFolderWatcher',
    'OutputFormatter',
]
