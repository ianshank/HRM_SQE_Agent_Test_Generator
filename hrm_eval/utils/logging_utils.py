"""
Logging utilities for HRM evaluation framework.

Provides structured logging with JSON formatting, multiple outputs (console, file),
and integration with experiment tracking systems.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors for console.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string with ANSI colors
        """
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        message = record.getMessage()
        
        log_str = (
            f"{color}[{timestamp}] {record.levelname:8s}{reset} "
            f"{record.name:20s} | {message}"
        )
        
        if record.exc_info:
            log_str += f"\n{self.formatException(record.exc_info)}"
        
        return log_str


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files
        console_output: Enable console logging
        file_output: Enable file logging
        json_format: Use JSON format for file logs
        log_filename: Custom log filename
        
    Returns:
        Configured root logger
        
    Raises:
        ValueError: If invalid log level provided
    """
    log_level = getattr(logging, level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    root_logger.handlers.clear()
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = ConsoleFormatter()
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    if file_output and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"hrm_eval_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_path / log_filename)
        file_handler.setLevel(log_level)
        
        if json_format:
            file_formatter = JsonFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    root_logger.info(f"Logging initialized at {level} level")
    if log_dir:
        root_logger.info(f"Log directory: {log_dir}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name for the logger (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding contextual information."""
    
    def process(self, msg: str, kwargs):
        """
        Process log message with extra context.
        
        Args:
            msg: Log message
            kwargs: Additional keyword arguments
            
        Returns:
            Tuple of processed message and kwargs
        """
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs

