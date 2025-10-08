"""
Security utilities for HRM evaluation system.

This module provides security-focused utilities including:
- Path validation and sanitization to prevent path traversal attacks
- Secure hash functions
- Input validation and sanitization
- Security logging and audit trail

All security-critical operations should use these utilities to ensure
consistent security posture across the codebase.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Union
import re

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass


class PathTraversalError(SecurityError):
    """Raised when path traversal attempt is detected."""
    pass


class InputValidationError(SecurityError):
    """Raised when input validation fails."""
    pass


class PathValidator:
    """
    Validator for file paths to prevent path traversal attacks.
    
    This class provides methods to validate that user-provided paths
    are safe and contained within expected directories.
    
    Example:
        validator = PathValidator(base_dir="/safe/directory")
        safe_path = validator.validate_path("user/input.txt")
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize path validator.
        
        Args:
            base_dir: Base directory to constrain all paths within
        """
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.exists():
            logger.warning(
                f"Base directory does not exist: {self.base_dir}. "
                f"It will be created when needed."
            )
    
    def validate_path(
        self,
        user_path: Union[str, Path],
        must_exist: bool = False,
        allow_create: bool = True,
    ) -> Path:
        """
        Validate that a user-provided path is safe.
        
        This method:
        1. Resolves the path to absolute form
        2. Checks it's within the base directory
        3. Rejects paths with traversal patterns
        4. Optionally validates existence
        
        Args:
            user_path: User-provided path to validate
            must_exist: If True, raise error if path doesn't exist
            allow_create: If True, allow paths that will be created
            
        Returns:
            Validated Path object
            
        Raises:
            PathTraversalError: If path traversal detected
            FileNotFoundError: If must_exist=True and path doesn't exist
            
        Example:
            safe_path = validator.validate_path("data/file.txt")
        """
        if not user_path:
            raise InputValidationError("Path cannot be empty")
        
        user_path_str = str(user_path)
        
        # Check for obviously malicious patterns
        dangerous_patterns = [
            '..',  # Parent directory references
            '~',   # Home directory expansion
            '$',   # Environment variable expansion
        ]
        
        for pattern in dangerous_patterns:
            if pattern in user_path_str:
                logger.error(
                    f"Path traversal attempt detected: {user_path_str} "
                    f"contains dangerous pattern: {pattern}"
                )
                raise PathTraversalError(
                    f"Path contains dangerous pattern '{pattern}': {user_path_str}"
                )
        
        # Resolve to absolute path
        if Path(user_path).is_absolute():
            # For absolute paths, resolve and check containment
            target_path = Path(user_path).resolve()
        else:
            # For relative paths, resolve relative to base_dir
            target_path = (self.base_dir / user_path).resolve()
        
        # Ensure target is within base directory
        try:
            target_path.relative_to(self.base_dir)
        except ValueError:
            logger.error(
                f"Path traversal attempt: {user_path_str} "
                f"resolves to {target_path} which is outside "
                f"base directory {self.base_dir}"
            )
            raise PathTraversalError(
                f"Path '{user_path_str}' attempts to escape base directory"
            )
        
        # Check existence if required
        if must_exist and not target_path.exists():
            raise FileNotFoundError(
                f"Required path does not exist: {target_path}"
            )
        
        if not allow_create and not target_path.exists():
            raise PathTraversalError(
                f"Path does not exist and creation not allowed: {target_path}"
            )
        
        logger.debug(f"Path validated: {user_path_str} -> {target_path}")
        return target_path
    
    def validate_directory(
        self,
        user_path: Union[str, Path],
        create_if_missing: bool = True,
    ) -> Path:
        """
        Validate a directory path and optionally create it.
        
        Args:
            user_path: User-provided directory path
            create_if_missing: If True, create directory if it doesn't exist
            
        Returns:
            Validated Path object for directory
            
        Raises:
            PathTraversalError: If path traversal detected
            ValueError: If path exists but is not a directory
        """
        validated_path = self.validate_path(
            user_path,
            must_exist=False,
            allow_create=True,
        )
        
        if validated_path.exists():
            if not validated_path.is_dir():
                raise ValueError(
                    f"Path exists but is not a directory: {validated_path}"
                )
        elif create_if_missing:
            validated_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {validated_path}")
        
        return validated_path


class SecureHasher:
    """
    Secure hashing utilities.
    
    Provides cryptographically secure hash functions to replace
    insecure algorithms like MD5 and SHA1.
    """
    
    @staticmethod
    def hash_string(
        data: str,
        algorithm: str = 'sha256',
        return_hex: bool = True,
    ) -> Union[str, bytes]:
        """
        Hash a string using a secure algorithm.
        
        Args:
            data: String data to hash
            algorithm: Hash algorithm ('sha256', 'sha512', 'sha3_256', etc.)
            return_hex: If True, return hexadecimal string; else return bytes
            
        Returns:
            Hash digest as hex string or bytes
            
        Example:
            hash_val = SecureHasher.hash_string("my data")
        """
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(
                f"Algorithm '{algorithm}' not available. "
                f"Available: {hashlib.algorithms_available}"
            )
        
        if algorithm in ['md5', 'sha1']:
            logger.warning(
                f"Insecure hash algorithm requested: {algorithm}. "
                f"Consider using sha256 or sha512 instead."
            )
        
        hasher = hashlib.new(algorithm)
        hasher.update(data.encode('utf-8'))
        
        if return_hex:
            return hasher.hexdigest()
        else:
            return hasher.digest()
    
    @staticmethod
    def hash_file(
        file_path: Path,
        algorithm: str = 'sha256',
        chunk_size: int = 8192,
    ) -> str:
        """
        Hash a file using a secure algorithm.
        
        Args:
            file_path: Path to file to hash
            algorithm: Hash algorithm to use
            chunk_size: Size of chunks to read (for large files)
            
        Returns:
            Hash digest as hexadecimal string
            
        Example:
            file_hash = SecureHasher.hash_file(Path("data.txt"))
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hasher = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        
        return hasher.hexdigest()


class InputValidator:
    """
    Validator for user inputs to prevent injection attacks.
    """
    
    @staticmethod
    def validate_filename(filename: str, max_length: int = 255) -> str:
        """
        Validate a filename for safety.
        
        Args:
            filename: Filename to validate
            max_length: Maximum allowed filename length
            
        Returns:
            Validated filename
            
        Raises:
            InputValidationError: If filename is invalid
        """
        if not filename:
            raise InputValidationError("Filename cannot be empty")
        
        if len(filename) > max_length:
            raise InputValidationError(
                f"Filename too long: {len(filename)} > {max_length}"
            )
        
        # Check for invalid characters
        invalid_chars = '<>:"|?*\x00'
        for char in invalid_chars:
            if char in filename:
                raise InputValidationError(
                    f"Filename contains invalid character: {char}"
                )
        
        # Reject files starting with dot (hidden files)
        if filename.startswith('.'):
            raise InputValidationError(
                "Filename cannot start with dot (hidden files not allowed)"
            )
        
        # Reject path separators
        if '/' in filename or '\\' in filename:
            raise InputValidationError(
                "Filename cannot contain path separators"
            )
        
        return filename
    
    @staticmethod
    def sanitize_string(
        data: str,
        max_length: Optional[int] = None,
        allow_newlines: bool = False,
    ) -> str:
        """
        Sanitize a string input.
        
        Args:
            data: String to sanitize
            max_length: Maximum allowed length
            allow_newlines: Whether to allow newline characters
            
        Returns:
            Sanitized string
            
        Raises:
            InputValidationError: If validation fails
        """
        if not isinstance(data, str):
            raise InputValidationError("Input must be a string")
        
        if max_length and len(data) > max_length:
            logger.warning(
                f"Input truncated from {len(data)} to {max_length} characters"
            )
            data = data[:max_length]
        
        if not allow_newlines:
            data = data.replace('\n', ' ').replace('\r', ' ')
        
        # Remove null bytes
        data = data.replace('\x00', '')
        
        return data


class SecurityAuditor:
    """
    Security event logging and auditing.
    
    Maintains an audit trail of security-relevant events.
    """
    
    def __init__(self, audit_log_path: Optional[Path] = None):
        """
        Initialize security auditor.
        
        Args:
            audit_log_path: Path to audit log file (optional)
        """
        self.audit_log_path = audit_log_path
        self.logger = logging.getLogger('security.audit')
    
    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = 'INFO',
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            description: Description of the event
            severity: Severity level (INFO, WARNING, ERROR, CRITICAL)
            metadata: Additional event metadata
        """
        log_entry = {
            'event_type': event_type,
            'description': description,
            'severity': severity,
            'metadata': metadata or {},
        }
        
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(f"Security Event: {event_type} - {description}")
        
        if self.audit_log_path:
            try:
                with open(self.audit_log_path, 'a') as f:
                    import json
                    import datetime
                    log_entry['timestamp'] = datetime.datetime.utcnow().isoformat()
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                self.logger.error(f"Failed to write to audit log: {e}")
