"""
Middleware for FastAPI application.

Provides logging, authentication, rate limiting, and error handling.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import time
import logging
from typing import Callable
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured logging of requests."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with logging."""
        request_id = id(request)
        start_time = time.time()
        
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            }
        )
        
        try:
            response = await call_next(request)
            
            process_time = time.time() - start_time
            
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"[{response.status_code}] in {process_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": process_time,
                }
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"after {process_time:.3f}s: {str(e)}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": process_time,
                },
                exc_info=True
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "InternalServerError",
                    "message": "An internal error occurred during request processing",
                }
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.
    
    Production use should leverage Redis or similar for distributed rate limiting.
    """
    
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application
            max_requests: Max requests per window
            window_seconds: Time window in seconds
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_counts = defaultdict(list)
        logger.info(
            f"Rate limiter initialized: {max_requests} requests per {window_seconds}s"
        )
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        
        current_time = time.time()
        
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if current_time - req_time < self.window_seconds
        ]
        
        if len(self.request_counts[client_ip]) >= self.max_requests:
            logger.warning(
                f"Rate limit exceeded for {client_ip}: "
                f"{len(self.request_counts[client_ip])} requests"
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s",
                    "retry_after": self.window_seconds,
                }
            )
        
        self.request_counts[client_ip].append(current_time)
        
        response = await call_next(request)
        
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware (placeholder).
    
    Production implementation should integrate with actual auth system
    (OAuth2, JWT, API keys, etc.).
    """
    
    def __init__(self, app, require_auth: bool = False):
        """
        Initialize authentication.
        
        Args:
            app: FastAPI application
            require_auth: Whether to require authentication
        """
        super().__init__(app)
        self.require_auth = require_auth
        self.public_paths = ["/api/v1/health", "/docs", "/openapi.json"]
        logger.info(f"Authentication middleware initialized (require_auth={require_auth})")
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with authentication."""
        if not self.require_auth or request.url.path in self.public_paths:
            return await call_next(request)
        
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Unauthorized",
                    "message": "Missing Authorization header",
                }
            )
        
        is_valid = await self._validate_token(auth_header)
        
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Unauthorized",
                    "message": "Invalid or expired token",
                }
            )
        
        return await call_next(request)
    
    async def _validate_token(self, auth_header: str) -> bool:
        """
        Validate authentication token.
        
        Args:
            auth_header: Authorization header value
            
        Returns:
            Whether token is valid
        """
        return True


def setup_cors(app):
    """
    Setup CORS middleware.
    
    Args:
        app: FastAPI application
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    logger.info("CORS middleware configured")

