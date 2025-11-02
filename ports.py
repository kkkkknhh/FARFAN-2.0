#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ports - Abstract I/O Interfaces
================================

Defines abstract protocols for all I/O operations in FARFAN 2.0.
These are the "ports" in Ports & Adapters (Hexagonal Architecture).

Core modules depend only on these protocols, never on concrete implementations.
Concrete implementations (adapters) live in infrastructure/.

Design Principles:
- Minimal surface area (fewest methods possible)
- Type-safe with Protocol
- No side effects in protocols themselves
- Easy to mock for testing
"""

from typing import Protocol, Dict, Any, Optional, List
from datetime import datetime


class FilePort(Protocol):
    """Protocol for file I/O operations.
    
    Minimal interface for reading and writing files.
    Implementations should handle encoding, error handling, and path resolution.
    """
    
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text file content.
        
        Args:
            path: File path (absolute or relative)
            encoding: Text encoding (default: utf-8)
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
            UnicodeDecodeError: If encoding is wrong
        """
        ...
    
    def write_text(self, path: str, content: str, encoding: str = "utf-8", 
                   create_dirs: bool = False) -> int:
        """Write text to file.
        
        Args:
            path: File path (absolute or relative)
            content: Text content to write
            encoding: Text encoding (default: utf-8)
            create_dirs: Create parent directories if they don't exist
            
        Returns:
            Number of bytes written
            
        Raises:
            PermissionError: If file can't be written
            FileNotFoundError: If parent directory doesn't exist and create_dirs=False
        """
        ...
    
    def exists(self, path: str) -> bool:
        """Check if file exists.
        
        Args:
            path: File path to check
            
        Returns:
            True if file exists, False otherwise
        """
        ...
    
    def read_json(self, path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read and parse JSON file.
        
        Args:
            path: File path to JSON file
            encoding: Text encoding (default: utf-8)
            
        Returns:
            Parsed JSON as dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        ...
    
    def write_json(self, path: str, data: Dict[str, Any], encoding: str = "utf-8",
                   indent: int = 2, create_dirs: bool = False) -> int:
        """Write dictionary to JSON file.
        
        Args:
            path: File path for JSON file
            data: Dictionary to serialize
            encoding: Text encoding (default: utf-8)
            indent: JSON indentation (default: 2)
            create_dirs: Create parent directories if they don't exist
            
        Returns:
            Number of bytes written
        """
        ...


class HttpPort(Protocol):
    """Protocol for HTTP operations.
    
    Minimal interface for making HTTP requests.
    Implementations should handle retries, timeouts, and error handling.
    """
    
    def get(self, url: str, headers: Optional[Dict[str, str]] = None,
            timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP GET request.
        
        Args:
            url: URL to request
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            
        Returns:
            Response with 'status', 'body', 'headers' keys
            
        Raises:
            TimeoutError: If request times out
            ConnectionError: If connection fails
        """
        ...
    
    def post(self, url: str, data: Dict[str, Any],
             headers: Optional[Dict[str, str]] = None,
             timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP POST request.
        
        Args:
            url: URL to request
            data: Data to send in request body
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            
        Returns:
            Response with 'status', 'body', 'headers' keys
            
        Raises:
            TimeoutError: If request times out
            ConnectionError: If connection fails
        """
        ...


class EnvPort(Protocol):
    """Protocol for environment variable access.
    
    Minimal interface for reading environment configuration.
    Implementations should provide defaults and validation.
    """
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not set
            
        Returns:
            Environment variable value or default
        """
        ...
    
    def get_required(self, key: str) -> str:
        """Get required environment variable.
        
        Args:
            key: Environment variable name
            
        Returns:
            Environment variable value
            
        Raises:
            KeyError: If environment variable is not set
        """
        ...
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get environment variable as integer.
        
        Args:
            key: Environment variable name
            default: Default value if not set
            
        Returns:
            Environment variable as integer or default
            
        Raises:
            ValueError: If value cannot be converted to int
        """
        ...
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get environment variable as boolean.
        
        Args:
            key: Environment variable name
            default: Default value if not set
            
        Returns:
            Environment variable as boolean or default
            
        Note:
            Treats "true", "1", "yes", "on" as True (case-insensitive)
            Treats "false", "0", "no", "off" as False (case-insensitive)
        """
        ...


class ClockPort(Protocol):
    """Protocol for time operations.
    
    Minimal interface for getting current time.
    Enables deterministic testing by injecting fake clocks.
    """
    
    def now(self) -> datetime:
        """Get current datetime.
        
        Returns:
            Current datetime (timezone-aware or naive depending on implementation)
        """
        ...
    
    def now_iso(self) -> str:
        """Get current datetime as ISO 8601 string.
        
        Returns:
            Current datetime in ISO 8601 format
        """
        ...
    
    def timestamp(self) -> float:
        """Get current Unix timestamp.
        
        Returns:
            Current Unix timestamp (seconds since epoch)
        """
        ...


class LogPort(Protocol):
    """Protocol for logging operations.
    
    Minimal interface for structured logging.
    Implementations should handle formatting and output destinations.
    """
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message.
        
        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        ...
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message.
        
        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        ...
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message.
        
        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        ...
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message.
        
        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        ...
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message.
        
        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        ...


class CachePort(Protocol):
    """Protocol for caching operations.
    
    Minimal interface for key-value caching.
    Implementations can use memory, Redis, etc.
    """
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        ...
    
    def delete(self, key: str) -> None:
        """Delete value from cache.
        
        Args:
            key: Cache key
        """
        ...
    
    def clear(self) -> None:
        """Clear all cached values."""
        ...


class ModelPort(Protocol):
    """Protocol for ML model operations.
    
    Minimal interface for loading and using ML models.
    Abstracts away model storage, loading, and inference.
    """
    
    def load(self, model_name: str, device: Optional[str] = None) -> Any:
        """Load ML model.
        
        Args:
            model_name: Model identifier (e.g., "BAAI/bge-m3")
            device: Device to load model on ("cpu", "cuda", etc.)
            
        Returns:
            Loaded model object
            
        Raises:
            ValueError: If model name is invalid
            RuntimeError: If model cannot be loaded
        """
        ...
    
    def embed_batch(self, model: Any, texts: List[str], 
                   batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for batch of texts.
        
        Args:
            model: Loaded model object
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        ...
