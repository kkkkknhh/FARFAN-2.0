#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependency Injection Factory
=============================

Creates production and test dependencies for FARFAN 2.0 orchestration.

This module provides factory functions that wire up all adapters (ports & adapters pattern).

Design Principles:
- Single responsibility: Only create dependencies
- No business logic
- Easy to test with mock dependencies
- Production and test configurations separated
"""

from typing import Dict, Any
from datetime import datetime
import logging

from ports import (
    FilePort,
    HttpPort,
    EnvPort,
    ClockPort,
    LogPort,
    CachePort,
    ModelPort,
)
from infrastructure.filesystem import LocalFileAdapter
from infrastructure.http import RequestsHttpAdapter
from infrastructure.environment import OsEnvAdapter, SystemClockAdapter as EnvSystemClockAdapter


# ============================================================================
# Production Dependencies
# ============================================================================


class StandardLogAdapter:
    """Production logging adapter using Python logging."""
    
    def __init__(self, logger_name: str = "farfan"):
        """Initialize with logger name."""
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)


class InMemoryCacheAdapter:
    """Simple in-memory cache adapter."""
    
    def __init__(self):
        """Initialize cache."""
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in cache (TTL ignored for simplicity)."""
        self._cache[key] = value
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()


class SimpleModelAdapter:
    """Simple model adapter - placeholder for actual ML model loading."""
    
    def load(self, model_name: str, device: str = None) -> Any:
        """Load ML model (placeholder - actual implementation would use transformers)."""
        # In production, this would load actual models
        # For now, return a placeholder
        return {"model_name": model_name, "device": device or "cpu"}
    
    def embed_batch(self, model: Any, texts: list, batch_size: int = 32) -> list:
        """Generate embeddings (placeholder)."""
        # In production, this would generate real embeddings
        # For now, return placeholder embeddings
        import numpy as np
        return [np.zeros(768).tolist() for _ in texts]


def create_production_dependencies() -> Dict[str, Any]:
    """
    Create production dependencies with real adapters.
    
    Returns:
        Dictionary with all port implementations wired up.
    """
    return {
        "file_port": LocalFileAdapter(base_path="."),
        "http_port": RequestsHttpAdapter(),
        "env_port": OsEnvAdapter(),
        "clock_port": EnvSystemClockAdapter(),
        "log_port": StandardLogAdapter("farfan.production"),
        "cache_port": InMemoryCacheAdapter(),
        "model_port": SimpleModelAdapter(),
    }


# ============================================================================
# Test Dependencies
# ============================================================================


class InMemoryFileAdapter:
    """In-memory file adapter for testing."""
    
    def __init__(self):
        """Initialize in-memory filesystem."""
        self.files: Dict[str, str] = {}
    
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from memory."""
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        return self.files[path]
    
    def write_text(self, path: str, content: str, encoding: str = "utf-8",
                   create_dirs: bool = False) -> int:
        """Write text to memory."""
        self.files[path] = content
        return len(content.encode(encoding))
    
    def exists(self, path: str) -> bool:
        """Check if file exists in memory."""
        return path in self.files
    
    def read_json(self, path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read and parse JSON from memory."""
        import json
        content = self.read_text(path, encoding)
        return json.loads(content)
    
    def write_json(self, path: str, data: Dict[str, Any], encoding: str = "utf-8",
                   indent: int = 2, create_dirs: bool = False) -> int:
        """Write dictionary to JSON in memory."""
        import json
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        return self.write_text(path, content, encoding, create_dirs)


class FakeClockAdapter:
    """Fake clock adapter for deterministic testing."""
    
    def __init__(self, fixed_time: datetime = None):
        """Initialize with fixed time."""
        self.fixed_time = fixed_time or datetime(2024, 1, 1, 12, 0, 0)
    
    def now(self) -> datetime:
        """Get fixed datetime."""
        return self.fixed_time
    
    def now_iso(self) -> str:
        """Get fixed datetime as ISO 8601 string."""
        return self.fixed_time.isoformat()
    
    def timestamp(self) -> float:
        """Get fixed Unix timestamp."""
        return self.fixed_time.timestamp()


class RecordingLogAdapter:
    """Recording log adapter for test assertions."""
    
    def __init__(self):
        """Initialize log recorder."""
        self.logs: list = []
    
    def _record(self, level: str, message: str, **kwargs: Any) -> None:
        """Record log entry."""
        self.logs.append({"level": level, "message": message, "extra": kwargs})
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Record debug message."""
        self._record("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Record info message."""
        self._record("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Record warning message."""
        self._record("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Record error message."""
        self._record("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Record critical message."""
        self._record("critical", message, **kwargs)


def create_test_dependencies() -> Dict[str, Any]:
    """
    Create test dependencies with mock/in-memory adapters.
    
    Returns:
        Dictionary with all port implementations wired up for testing.
    """
    return {
        "file_port": InMemoryFileAdapter(),
        "http_port": None,  # Tests should mock HTTP calls
        "env_port": OsEnvAdapter(),  # Can use real env for tests
        "clock_port": FakeClockAdapter(),
        "log_port": RecordingLogAdapter(),
        "cache_port": InMemoryCacheAdapter(),
        "model_port": SimpleModelAdapter(),  # Placeholder for tests
    }
