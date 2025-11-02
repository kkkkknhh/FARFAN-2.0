#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment and Clock Adapters
===============================

Concrete implementations of EnvPort and ClockPort.

These adapters handle:
- Environment variable access with type conversion
- Current time access with deterministic testing support
"""

import os
from datetime import datetime
from typing import Optional


class OsEnvAdapter:
    """Environment adapter using os.environ."""
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not set
            
        Returns:
            Environment variable value or default
        """
        return os.environ.get(key, default)
    
    def get_required(self, key: str) -> str:
        """Get required environment variable.
        
        Args:
            key: Environment variable name
            
        Returns:
            Environment variable value
            
        Raises:
            KeyError: If environment variable is not set
        """
        value = os.environ.get(key)
        if value is None:
            raise KeyError(f"Required environment variable not set: {key}")
        return value
    
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
        value = os.environ.get(key)
        if value is None:
            return default
        
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Cannot convert {key}={value} to int")
    
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
        value = os.environ.get(key)
        if value is None:
            return default
        
        value_lower = value.lower()
        if value_lower in ('true', '1', 'yes', 'on'):
            return True
        elif value_lower in ('false', '0', 'no', 'off'):
            return False
        else:
            return default


class DictEnvAdapter:
    """Environment adapter using a dictionary (for testing)."""
    
    def __init__(self, env_vars: Optional[dict] = None):
        """Initialize with environment variables.
        
        Args:
            env_vars: Dictionary of environment variables
        """
        self._env = env_vars or {}
    
    def set(self, key: str, value: str) -> None:
        """Set environment variable (testing helper).
        
        Args:
            key: Environment variable name
            value: Environment variable value
        """
        self._env[key] = value
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable."""
        return self._env.get(key, default)
    
    def get_required(self, key: str) -> str:
        """Get required environment variable."""
        value = self._env.get(key)
        if value is None:
            raise KeyError(f"Required environment variable not set: {key}")
        return value
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get environment variable as integer."""
        value = self._env.get(key)
        if value is None:
            return default
        
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Cannot convert {key}={value} to int")
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get environment variable as boolean."""
        value = self._env.get(key)
        if value is None:
            return default
        
        value_lower = value.lower()
        if value_lower in ('true', '1', 'yes', 'on'):
            return True
        elif value_lower in ('false', '0', 'no', 'off'):
            return False
        else:
            return default


class SystemClockAdapter:
    """Clock adapter using system time."""
    
    def now(self) -> datetime:
        """Get current datetime.
        
        Returns:
            Current datetime (timezone-naive)
        """
        return datetime.now()
    
    def now_iso(self) -> str:
        """Get current datetime as ISO 8601 string.
        
        Returns:
            Current datetime in ISO 8601 format
        """
        return datetime.now().isoformat()
    
    def timestamp(self) -> float:
        """Get current Unix timestamp.
        
        Returns:
            Current Unix timestamp (seconds since epoch)
        """
        return datetime.now().timestamp()


class FixedClockAdapter:
    """Clock adapter with fixed time (for testing).
    
    This adapter always returns the same time, making tests deterministic.
    """
    
    def __init__(self, fixed_time: datetime):
        """Initialize with fixed time.
        
        Args:
            fixed_time: The datetime to always return
        """
        self._fixed_time = fixed_time
    
    def set_time(self, new_time: datetime) -> None:
        """Update the fixed time (testing helper).
        
        Args:
            new_time: New datetime to return
        """
        self._fixed_time = new_time
    
    def now(self) -> datetime:
        """Get fixed datetime.
        
        Returns:
            The fixed datetime
        """
        return self._fixed_time
    
    def now_iso(self) -> str:
        """Get fixed datetime as ISO 8601 string.
        
        Returns:
            Fixed datetime in ISO 8601 format
        """
        return self._fixed_time.isoformat()
    
    def timestamp(self) -> float:
        """Get fixed Unix timestamp.
        
        Returns:
            Fixed Unix timestamp
        """
        return self._fixed_time.timestamp()
