#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filesystem Adapter
==================

Concrete implementation of FilePort for local filesystem access.

This adapter handles:
- Reading and writing text files
- JSON serialization/deserialization  
- Directory creation
- Error handling and reporting
"""

import json
from pathlib import Path
from typing import Dict, Any


class LocalFileAdapter:
    """Local filesystem implementation of FilePort protocol."""
    
    def __init__(self, base_path: str = "."):
        """Initialize adapter.
        
        Args:
            base_path: Base directory for relative paths (default: current directory)
        """
        self.base_path = Path(base_path).resolve()
    
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text file content.
        
        Args:
            path: File path (absolute or relative to base_path)
            encoding: Text encoding (default: utf-8)
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file can't be read
            UnicodeDecodeError: If encoding is wrong
        """
        file_path = self._resolve_path(path)
        return file_path.read_text(encoding=encoding)
    
    def write_text(self, path: str, content: str, encoding: str = "utf-8",
                   create_dirs: bool = False) -> int:
        """Write text to file.
        
        Args:
            path: File path (absolute or relative to base_path)
            content: Text content to write
            encoding: Text encoding (default: utf-8)
            create_dirs: Create parent directories if they don't exist
            
        Returns:
            Number of bytes written
            
        Raises:
            PermissionError: If file can't be written
            FileNotFoundError: If parent directory doesn't exist and create_dirs=False
        """
        file_path = self._resolve_path(path)
        
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_path.write_text(content, encoding=encoding)
        return file_path.stat().st_size
    
    def exists(self, path: str) -> bool:
        """Check if file exists.
        
        Args:
            path: File path to check
            
        Returns:
            True if file exists, False otherwise
        """
        file_path = self._resolve_path(path)
        return file_path.exists()
    
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
        content = self.read_text(path, encoding=encoding)
        return json.loads(content)
    
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
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        return self.write_text(path, content, encoding=encoding, create_dirs=create_dirs)
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base_path.
        
        Args:
            path: Path to resolve
            
        Returns:
            Resolved absolute Path object
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.base_path / p).resolve()


class InMemoryFileAdapter:
    """In-memory implementation of FilePort for testing.
    
    This adapter stores all files in memory, making it perfect for:
    - Unit tests (no filesystem I/O)
    - Integration tests (fast and isolated)
    - Development (no cleanup required)
    """
    
    def __init__(self):
        """Initialize in-memory filesystem."""
        self._files: Dict[str, str] = {}
    
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from in-memory file.
        
        Args:
            path: File path
            encoding: Ignored for in-memory implementation
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist in memory
        """
        if path not in self._files:
            raise FileNotFoundError(f"File not found: {path}")
        return self._files[path]
    
    def write_text(self, path: str, content: str, encoding: str = "utf-8",
                   create_dirs: bool = False) -> int:
        """Write text to in-memory file.
        
        Args:
            path: File path
            content: Text content to write
            encoding: Ignored for in-memory implementation
            create_dirs: Ignored for in-memory implementation
            
        Returns:
            Number of bytes written
        """
        self._files[path] = content
        return len(content.encode('utf-8'))
    
    def exists(self, path: str) -> bool:
        """Check if file exists in memory.
        
        Args:
            path: File path to check
            
        Returns:
            True if file exists, False otherwise
        """
        return path in self._files
    
    def read_json(self, path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read and parse JSON from in-memory file.
        
        Args:
            path: File path to JSON file
            encoding: Ignored for in-memory implementation
            
        Returns:
            Parsed JSON as dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If content is not valid JSON
        """
        content = self.read_text(path, encoding=encoding)
        return json.loads(content)
    
    def write_json(self, path: str, data: Dict[str, Any], encoding: str = "utf-8",
                   indent: int = 2, create_dirs: bool = False) -> int:
        """Write dictionary to in-memory JSON file.
        
        Args:
            path: File path for JSON file
            data: Dictionary to serialize
            encoding: Ignored for in-memory implementation
            indent: JSON indentation (default: 2)
            create_dirs: Ignored for in-memory implementation
            
        Returns:
            Number of bytes written
        """
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        return self.write_text(path, content, encoding=encoding, create_dirs=create_dirs)
    
    def clear(self) -> None:
        """Clear all in-memory files."""
        self._files.clear()
    
    def list_files(self) -> list[str]:
        """List all files in memory (useful for debugging tests).
        
        Returns:
            List of file paths
        """
        return list(self._files.keys())
