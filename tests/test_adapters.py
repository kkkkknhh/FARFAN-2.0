#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter Tests
=============

Tests for infrastructure adapters (filesystem, http, environment, clock).

Validates:
- LocalFileAdapter and InMemoryFileAdapter
- RequestsHttpAdapter and MockHttpAdapter (if requests is available)
- OsEnvAdapter and DictEnvAdapter
- SystemClockAdapter and FixedClockAdapter
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

from infrastructure.filesystem import LocalFileAdapter, InMemoryFileAdapter
from infrastructure.environment import OsEnvAdapter, DictEnvAdapter, SystemClockAdapter, FixedClockAdapter


class TestLocalFileAdapter:
    """Test local filesystem adapter."""
    
    def setup_method(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.adapter = LocalFileAdapter(base_path=self.temp_dir)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_write_and_read_text(self):
        """Test writing and reading text files."""
        content = "Test content\nLine 2"
        self.adapter.write_text("test.txt", content)
        
        read_content = self.adapter.read_text("test.txt")
        assert read_content == content
    
    def test_write_with_create_dirs(self):
        """Test creating parent directories."""
        content = "Nested file content"
        self.adapter.write_text("sub/dir/file.txt", content, create_dirs=True)
        
        read_content = self.adapter.read_text("sub/dir/file.txt")
        assert read_content == content
    
    def test_exists(self):
        """Test file existence check."""
        assert not self.adapter.exists("nonexistent.txt")
        
        self.adapter.write_text("exists.txt", "content")
        assert self.adapter.exists("exists.txt")
    
    def test_read_nonexistent_file(self):
        """Test that reading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            self.adapter.read_text("nonexistent.txt")
    
    def test_write_and_read_json(self):
        """Test writing and reading JSON files."""
        data = {
            "key1": "value1",
            "key2": 42,
            "nested": {"a": 1, "b": 2}
        }
        
        self.adapter.write_json("data.json", data)
        read_data = self.adapter.read_json("data.json")
        
        assert read_data == data
    
    def test_absolute_path(self):
        """Test that absolute paths are handled correctly."""
        abs_path = Path(self.temp_dir) / "absolute.txt"
        content = "Absolute path content"
        
        self.adapter.write_text(str(abs_path), content)
        read_content = self.adapter.read_text(str(abs_path))
        
        assert read_content == content


class TestInMemoryFileAdapter:
    """Test in-memory filesystem adapter."""
    
    def setup_method(self):
        """Create fresh adapter for each test."""
        self.adapter = InMemoryFileAdapter()
    
    def test_write_and_read_text(self):
        """Test writing and reading text in memory."""
        content = "Memory content"
        self.adapter.write_text("/test.txt", content)
        
        read_content = self.adapter.read_text("/test.txt")
        assert read_content == content
    
    def test_exists(self):
        """Test file existence check in memory."""
        assert not self.adapter.exists("/nonexistent.txt")
        
        self.adapter.write_text("/exists.txt", "content")
        assert self.adapter.exists("/exists.txt")
    
    def test_read_nonexistent_file(self):
        """Test that reading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            self.adapter.read_text("/nonexistent.txt")
    
    def test_write_and_read_json(self):
        """Test JSON operations in memory."""
        data = {"key": "value", "number": 123}
        
        self.adapter.write_json("/data.json", data)
        read_data = self.adapter.read_json("/data.json")
        
        assert read_data == data
    
    def test_clear(self):
        """Test clearing all files."""
        self.adapter.write_text("/file1.txt", "content1")
        self.adapter.write_text("/file2.txt", "content2")
        
        assert len(self.adapter.list_files()) == 2
        
        self.adapter.clear()
        assert len(self.adapter.list_files()) == 0
    
    def test_list_files(self):
        """Test listing files in memory."""
        self.adapter.write_text("/a.txt", "a")
        self.adapter.write_text("/b.txt", "b")
        self.adapter.write_text("/c.txt", "c")
        
        files = self.adapter.list_files()
        assert len(files) == 3
        assert "/a.txt" in files
        assert "/b.txt" in files
        assert "/c.txt" in files


class TestDictEnvAdapter:
    """Test dictionary-based environment adapter."""
    
    def test_get_existing(self):
        """Test getting existing environment variable."""
        adapter = DictEnvAdapter({"KEY1": "value1"})
        assert adapter.get("KEY1") == "value1"
    
    def test_get_nonexistent_with_default(self):
        """Test getting nonexistent variable with default."""
        adapter = DictEnvAdapter({})
        assert adapter.get("MISSING", "default") == "default"
    
    def test_get_required_exists(self):
        """Test getting required variable that exists."""
        adapter = DictEnvAdapter({"REQUIRED": "value"})
        assert adapter.get_required("REQUIRED") == "value"
    
    def test_get_required_missing(self):
        """Test that get_required raises error for missing variable."""
        adapter = DictEnvAdapter({})
        with pytest.raises(KeyError):
            adapter.get_required("MISSING")
    
    def test_get_int_valid(self):
        """Test getting integer environment variable."""
        adapter = DictEnvAdapter({"PORT": "8080"})
        assert adapter.get_int("PORT") == 8080
    
    def test_get_int_invalid(self):
        """Test that invalid integer raises error."""
        adapter = DictEnvAdapter({"PORT": "not-a-number"})
        with pytest.raises(ValueError):
            adapter.get_int("PORT")
    
    def test_get_int_missing_with_default(self):
        """Test getting missing integer with default."""
        adapter = DictEnvAdapter({})
        assert adapter.get_int("MISSING", 9000) == 9000
    
    def test_get_bool_true_values(self):
        """Test boolean true values."""
        adapter = DictEnvAdapter({
            "B1": "true",
            "B2": "1",
            "B3": "yes",
            "B4": "on",
            "B5": "TRUE",  # Case insensitive
        })
        
        assert adapter.get_bool("B1") is True
        assert adapter.get_bool("B2") is True
        assert adapter.get_bool("B3") is True
        assert adapter.get_bool("B4") is True
        assert adapter.get_bool("B5") is True
    
    def test_get_bool_false_values(self):
        """Test boolean false values."""
        adapter = DictEnvAdapter({
            "B1": "false",
            "B2": "0",
            "B3": "no",
            "B4": "off",
            "B5": "FALSE",  # Case insensitive
        })
        
        assert adapter.get_bool("B1") is False
        assert adapter.get_bool("B2") is False
        assert adapter.get_bool("B3") is False
        assert adapter.get_bool("B4") is False
        assert adapter.get_bool("B5") is False
    
    def test_get_bool_missing_with_default(self):
        """Test getting missing boolean with default."""
        adapter = DictEnvAdapter({})
        assert adapter.get_bool("MISSING", default=True) is True
        assert adapter.get_bool("MISSING", default=False) is False
    
    def test_set(self):
        """Test setting environment variable."""
        adapter = DictEnvAdapter({})
        adapter.set("NEW_KEY", "new_value")
        assert adapter.get("NEW_KEY") == "new_value"


class TestSystemClockAdapter:
    """Test system clock adapter."""
    
    def test_now(self):
        """Test getting current datetime."""
        adapter = SystemClockAdapter()
        before = datetime.now()
        result = adapter.now()
        after = datetime.now()
        
        assert before <= result <= after
    
    def test_now_iso(self):
        """Test getting current datetime as ISO string."""
        adapter = SystemClockAdapter()
        iso_str = adapter.now_iso()
        
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(iso_str)
        assert isinstance(parsed, datetime)
    
    def test_timestamp(self):
        """Test getting current timestamp."""
        adapter = SystemClockAdapter()
        before = datetime.now().timestamp()
        result = adapter.timestamp()
        after = datetime.now().timestamp()
        
        assert before <= result <= after


class TestFixedClockAdapter:
    """Test fixed clock adapter for deterministic testing."""
    
    def test_fixed_time(self):
        """Test that clock returns fixed time."""
        fixed_dt = datetime(2024, 1, 1, 12, 0, 0)
        adapter = FixedClockAdapter(fixed_dt)
        
        # Should return same time multiple times
        assert adapter.now() == fixed_dt
        assert adapter.now() == fixed_dt
    
    def test_now_iso(self):
        """Test ISO format of fixed time."""
        fixed_dt = datetime(2024, 1, 1, 12, 0, 0)
        adapter = FixedClockAdapter(fixed_dt)
        
        iso_str = adapter.now_iso()
        assert iso_str == "2024-01-01T12:00:00"
    
    def test_timestamp(self):
        """Test timestamp of fixed time."""
        fixed_dt = datetime(2024, 1, 1, 12, 0, 0)
        adapter = FixedClockAdapter(fixed_dt)
        
        ts = adapter.timestamp()
        assert ts == fixed_dt.timestamp()
    
    def test_set_time(self):
        """Test updating the fixed time."""
        initial = datetime(2024, 1, 1, 12, 0, 0)
        adapter = FixedClockAdapter(initial)
        
        assert adapter.now() == initial
        
        new_time = datetime(2024, 12, 31, 23, 59, 59)
        adapter.set_time(new_time)
        
        assert adapter.now() == new_time


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
