#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Resource Management Module
Validates memory monitoring and context managers
"""

import pytest
import time
import gc
from pathlib import Path
from resource_management import (
    get_memory_usage_mb,
    memory_profiling_decorator,
    managed_stage_execution,
    MemoryMonitor,
    cleanup_intermediate_data
)


class TestMemoryUtilities:
    """Test basic memory utilities"""
    
    def test_get_memory_usage(self):
        """Test that memory usage is returned as positive float"""
        mem = get_memory_usage_mb()
        assert isinstance(mem, float)
        assert mem > 0
    
    def test_memory_profiling_decorator(self, caplog):
        """Test memory profiling decorator"""
        @memory_profiling_decorator
        def allocate_memory():
            # Allocate some memory
            data = [0] * 1000000
            return len(data)
        
        result = allocate_memory()
        assert result == 1000000
        # Check that memory logging occurred (in caplog or logger)


class TestManagedStageExecution:
    """Test stage execution context manager"""
    
    def test_stage_execution_basic(self, caplog):
        """Test basic stage execution"""
        stage_executed = False
        
        with managed_stage_execution("Test Stage"):
            stage_executed = True
            # Simulate stage work
            data = [0] * 100
        
        assert stage_executed
        # Check that garbage collection was called
    
    def test_stage_execution_with_exception(self):
        """Test that context manager handles exceptions"""
        with pytest.raises(ValueError):
            with managed_stage_execution("Error Stage"):
                raise ValueError("Test error")
    
    def test_stage_execution_gc_optional(self):
        """Test stage execution without forced GC"""
        with managed_stage_execution("No GC Stage", force_gc=False):
            data = [0] * 100
        # Should complete without GC


class TestMemoryMonitor:
    """Test memory monitoring"""
    
    def test_memory_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = MemoryMonitor()
        assert monitor.initial_memory > 0
        assert monitor.peak_memory > 0
    
    def test_memory_monitor_check(self):
        """Test memory check"""
        monitor = MemoryMonitor()
        
        # Allocate some memory
        data = [0] * 1000000
        current = monitor.check("After allocation")
        
        assert current > 0
        assert monitor.peak_memory >= monitor.initial_memory
        
        # Clean up
        del data
        gc.collect()
    
    def test_memory_monitor_report(self):
        """Test memory report generation"""
        monitor = MemoryMonitor()
        
        # Do some work
        data = [0] * 100000
        monitor.check("Work done")
        del data
        
        report = monitor.report()
        assert 'initial_mb' in report
        assert 'final_mb' in report
        assert 'peak_mb' in report
        assert 'total_delta_mb' in report
        assert 'peak_delta_mb' in report
        
        assert report['initial_mb'] > 0
        assert report['final_mb'] > 0
        assert report['peak_mb'] >= report['initial_mb']
    
    def test_memory_monitor_log_interval(self, caplog):
        """Test that monitor respects log interval"""
        monitor = MemoryMonitor(log_interval_mb=1000.0)
        
        # Small allocation shouldn't trigger warning
        data = [0] * 100
        monitor.check("Small allocation")
        
        del data


class TestCleanupUtilities:
    """Test cleanup utilities"""
    
    def test_cleanup_intermediate_data(self):
        """Test cleanup of intermediate data"""
        data1 = [0] * 100000
        data2 = [0] * 100000
        data3 = [0] * 100000
        
        cleanup_intermediate_data(data1, data2, data3)
        # Objects should be deleted and GC called
        # Can't directly test deletion, but can verify no exceptions


class TestIntegrationScenarios:
    """Integration tests for resource management"""
    
    def test_pipeline_stage_simulation(self):
        """Simulate a pipeline stage execution"""
        monitor = MemoryMonitor()
        
        with managed_stage_execution("Stage 1"):
            # Simulate document loading
            documents = ["doc" * 1000 for _ in range(100)]
            monitor.check("Documents loaded")
        
        with managed_stage_execution("Stage 2"):
            # Simulate processing
            results = [{"data": i} for i in range(1000)]
            monitor.check("Processing complete")
        
        report = monitor.report()
        assert report['final_mb'] > 0
    
    def test_nested_context_managers(self):
        """Test nested context managers work correctly"""
        with managed_stage_execution("Outer Stage"):
            with managed_stage_execution("Inner Stage"):
                data = [0] * 1000
        # Should complete without errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
