#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resource Management Module
Provides context managers and utilities for managing memory and resources
"""

import gc
import logging
import psutil
import os
from contextlib import contextmanager
from typing import Any, Optional, Callable, Generator, Dict
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_bytes: int = process.memory_info().rss
    return float(memory_bytes / 1024 / 1024)


def memory_profiling_decorator(func: Callable) -> Callable:
    """
    Decorator that logs memory usage before and after function execution
    Useful for identifying memory leaks in critical functions
    
    Usage:
        @memory_profiling_decorator
        def heavy_operation():
            # ... code ...
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        mem_before = get_memory_usage_mb()
        logger.debug(f"[MEMORY] {func.__name__} - Before: {mem_before:.2f} MB")
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            mem_after = get_memory_usage_mb()
            mem_delta = mem_after - mem_before
            logger.info(
                f"[MEMORY] {func.__name__} - After: {mem_after:.2f} MB "
                f"(Δ {mem_delta:+.2f} MB)"
            )
    
    return wrapper


@contextmanager
def managed_stage_execution(stage_name: str, force_gc: bool = True) -> Generator[None, None, None]:
    """
    Context manager for stage execution with automatic resource cleanup
    
    Usage:
        with managed_stage_execution("Stage 4"):
            # ... stage code ...
    
    Args:
        stage_name: Name of the stage for logging
        force_gc: Whether to force garbage collection after stage
    """
    mem_before = get_memory_usage_mb()
    logger.info(f"[STAGE] {stage_name} starting - Memory: {mem_before:.2f} MB")
    
    try:
        yield
    finally:
        if force_gc:
            # Force garbage collection of intermediate objects
            collected = gc.collect()
            logger.debug(f"[GC] Collected {collected} objects after {stage_name}")
        
        mem_after = get_memory_usage_mb()
        mem_delta = mem_after - mem_before
        logger.info(
            f"[STAGE] {stage_name} completed - Memory: {mem_after:.2f} MB "
            f"(Δ {mem_delta:+.2f} MB)"
        )


@contextmanager
def heavy_document_loader(pdf_path: Path) -> Generator[Any, None, None]:
    """
    Context manager for loading heavy documents
    Automatically releases resources when done
    
    Usage:
        with heavy_document_loader(pdf_path) as doc:
            # ... process document ...
    
    Note: This is a template. Actual implementation should be integrated
    with the specific PDF processor being used (e.g., PyMuPDF)
    """
    import fitz  # PyMuPDF
    
    doc = None
    try:
        doc = fitz.open(str(pdf_path))
        logger.debug(f"Document loaded: {pdf_path.name}")
        yield doc
    finally:
        if doc is not None:
            doc.close()
            logger.debug(f"Document closed: {pdf_path.name}")
        # Force cleanup
        gc.collect()


def cleanup_intermediate_data(*objects: Any) -> None:
    """
    Explicitly delete and clean up intermediate data objects
    
    Args:
        *objects: Objects to delete and clean up
    """
    for obj in objects:
        del obj
    gc.collect()


class MemoryMonitor:
    """
    Monitor memory usage throughout pipeline execution
    Useful for detecting memory leaks and optimization
    """
    
    def __init__(self, log_interval_mb: float = 100.0):
        """
        Args:
            log_interval_mb: Log warning if memory increases by this amount
        """
        self.initial_memory = get_memory_usage_mb()
        self.last_check = self.initial_memory
        self.log_interval = log_interval_mb
        self.peak_memory = self.initial_memory
        logger.info(f"[MONITOR] Memory monitoring started at {self.initial_memory:.2f} MB")
    
    def check(self, label: str = "") -> float:
        """Check current memory and log if significant change"""
        current_memory = get_memory_usage_mb()
        delta = current_memory - self.last_check
        total_delta = current_memory - self.initial_memory
        
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        if abs(delta) >= self.log_interval:
            logger.warning(
                f"[MONITOR] {label} - Memory: {current_memory:.2f} MB "
                f"(Δ {delta:+.2f} MB, Total Δ {total_delta:+.2f} MB)"
            )
        
        self.last_check = current_memory
        return current_memory
    
    def report(self) -> Dict[str, float]:
        """Generate final memory report"""
        final_memory = get_memory_usage_mb()
        report = {
            'initial_mb': self.initial_memory,
            'final_mb': final_memory,
            'peak_mb': self.peak_memory,
            'total_delta_mb': final_memory - self.initial_memory,
            'peak_delta_mb': self.peak_memory - self.initial_memory
        }
        
        logger.info(
            f"[MONITOR] Memory Report - "
            f"Initial: {report['initial_mb']:.2f} MB, "
            f"Final: {report['final_mb']:.2f} MB, "
            f"Peak: {report['peak_mb']:.2f} MB, "
            f"Peak Δ: {report['peak_delta_mb']:+.2f} MB"
        )
        
        return report
