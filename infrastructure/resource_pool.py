#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resource Pool Manager - F4.3 Implementation
Manages GPU/CPU worker pool with timeout and memory limit enforcement

Implements:
- Worker pool management with async context manager
- Worker timeout monitoring and enforcement
- Memory limit monitoring and enforcement
- Task tracking and cleanup
- Governance Standard compliance for resource management
"""

import asyncio
import logging
import time
import psutil
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# ============================================================================
# Exceptions
# ============================================================================

class WorkerTimeoutError(Exception):
    """Raised when a worker exceeds the configured timeout"""
    pass


class WorkerMemoryError(Exception):
    """Raised when a worker exceeds the configured memory limit"""
    pass


# ============================================================================
# Data Structures
# ============================================================================

class DeviceType(str, Enum):
    """Type of compute device"""
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class ResourceConfig:
    """Configuration for resource pool"""
    max_workers: int
    worker_timeout_secs: int
    worker_memory_mb: int
    devices: List[str] = field(default_factory=lambda: ["cpu"])
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")
        if self.worker_timeout_secs <= 0:
            raise ValueError(f"worker_timeout_secs must be positive, got {self.worker_timeout_secs}")
        if self.worker_memory_mb <= 0:
            raise ValueError(f"worker_memory_mb must be positive, got {self.worker_memory_mb}")
        if not self.devices:
            self.devices = ["cpu"]


@dataclass
class Worker:
    """Computational worker with resource limits"""
    id: int
    device: str
    memory_limit_mb: int
    process: Optional[Any] = None
    _killed: bool = False
    
    def get_memory_usage_mb(self) -> float:
        """
        Get current memory usage of worker process in MB.
        
        Returns:
            Memory usage in megabytes
        """
        if self.process is not None:
            try:
                # Get memory info from process
                mem_info = self.process.memory_info()
                return mem_info.rss / (1024 * 1024)  # Convert bytes to MB
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return 0.0
        
        # Fallback: estimate based on current process memory
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            # Divide by max_workers as rough estimate
            return mem_info.rss / (1024 * 1024) * 0.1
        except Exception:
            return 0.0
    
    def kill(self):
        """
        Kill the worker process.
        
        This is called when worker exceeds timeout or memory limits.
        Implements governance standard for resource enforcement.
        """
        self._killed = True
        if self.process is not None:
            try:
                self.process.terminate()
                # Give it time to terminate gracefully
                try:
                    self.process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    # Force kill if graceful termination fails
                    self.process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    async def run_mcmc_sampling(self, link: Any) -> Any:
        """
        Execute MCMC sampling on assigned device.
        
        This is a placeholder for actual MCMC sampling implementation.
        In production, this would call the actual Bayesian inference engine.
        
        Args:
            link: Causal link to analyze
            
        Returns:
            Mechanism result from inference
        """
        # Simulate MCMC sampling work
        await asyncio.sleep(0.1)
        
        # Return mock result
        return {
            'type': 'técnico',
            'posterior_mean': 0.75,
            'necessity_test': {'passed': True, 'missing': []},
            'device': self.device,
            'worker_id': self.id
        }


# ============================================================================
# Resource Pool
# ============================================================================

class ResourcePool:
    """
    Manages pool of computational resources (GPU/CPU workers).
    Implements worker timeout and memory limits (Governance Standard).
    
    Features:
    - Pre-populated worker pool with configurable size
    - Async context manager for safe resource acquisition
    - Automatic timeout enforcement
    - Memory limit enforcement
    - Task tracking and monitoring
    - Automatic cleanup on context exit
    """
    
    def __init__(self, config: ResourceConfig):
        """
        Initialize resource pool with configuration.
        
        Args:
            config: Resource pool configuration
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_workers = config.max_workers
        self.worker_timeout_secs = config.worker_timeout_secs
        self.worker_memory_mb = config.worker_memory_mb
        
        # Worker pool and tracking
        self.available_workers = asyncio.Queue(maxsize=self.max_workers)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self._monitor_tasks: Dict[str, asyncio.Task] = {}
        
        # Pre-populate pool
        self.logger.info(f"Initializing resource pool with {self.max_workers} workers")
        for i in range(self.max_workers):
            worker = Worker(
                id=i,
                device=config.devices[i % len(config.devices)],
                memory_limit_mb=self.worker_memory_mb
            )
            # Use try-except to handle queue operations safely
            try:
                self.available_workers.put_nowait(worker)
            except asyncio.QueueFull:
                self.logger.warning(f"Queue full when adding worker {i}")
        
        self.logger.info(f"Resource pool initialized: {self.max_workers} workers on devices {config.devices}")
    
    @asynccontextmanager
    async def acquire_worker(self, task_id: str):
        """
        Acquire worker from pool with timeout and monitoring.
        
        This implements an async context manager pattern for safe resource handling.
        The worker is automatically returned to the pool when the context exits.
        
        Args:
            task_id: Unique identifier for the task
            
        Yields:
            Worker: Available worker from the pool
            
        Raises:
            asyncio.TimeoutError: If no worker available within 30 seconds
            WorkerTimeoutError: If worker exceeds configured timeout
            WorkerMemoryError: If worker exceeds configured memory limit
        """
        worker = None
        monitor_task = None
        
        try:
            # Wait for available worker with timeout
            self.logger.debug(f"Task {task_id} waiting for worker...")
            worker = await asyncio.wait_for(
                self.available_workers.get(),
                timeout=30.0
            )
            
            self.logger.info(f"Task {task_id} acquired worker {worker.id} on device {worker.device}")
            
            # Track active task
            self.active_tasks[task_id] = {
                'worker': worker,
                'start_time': time.time()
            }
            
            # Start monitoring worker
            monitor_task = asyncio.create_task(
                self._monitor_worker(task_id, worker)
            )
            self._monitor_tasks[task_id] = monitor_task
            
            yield worker
            
        finally:
            # Cleanup: cancel monitoring and return worker
            if monitor_task is not None:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Remove from tracking
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            if task_id in self._monitor_tasks:
                del self._monitor_tasks[task_id]
            
            # Return worker to pool if it wasn't killed
            if worker is not None and not worker._killed:
                try:
                    self.available_workers.put_nowait(worker)
                    self.logger.debug(f"Worker {worker.id} returned to pool")
                except asyncio.QueueFull:
                    self.logger.warning(f"Queue full when returning worker {worker.id}")
    
    async def _monitor_worker(self, task_id: str, worker: Worker):
        """
        Monitor worker and enforce timeout/memory limits.
        
        This runs as a background task and checks the worker every second.
        If limits are exceeded, the worker is killed and an exception is raised.
        
        Args:
            task_id: Task identifier for logging
            worker: Worker to monitor
            
        Raises:
            WorkerTimeoutError: If worker exceeds timeout
            WorkerMemoryError: If worker exceeds memory limit
        """
        start_time = time.time()
        
        while True:
            await asyncio.sleep(1.0)
            
            elapsed = time.time() - start_time
            memory_mb = worker.get_memory_usage_mb()
            
            # Timeout check
            if elapsed > self.worker_timeout_secs:
                self.logger.critical(
                    f"Worker timeout exceeded: {task_id} "
                    f"(elapsed: {elapsed:.1f}s, limit: {self.worker_timeout_secs}s)"
                )
                worker.kill()
                # Omission flagging (Governance Standard)
                raise WorkerTimeoutError(
                    f"Worker exceeded {self.worker_timeout_secs}s timeout"
                )
            
            # Memory check
            if memory_mb > self.worker_memory_mb:
                self.logger.critical(
                    f"Worker memory limit exceeded: {task_id} "
                    f"(usage: {memory_mb:.1f}MB, limit: {self.worker_memory_mb}MB)"
                )
                worker.kill()
                raise WorkerMemoryError(
                    f"Worker exceeded {self.worker_memory_mb}MB memory limit"
                )
            
            # Log periodic status (every 10 seconds)
            if int(elapsed) % 10 == 0:
                self.logger.debug(
                    f"Worker {worker.id} status - Task: {task_id}, "
                    f"Elapsed: {elapsed:.1f}s, Memory: {memory_mb:.1f}MB"
                )
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get current status of the resource pool.
        
        Returns:
            Dictionary with pool statistics
        """
        return {
            'total_workers': self.max_workers,
            'available_workers': self.available_workers.qsize(),
            'active_tasks': len(self.active_tasks),
            'worker_timeout_secs': self.worker_timeout_secs,
            'worker_memory_mb': self.worker_memory_mb,
            'active_task_ids': list(self.active_tasks.keys())
        }


# ============================================================================
# Bayesian Engine Integration
# ============================================================================

class BayesianInferenceEngine:
    """
    Bayesian inference engine with resource management.
    
    This demonstrates how to integrate the ResourcePool with the
    Bayesian inference engine for mechanism inference.
    """
    
    def __init__(self, resource_pool: ResourcePool):
        """
        Initialize with resource pool.
        
        Args:
            resource_pool: ResourcePool instance for worker management
        """
        self.resource_pool = resource_pool
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def infer_mechanism(self, link: Any) -> Dict[str, Any]:
        """
        Execute inference with resource management.
        
        Args:
            link: Causal link to analyze
            
        Returns:
            Mechanism inference result
            
        Raises:
            WorkerTimeoutError: If inference exceeds timeout
            WorkerMemoryError: If inference exceeds memory limit
        """
        # Handle both dict and object links
        if hasattr(link, 'id'):
            link_id = link.id
        elif isinstance(link, dict) and 'id' in link:
            link_id = link['id']
        else:
            cause_id = getattr(link, 'cause_id', link.get('cause_id', 'unknown') if isinstance(link, dict) else 'unknown')
            effect_id = getattr(link, 'effect_id', link.get('effect_id', 'unknown') if isinstance(link, dict) else 'unknown')
            link_id = f"{cause_id}-{effect_id}"
        
        task_id = f"infer_{link_id}"
        
        self.logger.info(f"Starting mechanism inference for {task_id}")
        
        async with self.resource_pool.acquire_worker(task_id) as worker:
            # Worker has GPU/CPU assigned, memory monitored
            self.logger.debug(
                f"Worker {worker.id} executing inference on device {worker.device}"
            )
            
            result = await worker.run_mcmc_sampling(link)
            
            self.logger.info(
                f"Mechanism inference complete for {task_id}: "
                f"type={result['type']}, posterior_mean={result['posterior_mean']:.3f}"
            )
            
            return result
