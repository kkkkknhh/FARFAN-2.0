#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Orchestrator with Backpressure Signaling - Audit Point 4.2
================================================================

⚠️ DEPRECATION NOTICE ⚠️
========================
This orchestrator has been DEPRECATED and consolidated into:
  orchestration/unified_orchestrator.py

Migration Guide:
- Async orchestration patterns are now built into UnifiedOrchestrator
- Backpressure signaling is handled by infrastructure/async_orchestrator.py
- Use UnifiedOrchestrator for all new orchestration needs

Implements asynchronous orchestration with queue management and backpressure
signaling for high-cost operations (Bayesian/GNN inference).

Design Principles:
- Queue-based job management with configurable size limits
- HTTP 503 backpressure signaling when queue is full
- Deque for efficient queue operations
- Job tracking and metrics for observability
- Graceful degradation under load

Audit Point 4.2 Compliance:
- Orchestrator signals HTTP 503 when queue > queue_size (e.g., 100)
- Uses deque for queue management
- Backpressure logging and metrics
- Flow control for scalable causal systems

Author: AI Systems Architect
Version: 1.0.0 (DEPRECATED)
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

# ============================================================================
# Exceptions
# ============================================================================


class QueueFullError(Exception):
    """Raised when job queue is full and cannot accept new jobs"""

    def __init__(self, queue_size: int, message: str = "Job queue is full"):
        self.queue_size = queue_size
        self.http_status = 503  # Service Unavailable
        super().__init__(f"{message} (size: {queue_size})")


class JobTimeoutError(Exception):
    """Raised when a job exceeds its timeout limit"""

    pass


# ============================================================================
# Data Structures
# ============================================================================


class JobStatus(str, Enum):
    """Status of a job in the queue"""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class JobMetrics:
    """Metrics for a single job"""

    job_id: str
    status: JobStatus
    queued_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    queue_wait_time: Optional[float] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None


@dataclass
class OrchestratorConfig:
    """Configuration for async orchestrator"""

    queue_size: int = 100
    max_workers: int = 5
    job_timeout_secs: int = 300
    enable_backpressure: bool = True
    log_backpressure: bool = True

    def __post_init__(self):
        """Validate configuration"""
        if self.queue_size <= 0:
            raise ValueError(f"queue_size must be positive, got {self.queue_size}")
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")
        if self.job_timeout_secs <= 0:
            raise ValueError(
                f"job_timeout_secs must be positive, got {self.job_timeout_secs}"
            )


@dataclass
class OrchestratorMetrics:
    """Metrics for orchestrator performance"""

    total_jobs_submitted: int = 0
    total_jobs_completed: int = 0
    total_jobs_failed: int = 0
    total_jobs_timeout: int = 0
    total_jobs_rejected: int = 0
    current_queue_size: int = 0
    peak_queue_size: int = 0
    avg_queue_wait_time: float = 0.0
    avg_execution_time: float = 0.0
    backpressure_events: int = 0


# ============================================================================
# Async Orchestrator
# ============================================================================


class AsyncOrchestrator:
    """
    Asynchronous orchestrator with backpressure signaling.

    Manages a queue of high-cost async operations (e.g., Bayesian inference,
    GNN processing) with configurable limits and backpressure signaling.

    Features:
    - Deque-based job queue with size limits
    - HTTP 503 signaling when queue is full
    - Job timeout enforcement
    - Metrics and observability
    - Graceful degradation under load

    Args:
        config: Orchestrator configuration

    Example:
        >>> config = OrchestratorConfig(queue_size=50, max_workers=3)
        >>> orchestrator = AsyncOrchestrator(config)
        >>> await orchestrator.start()
        >>> try:
        >>>     result = await orchestrator.submit_job(my_async_func, arg1, arg2)
        >>> except QueueFullError as e:
        >>>     # Handle backpressure (return HTTP 503)
        >>>     return {"error": "Service busy", "status": e.http_status}
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialize async orchestrator.

        Args:
            config: Orchestrator configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Job queue (deque for efficient FIFO operations)
        self.job_queue: deque = deque(maxlen=config.queue_size)
        self.job_tasks: Dict[str, asyncio.Task] = {}
        self.job_metrics: Dict[str, JobMetrics] = {}

        # Orchestrator metrics
        self.metrics = OrchestratorMetrics()

        # Worker pool
        self.worker_semaphore = asyncio.Semaphore(config.max_workers)
        self.workers: list[asyncio.Task] = []

        # State tracking
        self.is_running = False
        self._shutdown_event = asyncio.Event()

        self.logger.info(
            f"AsyncOrchestrator initialized: queue_size={config.queue_size}, "
            f"max_workers={config.max_workers}, timeout={config.job_timeout_secs}s"
        )

    def start(self):
        """
        Start the orchestrator and worker pool.

        This must be called before submitting jobs.
        """
        if self.is_running:
            self.logger.warning("Orchestrator already running")
            return

        self.is_running = True
        self._shutdown_event.clear()

        # Start worker tasks
        for i in range(self.config.max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self.workers.append(worker)

        self.logger.info(f"Orchestrator started with {self.config.max_workers} workers")

    async def shutdown(self):
        """
        Gracefully shutdown the orchestrator.

        Waits for all running jobs to complete before shutting down.
        """
        if not self.is_running:
            return

        self.logger.info("Shutting down orchestrator...")
        self.is_running = False
        self._shutdown_event.set()

        # Wait for all workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        # Cancel any remaining tasks
        for task in self.job_tasks.values():
            if not task.done():
                task.cancel()

        self.logger.info("Orchestrator shutdown complete")

    async def submit_job(
        self,
        func: Callable,
        *args,
        job_id: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Submit a job to the orchestrator queue.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            job_id: Optional job identifier (auto-generated if not provided)
            timeout: Optional timeout override (uses config default if not provided)
            **kwargs: Keyword arguments for func

        Returns:
            Result from the executed function

        Raises:
            QueueFullError: If queue is full (HTTP 503 backpressure)
            JobTimeoutError: If job exceeds timeout
            Exception: Re-raises exceptions from func
        """
        if not self.is_running:
            raise RuntimeError("Orchestrator not started - call start() first")

        # Generate job ID if not provided
        if job_id is None:
            job_id = f"job_{int(time.time() * 1000)}_{len(self.job_metrics)}"

        # Check queue capacity (backpressure point)
        current_queue_size = len(self.job_queue)
        if current_queue_size >= self.config.queue_size:
            self.metrics.total_jobs_rejected += 1
            self.metrics.backpressure_events += 1

            if self.config.log_backpressure:
                self.logger.warning(
                    f"BACKPRESSURE: Queue full ({current_queue_size}/{self.config.queue_size}) - "
                    f"rejecting job {job_id} with HTTP 503"
                )

            raise QueueFullError(
                queue_size=self.config.queue_size,
                message=f"Job queue full - cannot accept job {job_id}",
            )

        # Create job metrics
        job_timeout = timeout or self.config.job_timeout_secs
        job_metric = JobMetrics(
            job_id=job_id, status=JobStatus.QUEUED, queued_at=time.time()
        )
        self.job_metrics[job_id] = job_metric

        # Add job to queue
        job = {
            "job_id": job_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "timeout": job_timeout,
            "result_future": asyncio.Future(),
        }
        self.job_queue.append(job)

        # Update metrics
        self.metrics.total_jobs_submitted += 1
        self.metrics.current_queue_size = len(self.job_queue)
        self.metrics.peak_queue_size = max(
            self.metrics.peak_queue_size, self.metrics.current_queue_size
        )

        self.logger.debug(
            f"Job {job_id} queued (queue: {len(self.job_queue)}/{self.config.queue_size})"
        )

        # Wait for result
        return await job["result_future"]

    async def _worker_loop(self, worker_id: int):
        """
        Worker loop that processes jobs from the queue.

        Args:
            worker_id: Unique identifier for this worker
        """
        self.logger.info(f"Worker {worker_id} started")

        while self.is_running or len(self.job_queue) > 0:
            try:
                # Get job from queue (non-blocking)
                if not self.job_queue:
                    await asyncio.sleep(0.1)  # Avoid busy waiting
                    continue

                job = self.job_queue.popleft()
                job_id = job["job_id"]

                # Update metrics
                self.metrics.current_queue_size = len(self.job_queue)
                job_metric = self.job_metrics[job_id]
                job_metric.started_at = time.time()
                job_metric.queue_wait_time = (
                    job_metric.started_at - job_metric.queued_at
                )
                job_metric.status = JobStatus.RUNNING

                self.logger.info(
                    f"Worker {worker_id} processing job {job_id} "
                    f"(waited {job_metric.queue_wait_time:.2f}s)"
                )

                # Execute job with timeout
                try:
                    async with self.worker_semaphore:
                        result = await asyncio.wait_for(
                            job["func"](*job["args"], **job["kwargs"]),
                            timeout=job["timeout"],
                        )

                    # Job completed successfully
                    job_metric.completed_at = time.time()
                    job_metric.execution_time = (
                        job_metric.completed_at - job_metric.started_at
                    )
                    job_metric.status = JobStatus.COMPLETED

                    self.metrics.total_jobs_completed += 1
                    self._update_avg_metrics()

                    self.logger.info(
                        f"Job {job_id} completed in {job_metric.execution_time:.2f}s"
                    )

                    job["result_future"].set_result(result)

                except asyncio.TimeoutError:
                    # Job timeout
                    job_metric.completed_at = time.time()
                    job_metric.execution_time = (
                        job_metric.completed_at - job_metric.started_at
                    )
                    job_metric.status = JobStatus.TIMEOUT
                    job_metric.error = f"Job exceeded {job['timeout']}s timeout"

                    self.metrics.total_jobs_timeout += 1

                    self.logger.error(f"Job {job_id} timeout after {job['timeout']}s")

                    job["result_future"].set_exception(
                        JobTimeoutError(job_metric.error)
                    )

                except Exception as e:
                    # Job failed
                    job_metric.completed_at = time.time()
                    job_metric.execution_time = (
                        job_metric.completed_at - job_metric.started_at
                    )
                    job_metric.status = JobStatus.FAILED
                    job_metric.error = str(e)

                    self.metrics.total_jobs_failed += 1

                    self.logger.error(
                        f"Job {job_id} failed: {type(e).__name__}: {str(e)}"
                    )

                    job["result_future"].set_exception(e)

            except Exception as e:
                self.logger.error(
                    f"Worker {worker_id} error: {type(e).__name__}: {str(e)}",
                    exc_info=True,
                )

        self.logger.info(f"Worker {worker_id} stopped")

    def _update_avg_metrics(self):
        """Update average metrics from completed jobs"""
        completed_jobs = [
            m
            for m in self.job_metrics.values()
            if m.status == JobStatus.COMPLETED and m.queue_wait_time is not None
        ]

        if completed_jobs:
            self.metrics.avg_queue_wait_time = sum(
                j.queue_wait_time for j in completed_jobs
            ) / len(completed_jobs)
            self.metrics.avg_execution_time = sum(
                j.execution_time for j in completed_jobs if j.execution_time
            ) / len(completed_jobs)

    def get_metrics(self) -> OrchestratorMetrics:
        """
        Get current orchestrator metrics.

        Returns:
            OrchestratorMetrics with current statistics
        """
        self.metrics.current_queue_size = len(self.job_queue)
        return self.metrics

    def get_job_status(self, job_id: str) -> Optional[JobMetrics]:
        """
        Get status of a specific job.

        Args:
            job_id: Job identifier

        Returns:
            JobMetrics if job exists, None otherwise
        """
        return self.job_metrics.get(job_id)

    def get_queue_info(self) -> Dict[str, Any]:
        """
        Get current queue information.

        Returns:
            Dictionary with queue statistics
        """
        return {
            "current_size": len(self.job_queue),
            "max_size": self.config.queue_size,
            "utilization": len(self.job_queue) / self.config.queue_size,
            "is_full": len(self.job_queue) >= self.config.queue_size,
            "backpressure_active": len(self.job_queue) >= self.config.queue_size,
        }


# ============================================================================
# Factory Functions
# ============================================================================


def create_orchestrator(
    queue_size: int = 100,
    max_workers: int = 5,
    job_timeout_secs: int = 300,
    **kwargs,
) -> AsyncOrchestrator:
    """
    Factory function to create AsyncOrchestrator with sensible defaults.

    Args:
        queue_size: Maximum queue size (default: 100)
        max_workers: Maximum concurrent workers (default: 5)
        job_timeout_secs: Job timeout in seconds (default: 300)
        **kwargs: Additional configuration options

    Returns:
        Configured AsyncOrchestrator instance
    """
    config = OrchestratorConfig(
        queue_size=queue_size,
        max_workers=max_workers,
        job_timeout_secs=job_timeout_secs,
        **kwargs,
    )
    return AsyncOrchestrator(config)
