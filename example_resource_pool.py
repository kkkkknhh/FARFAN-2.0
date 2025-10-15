#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of Resource Pool Manager (F4.3)
Demonstrates resource pool configuration and usage with Bayesian inference
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from infrastructure.resource_pool import (
    BayesianInferenceEngine,
    ResourceConfig,
    ResourcePool,
    WorkerMemoryError,
    WorkerTimeoutError,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def example_basic_usage():
    """Example: Basic resource pool usage"""
    print("\n" + "=" * 70)
    print("  Example 1: Basic Resource Pool Usage")
    print("=" * 70)

    # Create configuration
    config = ResourceConfig(
        max_workers=3,
        worker_timeout_secs=60,
        worker_memory_mb=1024,
        devices=["cpu", "cpu", "cpu"],
    )

    # Initialize pool
    pool = ResourcePool(config)
    print(f"\n✓ Resource pool created with {config.max_workers} workers")

    # Check status
    status = pool.get_pool_status()
    print(f"✓ Pool status: {status}")

    # Acquire and use a worker
    async with pool.acquire_worker("example_task") as worker:
        print(f"\n✓ Acquired worker {worker.id} on device {worker.device}")

        # Simulate work
        await asyncio.sleep(0.5)

        # Check memory usage
        memory_mb = worker.get_memory_usage_mb()
        print(f"✓ Worker memory usage: {memory_mb:.2f} MB")

    print("✓ Worker returned to pool\n")


async def example_bayesian_inference():
    """Example: Using resource pool with Bayesian inference"""
    print("\n" + "=" * 70)
    print("  Example 2: Bayesian Inference with Resource Pool")
    print("=" * 70)

    # Create configuration for inference workload
    config = ResourceConfig(
        max_workers=2,
        worker_timeout_secs=120,
        worker_memory_mb=2048,
        devices=["cpu", "cpu"],
    )

    # Initialize pool and engine
    pool = ResourcePool(config)
    engine = BayesianInferenceEngine(pool)
    print(
        f"\n✓ Bayesian inference engine initialized with {config.max_workers} workers"
    )

    # Create mock causal links
    @dataclass
    class CausalLink:
        id: str
        cause_id: str
        effect_id: str

    links = [
        CausalLink(
            id="link_1",
            cause_id="programa_capacitacion",
            effect_id="mejora_competencias",
        ),
        CausalLink(
            id="link_2", cause_id="infraestructura_vial", effect_id="conectividad_rural"
        ),
        CausalLink(
            id="link_3", cause_id="sistema_salud", effect_id="cobertura_atencion"
        ),
    ]

    # Run inference on each link
    print("\nRunning mechanism inference:")
    for link in links:
        result = await engine.infer_mechanism(link)
        print(f"\n  Link: {link.cause_id} → {link.effect_id}")
        print(f"    Mechanism type: {result['type']}")
        print(f"    Posterior mean: {result['posterior_mean']:.3f}")
        print(f"    Necessity test: {result['necessity_test']}")
        print(f"    Executed on: Worker {result['worker_id']} ({result['device']})")

    # Check final status
    status = pool.get_pool_status()
    print(f"\n✓ Final pool status: {status}")


async def example_concurrent_inference():
    """Example: Concurrent inference with resource pool"""
    print("\n" + "=" * 70)
    print("  Example 3: Concurrent Inference Tasks")
    print("=" * 70)

    # Configure pool for concurrent workload
    config = ResourceConfig(
        max_workers=4,
        worker_timeout_secs=60,
        worker_memory_mb=1024,
        devices=["cpu"] * 4,
    )

    pool = ResourcePool(config)
    engine = BayesianInferenceEngine(pool)
    print(f"\n✓ Engine ready with {config.max_workers} workers for concurrent tasks")

    # Create multiple causal links
    @dataclass
    class CausalLink:
        id: str
        cause_id: str
        effect_id: str

    links = [
        CausalLink(id=f"link_{i}", cause_id=f"cause_{i}", effect_id=f"effect_{i}")
        for i in range(10)
    ]

    print(f"\nRunning {len(links)} inference tasks concurrently...")

    # Run all inference tasks concurrently
    import time

    start_time = time.time()

    results = await asyncio.gather(*[engine.infer_mechanism(link) for link in links])

    elapsed = time.time() - start_time

    print(f"\n✓ Completed {len(results)} inference tasks in {elapsed:.2f}s")
    print(f"✓ Average time per task: {elapsed / len(results):.2f}s")
    print(f"✓ Throughput: {len(results) / elapsed:.2f} tasks/sec")

    # Summarize results
    mechanism_types = {}
    for result in results:
        mech_type = result["type"]
        mechanism_types[mech_type] = mechanism_types.get(mech_type, 0) + 1

    print(f"\nMechanism type distribution:")
    for mech_type, count in mechanism_types.items():
        print(f"  {mech_type}: {count}")


async def example_error_handling():
    """Example: Error handling with timeouts"""
    print("\n" + "=" * 70)
    print("  Example 4: Error Handling and Timeouts")
    print("=" * 70)

    # Configure with short timeout for demonstration
    config = ResourceConfig(
        max_workers=1,
        worker_timeout_secs=2,
        worker_memory_mb=512,  # Short timeout
    )

    pool = ResourcePool(config)
    print(f"\n✓ Pool created with {config.worker_timeout_secs}s timeout")

    # Simulate task that exceeds timeout
    print("\nSimulating long-running task that will timeout...")

    try:
        async with pool.acquire_worker("long_task") as worker:
            print(f"✓ Acquired worker {worker.id}")
            print("  Sleeping for 3 seconds (exceeds 2s timeout)...")
            await asyncio.sleep(3)
            print("  This line should not be reached")
    except WorkerTimeoutError as e:
        print(f"\n✓ Timeout correctly detected: {e}")

    print("✓ System recovered gracefully from timeout")


async def example_pool_monitoring():
    """Example: Monitoring pool status during operations"""
    print("\n" + "=" * 70)
    print("  Example 5: Real-time Pool Monitoring")
    print("=" * 70)

    config = ResourceConfig(
        max_workers=3, worker_timeout_secs=60, worker_memory_mb=1024
    )

    pool = ResourcePool(config)

    async def monitored_task(task_id: str, duration: float):
        """Task that shows its progress"""
        async with pool.acquire_worker(task_id) as worker:
            print(f"\n  Task {task_id} started on worker {worker.id}")
            await asyncio.sleep(duration)
            print(f"  Task {task_id} completed")

    # Create monitoring task
    async def monitor_pool():
        """Monitor pool status periodically"""
        for _ in range(5):
            await asyncio.sleep(0.3)
            status = pool.get_pool_status()
            print(
                f"\n  [Monitor] Available: {status['available_workers']}/{status['total_workers']}, "
                f"Active: {status['active_tasks']}, "
                f"Tasks: {status['active_task_ids']}"
            )

    print("\nStarting concurrent tasks with monitoring:")

    # Run tasks and monitoring concurrently
    await asyncio.gather(
        monitored_task("task_A", 0.8),
        monitored_task("task_B", 1.0),
        monitored_task("task_C", 0.6),
        monitor_pool(),
    )

    print("\n✓ All tasks completed")

    # Final status
    status = pool.get_pool_status()
    print(f"✓ Final status: All {status['available_workers']} workers available")


async def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("  RESOURCE POOL MANAGER - EXAMPLE USAGE")
    print("  F4.3: GPU/CPU Resource Pool with Timeout & Memory Limits")
    print("=" * 70)

    await example_basic_usage()
    await example_bayesian_inference()
    await example_concurrent_inference()

    # Optional: Uncomment to see error handling (takes 2-3 seconds)
    # await example_error_handling()

    await example_pool_monitoring()

    print("\n" + "=" * 70)
    print("  ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Resource pool initialization and configuration")
    print("  ✓ Worker acquisition with async context manager")
    print("  ✓ Integration with Bayesian inference engine")
    print("  ✓ Concurrent task execution with resource limits")
    print("  ✓ Real-time pool monitoring and status tracking")
    print("  ✓ Automatic cleanup and worker return")
    print("\nGovernance Standard Compliance:")
    print("  ✓ Worker timeout enforcement")
    print("  ✓ Memory limit monitoring")
    print("  ✓ Task tracking and audit trail")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
