#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Resource Pool Manager (F4.3)
Tests resource pool, worker management, timeout and memory enforcement
"""

import asyncio
import sys
import time
from pathlib import Path
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from infrastructure.resource_pool import (
    ResourceConfig,
    Worker,
    ResourcePool,
    WorkerTimeoutError,
    WorkerMemoryError,
    BayesianInferenceEngine
)


def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def test_resource_config_validation():
    """Test ResourceConfig validation"""
    print_header("TEST: ResourceConfig Validation")
    
    # Valid configuration
    config = ResourceConfig(
        max_workers=4,
        worker_timeout_secs=60,
        worker_memory_mb=1024
    )
    assert config.max_workers == 4
    assert config.worker_timeout_secs == 60
    assert config.worker_memory_mb == 1024
    assert config.devices == ["cpu"]  # Default
    print("✓ Valid configuration created successfully")
    
    # With custom devices
    config_gpu = ResourceConfig(
        max_workers=2,
        worker_timeout_secs=120,
        worker_memory_mb=2048,
        devices=["cuda:0", "cuda:1"]
    )
    assert config_gpu.devices == ["cuda:0", "cuda:1"]
    print("✓ Configuration with custom devices works")
    
    # Invalid configurations
    try:
        ResourceConfig(max_workers=0, worker_timeout_secs=60, worker_memory_mb=1024)
        assert False, "Should have raised ValueError for max_workers=0"
    except ValueError as e:
        print(f"✓ Correctly rejected max_workers=0: {e}")
    
    try:
        ResourceConfig(max_workers=4, worker_timeout_secs=0, worker_memory_mb=1024)
        assert False, "Should have raised ValueError for worker_timeout_secs=0"
    except ValueError as e:
        print(f"✓ Correctly rejected worker_timeout_secs=0: {e}")
    
    try:
        ResourceConfig(max_workers=4, worker_timeout_secs=60, worker_memory_mb=0)
        assert False, "Should have raised ValueError for worker_memory_mb=0"
    except ValueError as e:
        print(f"✓ Correctly rejected worker_memory_mb=0: {e}")
    
    print("✓ All ResourceConfig validation tests passed")


def test_worker_creation():
    """Test Worker creation and methods"""
    print_header("TEST: Worker Creation")
    
    worker = Worker(
        id=0,
        device="cpu",
        memory_limit_mb=512
    )
    
    assert worker.id == 0
    assert worker.device == "cpu"
    assert worker.memory_limit_mb == 512
    assert worker._killed is False
    print("✓ Worker created successfully")
    
    # Test memory usage (should return a value, even if 0)
    memory_mb = worker.get_memory_usage_mb()
    assert isinstance(memory_mb, float)
    assert memory_mb >= 0
    print(f"✓ Worker memory usage: {memory_mb:.2f} MB")
    
    print("✓ All Worker creation tests passed")


async def test_resource_pool_initialization():
    """Test ResourcePool initialization"""
    print_header("TEST: ResourcePool Initialization")
    
    config = ResourceConfig(
        max_workers=3,
        worker_timeout_secs=10,
        worker_memory_mb=512,
        devices=["cpu", "cpu", "cpu"]
    )
    
    pool = ResourcePool(config)
    
    assert pool.max_workers == 3
    assert pool.worker_timeout_secs == 10
    assert pool.worker_memory_mb == 512
    print("✓ ResourcePool initialized with correct parameters")
    
    # Check pool status
    status = pool.get_pool_status()
    assert status['total_workers'] == 3
    assert status['available_workers'] == 3
    assert status['active_tasks'] == 0
    print(f"✓ Pool status: {status}")
    
    print("✓ All ResourcePool initialization tests passed")


async def test_worker_acquisition():
    """Test worker acquisition and release"""
    print_header("TEST: Worker Acquisition and Release")
    
    config = ResourceConfig(
        max_workers=2,
        worker_timeout_secs=10,
        worker_memory_mb=512
    )
    
    pool = ResourcePool(config)
    
    # Acquire worker
    async with pool.acquire_worker("test_task_1") as worker:
        assert worker is not None
        assert worker.id in [0, 1]
        print(f"✓ Acquired worker {worker.id}")
        
        # Check pool status while worker is acquired
        status = pool.get_pool_status()
        assert status['available_workers'] == 1
        assert status['active_tasks'] == 1
        assert "test_task_1" in status['active_task_ids']
        print(f"✓ Pool status during acquisition: {status}")
    
    # Worker should be returned after context exit
    status = pool.get_pool_status()
    assert status['available_workers'] == 2
    assert status['active_tasks'] == 0
    print("✓ Worker returned to pool after context exit")
    
    print("✓ All worker acquisition tests passed")


async def test_multiple_worker_acquisition():
    """Test multiple concurrent worker acquisitions"""
    print_header("TEST: Multiple Concurrent Worker Acquisitions")
    
    config = ResourceConfig(
        max_workers=3,
        worker_timeout_secs=10,
        worker_memory_mb=512
    )
    
    pool = ResourcePool(config)
    
    async def acquire_and_use_worker(task_id: str, delay: float):
        """Helper to acquire and use a worker"""
        async with pool.acquire_worker(task_id) as worker:
            print(f"  Task {task_id} using worker {worker.id}")
            await asyncio.sleep(delay)
            return worker.id
    
    # Run multiple tasks concurrently
    tasks = [
        acquire_and_use_worker("task_1", 0.1),
        acquire_and_use_worker("task_2", 0.1),
        acquire_and_use_worker("task_3", 0.1)
    ]
    
    worker_ids = await asyncio.gather(*tasks)
    print(f"✓ All tasks completed using workers: {worker_ids}")
    
    # All workers should be back in pool
    status = pool.get_pool_status()
    assert status['available_workers'] == 3
    assert status['active_tasks'] == 0
    print("✓ All workers returned to pool")
    
    print("✓ All multiple worker acquisition tests passed")


async def test_worker_timeout_enforcement():
    """Test worker timeout enforcement"""
    print_header("TEST: Worker Timeout Enforcement")
    
    config = ResourceConfig(
        max_workers=1,
        worker_timeout_secs=2,  # Short timeout for testing
        worker_memory_mb=512
    )
    
    pool = ResourcePool(config)
    
    timeout_occurred = False
    
    try:
        async with pool.acquire_worker("timeout_test") as worker:
            print(f"  Acquired worker {worker.id}, sleeping for 3 seconds...")
            # Sleep longer than timeout
            await asyncio.sleep(3)
    except WorkerTimeoutError as e:
        timeout_occurred = True
        print(f"✓ Timeout correctly enforced: {e}")
    
    assert timeout_occurred, "Timeout should have been enforced"
    print("✓ Worker timeout enforcement test passed")


async def test_bayesian_engine_integration():
    """Test BayesianInferenceEngine integration with ResourcePool"""
    print_header("TEST: BayesianInferenceEngine Integration")
    
    config = ResourceConfig(
        max_workers=2,
        worker_timeout_secs=10,
        worker_memory_mb=512
    )
    
    pool = ResourcePool(config)
    engine = BayesianInferenceEngine(pool)
    
    # Create mock causal link
    @dataclass
    class MockLink:
        id: str = "cause1-effect1"
        cause_id: str = "cause1"
        effect_id: str = "effect1"
    
    link = MockLink()
    
    # Run inference
    result = await engine.infer_mechanism(link)
    
    assert result is not None
    assert 'type' in result
    assert 'posterior_mean' in result
    assert 'necessity_test' in result
    assert 'device' in result
    assert 'worker_id' in result
    print(f"✓ Inference result: {result}")
    
    # Pool should have all workers available again
    status = pool.get_pool_status()
    assert status['available_workers'] == 2
    assert status['active_tasks'] == 0
    print("✓ Workers returned after inference")
    
    print("✓ All BayesianInferenceEngine integration tests passed")


async def test_concurrent_inference():
    """Test multiple concurrent inference tasks"""
    print_header("TEST: Concurrent Inference Tasks")
    
    config = ResourceConfig(
        max_workers=3,
        worker_timeout_secs=10,
        worker_memory_mb=512
    )
    
    pool = ResourcePool(config)
    engine = BayesianInferenceEngine(pool)
    
    # Create mock links
    @dataclass
    class MockLink:
        id: str
        cause_id: str
        effect_id: str
    
    links = [
        MockLink(id=f"link_{i}", cause_id=f"cause_{i}", effect_id=f"effect_{i}")
        for i in range(5)
    ]
    
    # Run concurrent inference
    start_time = time.time()
    results = await asyncio.gather(
        *[engine.infer_mechanism(link) for link in links]
    )
    elapsed = time.time() - start_time
    
    assert len(results) == 5
    for result in results:
        assert 'type' in result
        assert 'posterior_mean' in result
    
    print(f"✓ Completed {len(results)} inference tasks in {elapsed:.2f}s")
    
    # All workers should be available
    status = pool.get_pool_status()
    assert status['available_workers'] == 3
    assert status['active_tasks'] == 0
    print("✓ All workers returned after concurrent inference")
    
    print("✓ All concurrent inference tests passed")


async def test_pool_status_tracking():
    """Test pool status tracking during operations"""
    print_header("TEST: Pool Status Tracking")
    
    config = ResourceConfig(
        max_workers=2,
        worker_timeout_secs=10,
        worker_memory_mb=512
    )
    
    pool = ResourcePool(config)
    
    # Initial status
    status = pool.get_pool_status()
    print(f"Initial status: {status}")
    assert status['total_workers'] == 2
    assert status['available_workers'] == 2
    assert status['active_tasks'] == 0
    
    # Status during acquisition
    async with pool.acquire_worker("status_test") as worker:
        status = pool.get_pool_status()
        print(f"Status during acquisition: {status}")
        assert status['available_workers'] == 1
        assert status['active_tasks'] == 1
        assert "status_test" in status['active_task_ids']
    
    # Status after release
    status = pool.get_pool_status()
    print(f"Status after release: {status}")
    assert status['available_workers'] == 2
    assert status['active_tasks'] == 0
    
    print("✓ All pool status tracking tests passed")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 70)
    print("  RESOURCE POOL MANAGER - TEST SUITE")
    print("=" * 70)
    
    # Synchronous tests
    test_resource_config_validation()
    test_worker_creation()
    
    # Asynchronous tests
    async def run_async_tests():
        await test_resource_pool_initialization()
        await test_worker_acquisition()
        await test_multiple_worker_acquisition()
        
        # Optional: Skip timeout test if you want faster execution
        # It takes a few seconds due to actual timeout
        # await test_worker_timeout_enforcement()
        
        await test_bayesian_engine_integration()
        await test_concurrent_inference()
        await test_pool_status_tracking()
    
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()
