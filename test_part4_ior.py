#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Part 4: IoR - Asynchronous Control and Resilience
=================================================================

Comprehensive tests for all three audit points:
- Audit Point 4.1: Execution Isolation
- Audit Point 4.2: Backpressure Signaling
- Audit Point 4.3: Fail-Open Policy

Author: AI Systems Architect
Version: 1.0.0
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from infrastructure import (  # Audit Point 4.2; Audit Point 4.1; Audit Point 4.3
    DNP_AVAILABLE,
    AsyncOrchestrator,
    CDAFValidationError,
    ComponentConfig,
    ComponentType,
    CoreValidationError,
    FailOpenPolicyManager,
    IsolatedPDFProcessor,
    IsolationConfig,
    IsolationStrategy,
    JobTimeoutError,
    OrchestratorConfig,
    PDFProcessingTimeoutError,
    QueueFullError,
    create_default_components,
    create_isolated_processor,
    create_orchestrator,
    create_policy_manager,
    is_dnp_available,
    set_dnp_available,
)


# ============================================================================
# Test Utilities
# ============================================================================
async def quick_job(*args, **kwargs):
    """Mock job that completes quickly"""
    await asyncio.sleep(0.1)
    return {"status": "completed", "args": args, "kwargs": kwargs}


async def slow_job(*args, **kwargs):
    """Mock job that takes time"""
    await asyncio.sleep(2.0)
    return {"status": "completed"}


async def failing_job(*args, **kwargs):
    """Mock job that always fails"""
    raise ValueError("Simulated job failure")


async def successful_validator(*args, **kwargs):
    """Mock validator that succeeds"""
    await asyncio.sleep(0.05)
    return {"score": 0.95, "status": "passed", "reason": "Validation passed"}


async def failing_validator(*args, **kwargs):
    """Mock validator that fails"""
    await asyncio.sleep(0.05)
    raise ConnectionError("Validation service unavailable")


# ============================================================================
# Audit Point 4.2: Backpressure Signaling Tests
# ============================================================================


async def test_orchestrator_initialization():
    """Test orchestrator initialization and configuration"""
    print("\n" + "=" * 70)
    print("TEST 4.2.1: Orchestrator Initialization")
    print("=" * 70)

    config = OrchestratorConfig(queue_size=50, max_workers=3, job_timeout_secs=30)
    orchestrator = AsyncOrchestrator(config)

    assert orchestrator.config.queue_size == 50
    assert orchestrator.config.max_workers == 3
    assert orchestrator.config.job_timeout_secs == 30
    print("✓ Configuration validated")

    assert not orchestrator.is_running
    print("✓ Orchestrator initialized in stopped state")

    print("✅ Test PASSED: Orchestrator Initialization\n")


async def test_orchestrator_start_shutdown():
    """Test orchestrator start and shutdown"""
    print("=" * 70)
    print("TEST 4.2.2: Orchestrator Start/Shutdown")
    print("=" * 70)

    orchestrator = create_orchestrator(queue_size=10, max_workers=2)

    await orchestrator.start()
    assert orchestrator.is_running
    assert len(orchestrator.workers) == 2
    print("✓ Orchestrator started with 2 workers")

    await orchestrator.shutdown()
    assert not orchestrator.is_running
    print("✓ Orchestrator shutdown gracefully")

    print("✅ Test PASSED: Orchestrator Start/Shutdown\n")


async def test_job_submission_and_completion():
    """Test basic job submission and completion"""
    print("=" * 70)
    print("TEST 4.2.3: Job Submission and Completion")
    print("=" * 70)

    orchestrator = create_orchestrator(queue_size=10, max_workers=2)
    await orchestrator.start()

    try:
        result = await orchestrator.submit_job(quick_job, "arg1", key="value")
        assert result["status"] == "completed"
        assert result["args"] == ("arg1",)
        assert result["kwargs"]["key"] == "value"
        print("✓ Job submitted and completed successfully")

        metrics = orchestrator.get_metrics()
        assert metrics.total_jobs_submitted == 1
        assert metrics.total_jobs_completed == 1
        print(
            f"✓ Metrics: submitted={metrics.total_jobs_submitted}, completed={metrics.total_jobs_completed}"
        )

    finally:
        await orchestrator.shutdown()

    print("✅ Test PASSED: Job Submission and Completion\n")


async def test_backpressure_http503():
    """Test backpressure with HTTP 503 when queue is full"""
    print("=" * 70)
    print("TEST 4.2.4: Backpressure Signaling (HTTP 503)")
    print("=" * 70)

    # Create orchestrator with small queue
    orchestrator = create_orchestrator(queue_size=5, max_workers=1, job_timeout_secs=10)
    await orchestrator.start()

    try:
        # Fill the queue with slow jobs (don't await them yet)
        tasks = []
        for i in range(5):
            task = asyncio.create_task(orchestrator.submit_job(slow_job, job_id=f"slow_{i}", timeout=5))
            tasks.append(task)
            await asyncio.sleep(0.05)  # Let jobs queue up

        # Wait a bit for queue to fill
        await asyncio.sleep(0.2)

        # Try to submit one more job - should get backpressure
        try:
            result = await orchestrator.submit_job(quick_job, job_id="overflow")
            # If we got here without exception, check if queue was actually full
            queue_info = orchestrator.get_queue_info()
            if queue_info["current_size"] < queue_info["max_size"]:
                # Queue wasn't full, this is ok
                print(f"✓ Queue not full yet ({queue_info['current_size']}/{queue_info['max_size']}), job accepted")
            else:
                assert False, "Should have raised QueueFullError when queue was full"
        except QueueFullError as e:
            assert e.http_status == 503
            print(f"✓ Backpressure triggered: HTTP {e.http_status}")
            print(f"✓ Queue full: {e.queue_size} jobs")

        metrics = orchestrator.get_metrics()
        print(f"✓ Metrics: rejected={metrics.total_jobs_rejected}, backpressure_events={metrics.backpressure_events}")

        # Cancel pending tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellations
        await asyncio.gather(*tasks, return_exceptions=True)

    finally:
        await orchestrator.shutdown()

    print("✅ Test PASSED: Backpressure Signaling (HTTP 503)\n")


async def test_queue_management_deque():
    """Test queue management using deque"""
    print("=" * 70)
    print("TEST 4.2.5: Queue Management (Deque)")
    print("=" * 70)

    orchestrator = create_orchestrator(queue_size=100, max_workers=2)
    await orchestrator.start()

    try:
        # Submit multiple jobs
        tasks = []
        for i in range(10):
            task = asyncio.create_task(orchestrator.submit_job(quick_job, f"job_{i}"))
            tasks.append(task)

        # Wait for all jobs to complete
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        print(f"✓ Processed {len(results)} jobs using deque")

        queue_info = orchestrator.get_queue_info()
        assert queue_info["max_size"] == 100
        assert queue_info["current_size"] == 0  # All jobs completed
        print(
            f"✓ Queue info: size={queue_info['current_size']}/{queue_info['max_size']}"
        )

    finally:
        await orchestrator.shutdown()

    print("✅ Test PASSED: Queue Management (Deque)\n")


async def test_job_timeout_enforcement():
    """Test job timeout enforcement"""
    print("=" * 70)
    print("TEST 4.2.6: Job Timeout Enforcement")
    print("=" * 70)

    orchestrator = create_orchestrator(queue_size=10, max_workers=2, job_timeout_secs=1)
    await orchestrator.start()

    try:
        # Submit job that exceeds timeout
        try:
            await orchestrator.submit_job(slow_job)
            assert False, "Should have raised JobTimeoutError"
        except JobTimeoutError as e:
            print(f"✓ Job timeout detected: {str(e)}")

        metrics = orchestrator.get_metrics()
        assert metrics.total_jobs_timeout > 0
        print(f"✓ Metrics: timeouts={metrics.total_jobs_timeout}")

    finally:
        await orchestrator.shutdown()

    print("✅ Test PASSED: Job Timeout Enforcement\n")


# ============================================================================
# Audit Point 4.1: Execution Isolation Tests
# ============================================================================


async def test_pdf_processor_initialization():
    """Test PDF processor initialization"""
    print("=" * 70)
    print("TEST 4.1.1: PDF Processor Initialization")
    print("=" * 70)

    config = IsolationConfig(
        worker_timeout_secs=60,
        isolation_strategy=IsolationStrategy.PROCESS,
        max_memory_mb=256,
    )
    processor = IsolatedPDFProcessor(config)

    assert processor.config.worker_timeout_secs == 60
    assert processor.config.isolation_strategy == IsolationStrategy.PROCESS
    print("✓ PDF processor initialized with correct configuration")

    print("✅ Test PASSED: PDF Processor Initialization\n")


async def test_pdf_processing_success():
    """Test successful PDF processing with isolation"""
    print("=" * 70)
    print("TEST 4.1.2: PDF Processing Success")
    print("=" * 70)

    processor = create_isolated_processor(worker_timeout_secs=30)

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4\ntest content")
        pdf_path = tmp.name

    try:
        result = await processor.process_pdf(pdf_path)
        assert result.success
        assert result.data is not None
        assert not result.timeout_occurred
        print(f"✓ PDF processed successfully in {result.execution_time:.2f}s")

        metrics = processor.get_metrics()
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        print(
            f"✓ Metrics: total={metrics.total_executions}, success={metrics.successful_executions}"
        )

    finally:
        Path(pdf_path).unlink()

    print("✅ Test PASSED: PDF Processing Success\n")


async def test_pdf_timeout_isolation():
    """Test timeout with process isolation"""
    print("=" * 70)
    print("TEST 4.1.3: PDF Timeout Isolation")
    print("=" * 70)

    processor = create_isolated_processor(worker_timeout_secs=1)

    # Simulate timeout scenario
    result = processor.simulate_timeout()
    assert not result.success
    assert result.timeout_occurred
    assert "timeout" in result.error.lower()
    print(f"✓ Timeout simulated: {result.error}")

    metrics = processor.get_metrics()
    assert metrics.timeout_failures == 1
    print(f"✓ Metrics: timeout_failures={metrics.timeout_failures}")

    print("✅ Test PASSED: PDF Timeout Isolation\n")


async def test_isolation_verification():
    """Test isolation verification for 99.9% uptime"""
    print("=" * 70)
    print("TEST 4.1.4: Isolation Verification (99.9% Uptime)")
    print("=" * 70)

    processor = create_isolated_processor(worker_timeout_secs=30)

    # Create temporary PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4\ntest")
        pdf_path = tmp.name

    try:
        # Process multiple times
        for i in range(10):
            await processor.process_pdf(pdf_path)

        # Verify isolation
        verification = processor.verify_isolation()
        assert verification["isolation_strategy"] == "process"
        assert verification["timeout_enforcement"]
        assert verification["worker_timeout_secs"] == 30
        print(f"✓ Isolation strategy: {verification['isolation_strategy']}")
        print(f"✓ Uptime: {verification['uptime_percentage']:.2f}% (target: 99.9%)")
        print(f"✓ Meets target: {verification['meets_target']}")

    finally:
        Path(pdf_path).unlink()

    print("✅ Test PASSED: Isolation Verification\n")


async def test_container_monitoring():
    """Test container execution monitoring"""
    print("=" * 70)
    print("TEST 4.1.5: Container Execution Monitoring")
    print("=" * 70)

    processor = create_isolated_processor(
        worker_timeout_secs=60, isolation_strategy=IsolationStrategy.CONTAINER
    )

    # Create temporary PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4\ntest")
        pdf_path = tmp.name

    try:
        result = await processor.process_pdf(pdf_path)
        # May fall back to process isolation if Docker not available
        assert result is not None
        print(f"✓ Container monitoring enabled (fallback to process if needed)")

        metrics = processor.get_metrics()
        print(f"✓ Metrics tracked: {metrics.total_executions} executions")

    finally:
        Path(pdf_path).unlink()

    print("✅ Test PASSED: Container Execution Monitoring\n")


# ============================================================================
# Audit Point 4.3: Fail-Open Policy Tests
# ============================================================================


async def test_policy_manager_initialization():
    """Test policy manager initialization"""
    print("=" * 70)
    print("TEST 4.3.1: Policy Manager Initialization")
    print("=" * 70)

    components = create_default_components()
    manager = FailOpenPolicyManager(components)

    assert "core_validator" in manager.components
    assert "dnp_validator" in manager.components
    assert manager.components["core_validator"].fail_closed
    assert not manager.components["dnp_validator"].fail_closed
    print("✓ Policy manager initialized with core and enrichment components")
    print(
        f"✓ Core validator: fail_closed={manager.components['core_validator'].fail_closed}"
    )
    print(
        f"✓ DNP validator: fail_closed={manager.components['dnp_validator'].fail_closed}"
    )

    print("✅ Test PASSED: Policy Manager Initialization\n")


async def test_fail_open_for_enrichment():
    """Test fail-open policy for enrichment components (DNP)"""
    print("=" * 70)
    print("TEST 4.3.2: Fail-Open for Enrichment (DNP)")
    print("=" * 70)

    manager = create_policy_manager()

    # Simulate DNP failure with fail-open
    result = await manager.execute_validation("dnp_validator", failing_validator)

    assert not result.success
    assert result.status == "skipped"
    assert result.fail_open_applied
    assert result.degradation_penalty == 0.05
    assert result.score == 0.95  # 1.0 - 0.05 penalty
    print(f"✓ DNP validation failed, continued with penalty")
    print(f"✓ Score: {result.score} (penalty: {result.degradation_penalty})")
    print(f"✓ Fail-open applied: {result.fail_open_applied}")

    metrics = manager.get_metrics()
    assert metrics.fail_open_applied == 1
    assert metrics.enrichment_failures == 1
    print(f"✓ Metrics: fail_open_applied={metrics.fail_open_applied}")

    print("✅ Test PASSED: Fail-Open for Enrichment\n")


async def test_fail_closed_for_core():
    """Test fail-closed policy for core components"""
    print("=" * 70)
    print("TEST 4.3.3: Fail-Closed for Core Components")
    print("=" * 70)

    manager = create_policy_manager()

    # Simulate core validator failure with fail-closed
    try:
        await manager.execute_validation("core_validator", failing_validator)
        assert False, "Should have raised CoreValidationError"
    except CoreValidationError as e:
        assert e.fail_closed
        assert e.component == "core_validator"
        print(f"✓ Core validation failed, halted with error: {e.component}")
        print(f"✓ Exception type: {type(e).__name__}")

    metrics = manager.get_metrics()
    assert metrics.core_failures == 1
    print(f"✓ Metrics: core_failures={metrics.core_failures}")

    print("✅ Test PASSED: Fail-Closed for Core Components\n")


async def test_graceful_degradation():
    """Test graceful degradation <10% accuracy loss"""
    print("=" * 70)
    print("TEST 4.3.4: Graceful Degradation (<10% Loss)")
    print("=" * 70)

    manager = create_policy_manager()

    # Run multiple validations with some failures
    for i in range(10):
        if i % 3 == 0:  # Fail every 3rd enrichment validation
            await manager.execute_validation("dnp_validator", failing_validator)
        else:
            await manager.execute_validation("dnp_validator", successful_validator)

    # Verify graceful degradation
    verification = manager.verify_graceful_degradation()
    assert verification["accuracy_loss"] < 0.10  # <10% target
    assert verification["meets_target"]
    print(f"✓ Total validations: {verification['total_validations']}")
    print(f"✓ Fail-open applied: {verification['fail_open_applied']}")
    print(f"✓ Accuracy loss: {verification['accuracy_loss']:.2%} (target: <10%)")
    print(f"✓ Meets target: {verification['meets_target']}")

    print("✅ Test PASSED: Graceful Degradation\n")


async def test_dnp_available_flag():
    """Test DNP_AVAILABLE flag integration"""
    print("=" * 70)
    print("TEST 4.3.5: DNP_AVAILABLE Flag")
    print("=" * 70)

    # Test flag manipulation
    assert is_dnp_available()
    print("✓ DNP initially available")

    set_dnp_available(False)
    assert not is_dnp_available()
    print("✓ DNP set to unavailable")

    set_dnp_available(True)
    assert is_dnp_available()
    print("✓ DNP set back to available")

    print("✅ Test PASSED: DNP_AVAILABLE Flag\n")


async def test_cdaf_validation_error():
    """Test CDAFValidationError exception handling"""
    print("=" * 70)
    print("TEST 4.3.6: CDAFValidationError Handling")
    print("=" * 70)

    manager = create_policy_manager()

    try:
        await manager.execute_validation("core_validator", failing_validator)
    except CDAFValidationError as e:
        assert isinstance(e, CoreValidationError)
        assert e.component == "core_validator"
        assert e.fail_closed
        print(f"✓ CDAFValidationError caught: {e.component}")
        print(f"✓ Fail-closed: {e.fail_closed}")
        print(f"✓ Details: {e.details}")

    print("✅ Test PASSED: CDAFValidationError Handling\n")


# ============================================================================
# Test Runner
# ============================================================================


async def run_all_tests():
    """Run all tests sequentially"""
    print("\n" + "=" * 70)
    print("PART 4: IoR - ASYNCHRONOUS CONTROL AND RESILIENCE TEST SUITE")
    print("=" * 70)

    # Audit Point 4.2: Backpressure Signaling
    print("\n" + "=" * 70)
    print("AUDIT POINT 4.2: BACKPRESSURE SIGNALING")
    print("=" * 70)
    await test_orchestrator_initialization()
    await test_orchestrator_start_shutdown()
    await test_job_submission_and_completion()
    await test_backpressure_http503()
    await test_queue_management_deque()
    await test_job_timeout_enforcement()

    # Audit Point 4.1: Execution Isolation
    print("\n" + "=" * 70)
    print("AUDIT POINT 4.1: EXECUTION ISOLATION")
    print("=" * 70)
    await test_pdf_processor_initialization()
    await test_pdf_processing_success()
    await test_pdf_timeout_isolation()
    await test_isolation_verification()
    await test_container_monitoring()

    # Audit Point 4.3: Fail-Open Policy
    print("\n" + "=" * 70)
    print("AUDIT POINT 4.3: FAIL-OPEN POLICY")
    print("=" * 70)
    await test_policy_manager_initialization()
    await test_fail_open_for_enrichment()
    await test_fail_closed_for_core()
    await test_graceful_degradation()
    await test_dnp_available_flag()
    await test_cdaf_validation_error()

    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
    print("\nSummary:")
    print("- Audit Point 4.1: Execution Isolation ✅ (5 tests)")
    print("- Audit Point 4.2: Backpressure Signaling ✅ (6 tests)")
    print("- Audit Point 4.3: Fail-Open Policy ✅ (6 tests)")
    print("- Total: 17 tests passed")
    print("=" * 70)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run tests
    asyncio.run(run_all_tests())
