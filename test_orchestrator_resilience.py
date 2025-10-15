#!/usr/bin/env python3
"""
Test script para validar los componentes de resiliencia del orchestrator
"""

import tempfile
from pathlib import Path
from dataclasses import dataclass

# Test imports
from risk_registry import RiskRegistry, RiskSeverity, RiskCategory, RiskDefinition
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitBreakerRegistry
from pipeline_checkpoint import PipelineCheckpoint
from pipeline_metrics import PipelineMetrics, AlertLevel


def test_risk_registry():
    """Test RiskRegistry functionality"""
    print("\n" + "="*60)
    print("TEST: RiskRegistry")
    print("="*60)
    
    registry = RiskRegistry()
    print(f"✓ Registry initialized with {len(registry.risks)} default risks")
    
    # Test risk assessment
    assessments = registry.assess_stage_risks("LOAD_DOCUMENT")
    print(f"✓ Assessed {len(assessments)} risks for LOAD_DOCUMENT stage")
    
    # Test finding risk by exception
    test_exception = FileNotFoundError("Test file not found")
    risk = registry.find_risk_by_exception(test_exception, "LOAD_DOCUMENT")
    if risk:
        print(f"✓ Found matching risk: {risk.risk_id}")
    
    # Test mitigation execution
    pdf_corrupt_risk = registry.risks.get("PDF_CORRUPT")
    if pdf_corrupt_risk:
        attempt = registry.execute_mitigation(pdf_corrupt_risk, None)
        print(f"✓ Mitigation attempt executed: success={attempt.success}")
    
    # Test stats
    stats = registry.get_mitigation_stats()
    print(f"✓ Mitigation stats: {stats.get('total_attempts', 0)} total attempts")
    
    print("✅ RiskRegistry tests passed")


def test_circuit_breaker():
    """Test CircuitBreaker functionality"""
    print("\n" + "="*60)
    print("TEST: CircuitBreaker")
    print("="*60)
    
    config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)
    breaker = CircuitBreaker("test_stage", config)
    print(f"✓ CircuitBreaker initialized in state: {breaker.state.value}")
    
    # Test successful call
    def success_func():
        return "success"
    
    result = breaker.call(success_func)
    print(f"✓ Successful call returned: {result}")
    
    # Test failure
    def fail_func():
        raise ValueError("Test failure")
    
    try:
        breaker.call(fail_func)
    except ValueError:
        print(f"✓ Failure recorded, count: {breaker.failure_count}")
    
    try:
        breaker.call(fail_func)
    except ValueError:
        print(f"✓ Second failure recorded, state: {breaker.state.value}")
    
    # Test stats
    stats = breaker.get_stats()
    print(f"✓ Stats: {stats['total_calls']} calls, {stats['total_failures']} failures")
    
    # Test registry
    registry = CircuitBreakerRegistry()
    breaker2 = registry.get_or_create("stage2")
    print(f"✓ Registry created breaker for stage2")
    
    all_stats = registry.get_all_stats()
    print(f"✓ Registry has {len(all_stats)} breakers")
    
    print("✅ CircuitBreaker tests passed")


def test_pipeline_checkpoint():
    """Test PipelineCheckpoint functionality"""
    print("\n" + "="*60)
    print("TEST: PipelineCheckpoint")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    checkpoint = PipelineCheckpoint(temp_dir)
    print(f"✓ Checkpoint initialized at: {temp_dir}")
    
    # Create mock context
    @dataclass
    class MockContext:
        policy_code: str
        stage_data: str
    
    ctx = MockContext(policy_code="TEST-001", stage_data="test data")
    
    # Save checkpoint
    checkpoint_id = checkpoint.save(
        policy_code="TEST-001",
        stage_name="TEST_STAGE",
        context=ctx,
        execution_time_ms=100.5,
        success=True
    )
    print(f"✓ Checkpoint saved with ID: {checkpoint_id}")
    
    # Load checkpoint
    metadata, loaded_ctx = checkpoint.load(checkpoint_id)
    print(f"✓ Checkpoint loaded: {metadata.stage_name}")
    
    # List checkpoints
    checkpoints = checkpoint.list_checkpoints("TEST-001")
    print(f"✓ Found {len(checkpoints)} checkpoints for TEST-001")
    
    # Get chain
    chain = checkpoint.get_checkpoint_chain(checkpoint_id)
    print(f"✓ Checkpoint chain has {len(chain)} entries")
    
    print("✅ PipelineCheckpoint tests passed")


def test_pipeline_metrics():
    """Test PipelineMetrics functionality"""
    print("\n" + "="*60)
    print("TEST: PipelineMetrics")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    metrics = PipelineMetrics(temp_dir)
    print(f"✓ Metrics initialized at: {temp_dir}")
    
    # Start execution
    metrics.start_execution("TEST-001")
    print("✓ Execution started")
    
    # Stage 1
    metrics.start_stage("STAGE_1")
    metrics.record_risk_assessment("RISK_1")
    metrics.record_circuit_breaker_state("CLOSED")
    metrics.end_stage(success=True)
    print("✓ Stage 1 completed")
    
    # Stage 2 with failure
    metrics.start_stage("STAGE_2")
    metrics.record_risk_assessment("RISK_2")
    metrics.record_mitigation("RISK_2", "DATA_QUALITY", "MEDIUM")
    metrics.emit_alert(AlertLevel.WARNING, "Test warning", {"stage": "STAGE_2"})
    metrics.end_stage(success=False, error_message="Test error")
    print("✓ Stage 2 failed with mitigation")
    
    # End execution
    metrics.end_execution(success=False)
    print("✓ Execution ended")
    
    # Get stats
    success_rates = metrics.get_stage_success_rates()
    print(f"✓ Success rates: {success_rates}")
    
    avg_times = metrics.get_average_stage_times()
    print(f"✓ Average times: {avg_times}")
    
    print("✅ PipelineMetrics tests passed")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TESTING ORCHESTRATOR RESILIENCE COMPONENTS")
    print("="*60)
    
    try:
        test_risk_registry()
        test_circuit_breaker()
        test_pipeline_checkpoint()
        test_pipeline_metrics()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
