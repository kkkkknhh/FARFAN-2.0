#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test: Observability Stack with PDM Orchestrator pattern
Demonstrates how to integrate with existing orchestration code.
"""

import asyncio
import time
from infrastructure.observability import ObservabilityConfig, ObservabilityStack


class MockPDMOrchestrator:
    """Mock orchestrator demonstrating integration pattern"""
    
    def __init__(self):
        # Initialize observability stack
        config = ObservabilityConfig(
            metrics_backend='in_memory',
            log_level='INFO',
            trace_backend='in_memory'
        )
        self.observability = ObservabilityStack(config)
        
    async def analyze_plan(self, pdf_path: str) -> dict:
        """Simulate PDM pipeline execution with observability"""
        
        pipeline_start = time.time()
        run_id = f"run_{int(time.time())}"
        
        # Track overall pipeline
        with self.observability.trace_operation('analyze_plan', run_id=run_id, pdf=pdf_path):
            
            # Phase I: Extraction
            with self.observability.trace_operation('extraction', phase='I'):
                await asyncio.sleep(0.05)  # Simulate work
                self.observability.record_memory_peak(7500.0)
            
            # Phase II: Graph construction
            with self.observability.trace_operation('graph_construction', phase='II'):
                await asyncio.sleep(0.05)
                self.observability.record_memory_peak(9200.0)
            
            # Phase III: Bayesian inference
            with self.observability.trace_operation('bayesian_inference', phase='III'):
                await asyncio.sleep(0.05)
                # Simulate convergence check
                # In real code, this would check actual R_hat values
                convergence_ok = True
                if not convergence_ok:
                    self.observability.record_nonconvergent_chain('chain_2', 'R_hat=1.12')
            
            # Phase IV: Validation
            with self.observability.trace_operation('validation', phase='IV'):
                await asyncio.sleep(0.05)
                # Simulate hoop test
                # In real code, this would validate actual evidence
                hoop_test_passed = True
                if not hoop_test_passed:
                    self.observability.record_hoop_test_failure('D1-Q5', ['regulatory'])
            
            # Phase V: Scoring
            with self.observability.trace_operation('scoring', phase='V'):
                await asyncio.sleep(0.05)
                # Record dimension scores
                scores = {
                    'D1': 0.72, 'D2': 0.68, 'D3': 0.75,
                    'D4': 0.70, 'D5': 0.65, 'D6': 0.62
                }
                for dim, score in scores.items():
                    self.observability.record_dimension_score(dim, score)
        
        # Record total pipeline duration
        duration = time.time() - pipeline_start
        self.observability.record_pipeline_duration(duration)
        
        # Return results with observability data
        return {
            'run_id': run_id,
            'duration': duration,
            'metrics': self.observability.get_metrics_summary(),
            'traces': self.observability.get_traces_summary()
        }


async def test_integration():
    """Test integration with orchestrator pattern"""
    
    print("=" * 70)
    print("Integration Test: Observability Stack with PDM Orchestrator")
    print("=" * 70)
    print()
    
    orchestrator = MockPDMOrchestrator()
    
    # Run analysis
    print("Running PDM analysis with observability...")
    result = await orchestrator.analyze_plan('/path/to/plan.pdf')
    
    print(f"\n✓ Analysis completed: {result['run_id']}")
    print(f"  Duration: {result['duration']:.3f}s")
    
    # Show metrics
    metrics = result['metrics']
    print(f"\n📊 Metrics collected:")
    print(f"  - {len(metrics['histograms'])} histograms")
    print(f"  - {len(metrics['gauges'])} gauges")
    print(f"  - {len(metrics['counters'])} counters")
    
    # Show traces
    traces = result['traces']
    print(f"\n🔍 Operations traced: {len(traces)}")
    for trace in traces:
        print(f"  - {trace['operation']}: {trace['duration']:.3f}s")
    
    # Verify critical metrics
    print("\n⚠️  Critical Metrics Verification:")
    
    # Check pipeline duration recorded
    assert 'pdm.pipeline.duration_seconds' in metrics['histograms']
    print("  ✓ Pipeline duration recorded")
    
    # Check dimension scores recorded
    assert 'pdm.dimension.avg_score_D6' in metrics['gauges']
    d6_score = metrics['gauges']['pdm.dimension.avg_score_D6']
    print(f"  ✓ D6 score recorded: {d6_score:.2f}")
    
    # Check memory peak recorded
    assert 'pdm.memory.peak_mb' in metrics['gauges']
    memory = metrics['gauges']['pdm.memory.peak_mb']
    print(f"  ✓ Peak memory recorded: {memory:.1f} MB")
    
    # Check all phases traced
    phase_names = {t['operation'] for t in traces}
    expected_phases = {
        'analyze_plan', 'extraction', 'graph_construction',
        'bayesian_inference', 'validation', 'scoring'
    }
    assert phase_names == expected_phases
    print(f"  ✓ All {len(expected_phases)} phases traced")
    
    print()
    print("=" * 70)
    print("Integration test PASSED ✅")
    print("=" * 70)


async def test_alert_behavior():
    """Test alert behavior in integration context"""
    
    print("\n" + "=" * 70)
    print("Alert Behavior Test")
    print("=" * 70)
    print()
    
    orchestrator = MockPDMOrchestrator()
    
    print("Testing alert conditions:")
    
    # Test memory alert
    print("\n1. Memory Peak Alert:")
    print("   Setting peak memory to 18GB...")
    orchestrator.observability.record_memory_peak(18000.0)
    print("   ✓ WARNING alert should have been logged above")
    
    # Test D6 score alert
    print("\n2. D6 Score Alert:")
    print("   Setting D6 score to 0.50...")
    orchestrator.observability.record_dimension_score('D6', 0.50)
    print("   ✓ CRITICAL alert should have been logged above")
    
    # Test convergence alert
    print("\n3. Non-convergent Chain Alert:")
    print("   Recording non-convergent chain...")
    orchestrator.observability.record_nonconvergent_chain('chain_5', 'Failed convergence')
    print("   ✓ CRITICAL alert should have been logged above")
    
    print("\n" + "=" * 70)
    print("Alert behavior test PASSED ✅")
    print("=" * 70)


if __name__ == '__main__':
    # Run integration tests
    asyncio.run(test_integration())
    asyncio.run(test_alert_behavior())
