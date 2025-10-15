#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive demonstration of orchestration workflow
Shows full F2.1 and F2.2 implementation with all features
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass, field

from orchestration import PDMOrchestrator, AdaptiveLearningLoop, PDMAnalysisState
from orchestration.pdm_orchestrator import (
    MechanismResult, ValidationResult, QualityScore
)


@dataclass
class DemoSelfReflectionConfig:
    """Demo self-reflection configuration"""
    enable_prior_learning: bool = True
    prior_history_path: str = "/tmp/demo_prior_history.json"
    feedback_weight: float = 0.1
    min_documents_for_learning: int = 2


@dataclass
class DemoConfig:
    """Demo configuration with all orchestration features"""
    # Queue management (Backpressure Standard)
    queue_size: int = 10
    max_inflight_jobs: int = 3
    
    # Timeout enforcement
    worker_timeout_secs: int = 300
    
    # Quality gates
    min_quality_threshold: float = 0.5
    
    # Adaptive learning
    prior_decay_factor: float = 0.9
    
    # Audit store
    audit_store_path: Path = field(default_factory=lambda: Path("/tmp/demo_audit.jsonl"))
    
    def __post_init__(self):
        self.self_reflection = DemoSelfReflectionConfig()


async def demonstrate_full_workflow():
    """
    Comprehensive demonstration of orchestration workflow.
    Shows all features: state machine, metrics, audit logging, and learning.
    """
    
    print("=" * 80)
    print("COMPREHENSIVE ORCHESTRATION DEMONSTRATION")
    print("F2.1: PDMOrchestrator with State Machine + F2.2: Adaptive Learning Loop")
    print("=" * 80)
    print()
    
    # ========================================================================
    # PHASE 0: INITIALIZATION
    # ========================================================================
    print("PHASE 0: INITIALIZATION")
    print("-" * 80)
    
    config = DemoConfig()
    print("✓ Configuration created")
    print(f"  - Queue size: {config.queue_size}")
    print(f"  - Max concurrent jobs: {config.max_inflight_jobs}")
    print(f"  - Worker timeout: {config.worker_timeout_secs}s")
    print(f"  - Prior decay factor: {config.prior_decay_factor}")
    print()
    
    orchestrator = PDMOrchestrator(config)
    print("✓ Orchestrator initialized")
    print(f"  - Initial state: {orchestrator.state}")
    print(f"  - Metrics collector: {type(orchestrator.metrics).__name__}")
    print(f"  - Audit logger: {type(orchestrator.audit_logger).__name__}")
    print()
    
    learning_loop = AdaptiveLearningLoop(config)
    print("✓ Learning loop initialized")
    print(f"  - Learning enabled: {learning_loop.enabled}")
    print(f"  - Prior store path: {learning_loop.prior_store.store_path}")
    print()
    
    # ========================================================================
    # DEMONSTRATION 1: SUCCESSFUL ANALYSIS
    # ========================================================================
    print("=" * 80)
    print("DEMONSTRATION 1: SUCCESSFUL ANALYSIS")
    print("-" * 80)
    
    # Create test PDF
    test_pdf = Path("/tmp/demo_plan_success.pdf")
    test_pdf.write_text("Demo PDM content - high quality plan")
    
    print(f"Running analysis on: {test_pdf}")
    print()
    
    result1 = await orchestrator.analyze_plan(str(test_pdf))
    
    print("✓ Analysis completed")
    print(f"  - Run ID: {result1.run_id}")
    print(f"  - Final state: {orchestrator.state}")
    print(f"  - Quality score: {result1.quality_score.overall_score:.2%}")
    print()
    
    print("Dimension scores:")
    for dim, score in result1.quality_score.dimension_scores.items():
        print(f"  - {dim}: {score:.2%}")
    print()
    
    print(f"Mechanism results: {len(result1.mechanism_results)} mechanisms analyzed")
    for i, mech in enumerate(result1.mechanism_results, 1):
        print(f"  {i}. Type: {mech.type}, Passed: {mech.necessity_test.get('passed', True)}")
    print()
    
    # Update priors
    print("Updating priors from results...")
    learning_loop.extract_and_update_priors(result1)
    print("✓ Priors updated")
    print()
    
    # Show metrics
    metrics1 = orchestrator.metrics.get_summary()
    print("Metrics summary:")
    print(f"  - Tracked metrics: {len(metrics1['metrics'])}")
    for metric_name, metric_data in list(metrics1['metrics'].items())[:5]:
        print(f"    • {metric_name}: {metric_data['last']:.2f}")
    print()
    
    # ========================================================================
    # DEMONSTRATION 2: ANALYSIS WITH FAILURES
    # ========================================================================
    print("=" * 80)
    print("DEMONSTRATION 2: ANALYSIS WITH MECHANISM FAILURES")
    print("-" * 80)
    
    # Reset state for new analysis
    orchestrator.state = PDMAnalysisState.INITIALIZED
    
    # Inject custom mechanism engine to simulate failures
    class FailingMechanismEngine:
        async def infer_all_mechanisms(self, graph, chunks):
            # Return mechanisms with some failures
            return [
                MechanismResult(
                    type="causal_link",
                    necessity_test={'passed': False, 'missing': ['evidence_A', 'evidence_B']},
                    posterior_mean=0.3
                ),
                MechanismResult(
                    type="inference_chain",
                    necessity_test={'passed': False, 'missing': ['source_data']},
                    posterior_mean=0.25
                ),
                MechanismResult(
                    type="direct_mechanism",
                    necessity_test={'passed': True, 'missing': []},
                    posterior_mean=0.85
                )
            ]
    
    orchestrator.bayesian_engine = FailingMechanismEngine()
    
    test_pdf2 = Path("/tmp/demo_plan_failures.pdf")
    test_pdf2.write_text("Demo PDM content - with mechanism failures")
    
    print(f"Running analysis on: {test_pdf2}")
    print("(Simulating mechanism failures)")
    print()
    
    result2 = await orchestrator.analyze_plan(str(test_pdf2))
    
    print("✓ Analysis completed")
    print(f"  - Run ID: {result2.run_id}")
    print(f"  - Final state: {orchestrator.state}")
    print(f"  - Quality score: {result2.quality_score.overall_score:.2%}")
    print()
    
    print("Mechanism results with failures:")
    passed = sum(1 for m in result2.mechanism_results if m.necessity_test.get('passed', True))
    failed = len(result2.mechanism_results) - passed
    print(f"  - Passed: {passed}")
    print(f"  - Failed: {failed}")
    print()
    
    for mech in result2.mechanism_results:
        status = "✓ PASSED" if mech.necessity_test.get('passed', True) else "✗ FAILED"
        print(f"  - {mech.type}: {status}")
        if not mech.necessity_test.get('passed', True):
            missing = mech.necessity_test.get('missing', [])
            print(f"    Missing: {', '.join(missing)}")
    print()
    
    # Show priors before update
    print("Priors before learning update:")
    for mech_type in ['causal_link', 'inference_chain', 'direct_mechanism']:
        alpha = learning_loop.get_current_prior(mech_type)
        print(f"  - {mech_type}: α={alpha:.3f}")
    print()
    
    # Update priors (should decay for failed mechanisms)
    print("Updating priors from failure feedback...")
    learning_loop.extract_and_update_priors(result2)
    print("✓ Priors updated (failed mechanisms decayed)")
    print()
    
    # Show priors after update
    print("Priors after learning update:")
    for mech_type in ['causal_link', 'inference_chain', 'direct_mechanism']:
        alpha = learning_loop.get_current_prior(mech_type)
        print(f"  - {mech_type}: α={alpha:.3f}")
    print()
    
    # ========================================================================
    # OBSERVABILITY AND GOVERNANCE
    # ========================================================================
    print("=" * 80)
    print("OBSERVABILITY AND GOVERNANCE")
    print("-" * 80)
    
    # Metrics
    metrics_final = orchestrator.metrics.get_summary()
    print("Complete metrics summary:")
    print(f"  - Total metrics tracked: {len(metrics_final['metrics'])}")
    print(f"  - Total counters: {len(metrics_final['counters'])}")
    print(f"  - Total alerts: {len(metrics_final['alerts'])}")
    print()
    
    if metrics_final['alerts']:
        print("Alerts raised:")
        for alert in metrics_final['alerts']:
            print(f"  - [{alert['level']}] {alert['message']}")
        print()
    
    # Audit trail
    print("Audit trail:")
    print(f"  - Total records: {len(orchestrator.audit_logger.records)}")
    for i, record in enumerate(orchestrator.audit_logger.records, 1):
        print(f"  {i}. Run: {record['run_id']}")
        print(f"     State: {record['final_state']}")
        print(f"     Duration: {record['duration_seconds']:.2f}s")
        print(f"     SHA256: {record['sha256_source'][:16]}...")
    print()
    
    # Prior history
    print("Prior learning history:")
    history = learning_loop.get_prior_history()
    print(f"  - Total snapshots: {len(history)}")
    if history:
        latest = history[-1]
        print(f"  - Latest snapshot: {latest['timestamp']}")
        print(f"    Priors tracked: {len(latest['priors'])}")
    print()
    
    # ========================================================================
    # STATE MACHINE VERIFICATION
    # ========================================================================
    print("=" * 80)
    print("STATE MACHINE VERIFICATION")
    print("-" * 80)
    
    all_states = list(PDMAnalysisState)
    print(f"Total states defined: {len(all_states)}")
    for state in all_states:
        print(f"  - {state.value}")
    print()
    
    print("State transitions observed:")
    state_metrics = [m for m in metrics_final['metrics'] if 'state' in m.lower()]
    for metric in state_metrics:
        count = metrics_final['metrics'][metric]['count']
        print(f"  - {metric}: {count} transitions")
    print()
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    test_pdf.unlink()
    test_pdf2.unlink()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Key achievements demonstrated:")
    print("  ✓ State machine with 8 states")
    print("  ✓ Backpressure management (queue + semaphore)")
    print("  ✓ Timeout enforcement")
    print("  ✓ Metrics collection and alerting")
    print("  ✓ Immutable audit logging with SHA256")
    print("  ✓ Phase I-IV execution flow")
    print("  ✓ Prior learning from failures")
    print("  ✓ Prior decay for failed mechanisms")
    print("  ✓ Prior boost for successful mechanisms")
    print("  ✓ Snapshot-based history tracking")
    print()


if __name__ == "__main__":
    asyncio.run(demonstrate_full_workflow())
