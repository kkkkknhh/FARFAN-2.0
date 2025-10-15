#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of PDMOrchestrator and AdaptiveLearningLoop
Demonstrates how to use the orchestration components
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass

# Import orchestration components
from orchestration.pdm_orchestrator import (
    PDMOrchestrator,
    PDMAnalysisState
)
from orchestration.learning_loop import AdaptiveLearningLoop


@dataclass
class ExampleSelfReflectionConfig:
    """Example self-reflection configuration"""
    enable_prior_learning: bool = True
    prior_history_path: str = "data/prior_history.json"
    feedback_weight: float = 0.1
    min_documents_for_learning: int = 5


@dataclass
class ExampleConfig:
    """Example configuration"""
    queue_size: int = 10
    max_inflight_jobs: int = 3
    worker_timeout_secs: int = 300
    min_quality_threshold: float = 0.5
    prior_decay_factor: float = 0.9
    
    def __post_init__(self):
        self.self_reflection = ExampleSelfReflectionConfig()


async def main():
    """Example orchestration workflow"""
    print("=" * 70)
    print("PDM Orchestrator - Example Usage")
    print("=" * 70)
    print()
    
    # 1. Initialize configuration
    config = ExampleConfig()
    print("✓ Configuration initialized")
    
    # 2. Create orchestrator
    orchestrator = PDMOrchestrator(config)
    print(f"✓ Orchestrator created (state: {orchestrator.state})")
    
    # 3. Create adaptive learning loop
    learning_loop = AdaptiveLearningLoop(config)
    print(f"✓ Learning loop created (enabled: {learning_loop.enabled})")
    print()
    
    # 4. Example: Analyze a plan (with mock PDF)
    print("Simulating plan analysis...")
    
    # Create a dummy PDF file for demonstration
    test_pdf = Path("/tmp/example_plan.pdf")
    test_pdf.write_text("Example PDM content")
    
    try:
        # Run analysis (will use fallback implementations in this example)
        result = await orchestrator.analyze_plan(str(test_pdf))
        
        print(f"✓ Analysis completed (run_id: {result.run_id})")
        print(f"  State: {orchestrator.state}")
        print(f"  Quality Score: {result.quality_score.overall_score:.2f}")
        print(f"  Dimension Scores: {result.quality_score.dimension_scores}")
        print(f"  Recommendations: {len(result.recommendations)}")
        print()
        
        # 5. Update priors based on results (if learning enabled)
        if learning_loop.enabled:
            print("Updating priors from analysis feedback...")
            learning_loop.extract_and_update_priors(result)
            print("✓ Priors updated")
            
            # Show example prior
            example_prior = learning_loop.get_current_prior("fallback")
            print(f"  Example prior (fallback): α={example_prior:.3f}")
            print()
        
        # 6. Show metrics summary
        print("Metrics Summary:")
        metrics = orchestrator.metrics.get_summary()
        print(f"  Metrics tracked: {len(metrics['metrics'])}")
        print(f"  Counters: {metrics['counters']}")
        print(f"  Alerts: {len(metrics['alerts'])}")
        
        # 7. Show audit trail
        print()
        print("Audit Trail:")
        print(f"  Records logged: {len(orchestrator.audit_logger.records)}")
        if orchestrator.audit_logger.records:
            last_record = orchestrator.audit_logger.records[-1]
            print(f"  Last record run_id: {last_record.get('run_id')}")
            print(f"  Final state: {last_record.get('final_state')}")
            print(f"  Duration: {last_record.get('duration_seconds', 0):.2f}s")
        
    finally:
        # Clean up
        if test_pdf.exists():
            test_pdf.unlink()
    
    print()
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    print("This example demonstrates the orchestration components.")
    print("In production, you would inject actual pipeline components:")
    print("  - extraction_pipeline: ExtractionPipeline")
    print("  - causal_builder: CausalGraphBuilder")
    print("  - bayesian_engine: BayesianInferenceOrchestrator")
    print("  - validator: AxiomaticValidator")
    print("  - scorer: QualityScorer")
    print()
    
    asyncio.run(main())
