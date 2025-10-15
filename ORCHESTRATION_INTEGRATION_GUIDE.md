# Integration Guide: Orchestration with CDAF Framework

This guide shows how to integrate the new orchestration module with the existing CDAF framework.

## Overview

The orchestration module (`orchestration/`) provides:
1. **PDMOrchestrator**: State machine-based orchestrator for Phase 0-IV execution
2. **AdaptiveLearningLoop**: Prior learning from historical failures

## Step-by-Step Integration

### 1. Import Required Components

```python
from pathlib import Path
from orchestration import PDMOrchestrator, AdaptiveLearningLoop
```

### 2. Using with Existing ConfigLoader

The existing `ConfigLoader` class in `dereck_beach` already supports the required configuration structure. To use it with the orchestrator:

```python
# Load existing CDAF config
from dereck_beach import ConfigLoader

config_path = Path("config.yaml")
config = ConfigLoader(config_path)

# Create orchestrator - it will extract needed config
orchestrator = PDMOrchestrator(config)

# Create learning loop
learning_loop = AdaptiveLearningLoop(config)
```

### 3. Inject Existing Pipeline Components

The orchestrator uses dependency injection. Connect existing CDAF components:

```python
# Assume you have a CDAFFramework instance
from dereck_beach import CDAFFramework

framework = CDAFFramework(config_path, output_dir)

# Inject components into orchestrator
# Note: These need to be wrapped with async interfaces

# Option A: Create async wrappers
class AsyncExtractionPipeline:
    def __init__(self, pdf_processor):
        self.pdf_processor = pdf_processor
    
    async def extract_complete(self, pdf_path: str):
        # Wrap sync call
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self._extract_sync, 
            pdf_path
        )
        return result
    
    def _extract_sync(self, pdf_path):
        from orchestration.pdm_orchestrator import ExtractionResult
        # Use existing PDF processor
        # This is a placeholder - adapt to actual API
        return ExtractionResult(
            semantic_chunks=[],
            tables=[],
            extraction_quality={'score': 0.8}
        )

# Similar wrappers for other components...

# Option B: Directly inject if they already support async
orchestrator.extraction_pipeline = framework.pdf_processor  # if async
orchestrator.causal_builder = framework.causal_extractor  # if async
orchestrator.bayesian_engine = framework.bayesian_mechanism  # if async
orchestrator.validator = framework.op_auditor  # if async
orchestrator.scorer = framework.reporting_engine  # if async
```

### 4. Configuration Enhancement

Add orchestration config to your YAML config file:

```yaml
# Existing config sections...
patterns:
  # ... existing patterns ...

lexicons:
  # ... existing lexicons ...

# Add orchestration configuration
orchestration:
  queue_size: 10
  max_inflight_jobs: 3
  worker_timeout_secs: 300
  min_quality_threshold: 0.5
  prior_decay_factor: 0.9

# Enhance self_reflection config
self_reflection:
  enable_prior_learning: true
  feedback_weight: 0.1
  prior_history_path: "data/prior_history.json"
  min_documents_for_learning: 5
```

The orchestrator will read these values from the config object.

### 5. Running Analysis with Orchestration

```python
import asyncio

async def run_orchestrated_analysis(pdf_path: str):
    """Run analysis with orchestration and learning"""
    
    # 1. Configure
    config = ConfigLoader(Path("config.yaml"))
    
    # 2. Create orchestrator and learning loop
    orchestrator = PDMOrchestrator(config)
    learning_loop = AdaptiveLearningLoop(config)
    
    # 3. Inject components (see step 3 above)
    # ... inject components here ...
    
    # 4. Run analysis
    result = await orchestrator.analyze_plan(pdf_path)
    
    # 5. Learn from results
    if learning_loop.enabled:
        learning_loop.extract_and_update_priors(result)
    
    # 6. Return results
    return result

# Run it
result = asyncio.run(run_orchestrated_analysis("plan.pdf"))
print(f"Analysis completed: {result.run_id}")
print(f"Quality score: {result.quality_score.overall_score}")
```

### 6. Accessing Metrics and Audit Logs

```python
# Get metrics summary
metrics = orchestrator.metrics.get_summary()
print(f"Metrics: {metrics['metrics']}")
print(f"Counters: {metrics['counters']}")
print(f"Alerts: {metrics['alerts']}")

# Access audit trail
audit_records = orchestrator.audit_logger.records
for record in audit_records:
    print(f"Run {record['run_id']}: {record['final_state']}")
```

### 7. Monitoring Prior Learning

```python
# Check current priors
mechanism_types = ['causal_link', 'inference', 'mechanism']
for mech_type in mechanism_types:
    alpha = learning_loop.get_current_prior(mech_type)
    print(f"{mech_type}: α={alpha:.3f}")

# View learning history
history = learning_loop.get_prior_history()
print(f"Total snapshots: {len(history)}")

# View specific mechanism history
causal_history = learning_loop.get_prior_history('causal_link')
for snapshot in causal_history:
    print(f"  {snapshot['timestamp']}: α={snapshot['prior']['alpha']}")
```

## Example: Complete Integration

Here's a complete example integrating everything:

```python
#!/usr/bin/env python3
import asyncio
from pathlib import Path
from orchestration import PDMOrchestrator, AdaptiveLearningLoop

async def main():
    # Load configuration (uses existing ConfigLoader)
    # For now, create a simple config object
    from dataclasses import dataclass
    
    @dataclass
    class SimpleConfig:
        queue_size: int = 10
        max_inflight_jobs: int = 3
        worker_timeout_secs: int = 300
        min_quality_threshold: float = 0.5
        prior_decay_factor: float = 0.9
        
        @dataclass
        class SelfReflection:
            enable_prior_learning: bool = True
            prior_history_path: str = "data/prior_history.json"
        
        def __post_init__(self):
            self.self_reflection = self.SelfReflection()
    
    config = SimpleConfig()
    
    # Create orchestration components
    orchestrator = PDMOrchestrator(config)
    learning_loop = AdaptiveLearningLoop(config)
    
    # TODO: Inject actual pipeline components here
    # orchestrator.extraction_pipeline = ...
    # orchestrator.causal_builder = ...
    # etc.
    
    # Run analysis
    pdf_path = "path/to/plan.pdf"
    result = await orchestrator.analyze_plan(pdf_path)
    
    # Update priors from results
    learning_loop.extract_and_update_priors(result)
    
    # Report results
    print(f"Analysis completed: {result.run_id}")
    print(f"State: {orchestrator.state}")
    print(f"Quality: {result.quality_score.overall_score:.2%}")
    
    # Show metrics
    metrics = orchestrator.metrics.get_summary()
    print(f"Tracked metrics: {list(metrics['metrics'].keys())}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Fallback Behavior

The orchestrator has built-in fallback implementations for all components. If a component is not injected:

- `extraction_pipeline`: Returns basic ExtractionResult with sample data
- `causal_builder`: Returns empty NetworkX DiGraph
- `bayesian_engine`: Returns single MechanismResult
- `validator`: Returns passing ValidationResult
- `scorer`: Returns simple QualityScore based on extraction quality

This allows incremental integration - you can test the orchestrator before all components are ready.

## Testing Integration

Test your integration:

```python
# test_integration.py
import asyncio
from orchestration import PDMOrchestrator, AdaptiveLearningLoop

async def test_basic_integration():
    """Test basic integration without full pipeline"""
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        queue_size: int = 5
        max_inflight_jobs: int = 2
        worker_timeout_secs: int = 60
        min_quality_threshold: float = 0.5
        prior_decay_factor: float = 0.9
        
        @dataclass 
        class SelfReflection:
            enable_prior_learning: bool = True
            prior_history_path: str = "/tmp/test_priors.json"
        
        def __post_init__(self):
            self.self_reflection = self.SelfReflection()
    
    config = TestConfig()
    orchestrator = PDMOrchestrator(config)
    learning_loop = AdaptiveLearningLoop(config)
    
    # Create test file
    from pathlib import Path
    test_pdf = Path("/tmp/test.pdf")
    test_pdf.write_text("test")
    
    try:
        # Run with fallbacks
        result = await orchestrator.analyze_plan(str(test_pdf))
        assert result is not None
        assert result.run_id is not None
        
        # Test learning
        learning_loop.extract_and_update_priors(result)
        
        print("✓ Integration test passed")
        return True
    finally:
        test_pdf.unlink()

if __name__ == "__main__":
    asyncio.run(test_basic_integration())
```

## Next Steps

1. **Adapt Existing Components**: Make existing CDAF components async-compatible or create async wrappers
2. **Configure Properly**: Add orchestration config to your YAML file
3. **Inject Components**: Wire up real pipeline components instead of using fallbacks
4. **Monitor**: Use metrics and audit logs to track pipeline performance
5. **Learn**: Enable prior learning to improve analysis over time

## Troubleshooting

**Q: Import errors for orchestration module?**  
A: Ensure the orchestration directory is in your Python path. From the repo root: `export PYTHONPATH="${PYTHONPATH}:."`

**Q: Config attributes not found?**  
A: The orchestrator uses `getattr()` with defaults. It will work even if config doesn't have all attributes.

**Q: Audit logs not persisting?**  
A: Ensure the parent directory exists: `mkdir -p data/` before running.

**Q: Learning loop not updating priors?**  
A: Check `config.self_reflection.enable_prior_learning` is `True` and the prior_history_path directory is writable.

## See Also

- [Orchestration Module README](orchestration/README.md) - Detailed component documentation
- [Test Suite](test_orchestration.py) - Unit tests and examples
- [Example Usage](example_orchestration.py) - Working example with fallbacks
