# Risk Mitigation Layer - Documentation

## Overview

The Risk Mitigation Layer provides pre-execution risk assessment and automated mitigation for the FARFAN 2.0 pipeline. It evaluates stage-specific detector predicates, invokes mitigation strategies, and implements severity-based escalation policies.

## Architecture

### Core Components

1. **RiskRegistry**: Central storage for risk definitions organized by pipeline stage
2. **RiskMitigationLayer**: Orchestrates risk assessment and mitigation execution
3. **Risk Detection**: Evaluates predicates before stage execution
4. **Mitigation Strategies**: Automated remediation with retry logic
5. **Structured Logging**: Comprehensive event capture for auditing

## Risk Severity Levels

### Severity-Based Escalation Policy

| Severity | Max Attempts | Behavior | Exception on Failure |
|----------|-------------|----------|---------------------|
| **CRITICAL** | 1 | Abort immediately if mitigation fails | `CriticalRiskUnmitigatedException` |
| **HIGH** | 2 | Retry once before aborting | `HighRiskUnmitigatedException` |
| **MEDIUM** | 3 | Retry twice with fallback options | None (documented degradation) |
| **LOW** | 1 | Use documented fallback only | None (documented degradation) |

## Risk Categories

### Stage-Specific Categories

#### Stage 1-2: Document Extraction
- `PDF_CORRUPTED`: Unreadable or corrupted PDF file
- `PDF_UNREADABLE`: PDF cannot be parsed
- `MISSING_SECTIONS`: Insufficient document sections identified
- `EMPTY_DOCUMENT`: Document has insufficient content

#### Stage 3: Semantic Analysis
- `NLP_MODEL_UNAVAILABLE`: Required NLP model not loaded
- `TEXT_TOO_SHORT`: Insufficient text for analysis
- `ENCODING_ERROR`: Character encoding issues

#### Stage 4: Causal Extraction
- `NO_CAUSAL_CHAINS`: No causal relationships detected
- `GRAPH_DISCONNECTED`: Causal graph has disconnected components
- `INSUFFICIENT_NODES`: Too few nodes for meaningful analysis

#### Stage 5: Mechanism Inference
- `BAYESIAN_INFERENCE_FAILURE`: Bayesian model inference failed
- `INSUFFICIENT_OBSERVATIONS`: Not enough data for inference

#### Stage 6: Financial Audit
- `MISSING_BUDGET_DATA`: No budget information found
- `BUDGET_INCONSISTENCY`: Budget totals don't match
- `NEGATIVE_ALLOCATIONS`: Invalid negative budget values

#### Stage 7: DNP Validation
- `DNP_STANDARDS_VIOLATION`: DNP compliance standards not met
- `COMPETENCIA_MISMATCH`: Municipal competence misalignment
- `MISSING_MGA_INDICATORS`: Required MGA indicators absent

#### Stage 8: Question Answering
- `INSUFFICIENT_EVIDENCE`: Weak evidence for answers
- `MODULE_UNAVAILABLE`: Required module not initialized

#### Stage 9: Report Generation
- `REPORT_GENERATION_FAILURE`: Report creation failed
- `DATA_SERIALIZATION_ERROR`: Cannot serialize output data

## Usage

### Basic Setup

```python
from risk_mitigation_layer import (
    RiskRegistry, RiskMitigationLayer,
    create_default_risk_registry
)

# Create registry with predefined common risks
registry = create_default_risk_registry()

# Initialize mitigation layer
mitigation_layer = RiskMitigationLayer(registry)
```

### Registering Custom Risks

```python
from risk_mitigation_layer import Risk, RiskSeverity, RiskCategory

# Define a custom risk
custom_risk = Risk(
    category=RiskCategory.PDF_CORRUPTED,
    severity=RiskSeverity.HIGH,
    probability=0.15,
    impact=0.9,
    detector_predicate=lambda ctx: not hasattr(ctx, 'pdf_path'),
    mitigation_strategy=lambda ctx: "Requesting alternative document source",
    description="PDF file path is not accessible"
)

# Register it for a specific stage
registry.register_risk("STAGE_1_2", custom_risk)
```

### Wrapping Stage Execution

```python
def my_stage_function(context):
    # Your stage logic here
    context.processed = True
    return context

# Wrap with risk assessment
result = mitigation_layer.wrap_stage_execution(
    stage="STAGE_4",
    stage_function=my_stage_function,
    context=pipeline_context
)
```

### Handling Exceptions

```python
from risk_mitigation_layer import (
    CriticalRiskUnmitigatedException,
    HighRiskUnmitigatedException
)

try:
    result = mitigation_layer.wrap_stage_execution(
        stage="STAGE_1_2",
        stage_function=extract_document,
        context=ctx
    )
except CriticalRiskUnmitigatedException as e:
    logger.critical(f"Abort: {e.risk.category.value}")
    # Cannot continue - critical risk unmitigated
    raise
    
except HighRiskUnmitigatedException as e:
    logger.error(f"Abort after retry: {e.risk.category.value}")
    # High risk persisted after retry
    raise
```

## Structured Logging

### Log Event Types

#### Risk Detection
```
[RISK DETECTED] category=empty_document, severity=CRITICAL, probability=0.10, 
impact=1.00, score=0.10, description=Documento vacío o con menos de 100 caracteres
```

#### Mitigation Attempt
```
[MITIGATION ATTEMPT] attempt=1/2, category=no_causal_chains
```

#### Mitigation Success
```
[MITIGATION SUCCESS] category=missing_sections, attempt=2/3, 
outcome=Fallback strategy applied successfully
```

#### Mitigation Failure
```
[MITIGATION FAILED] category=pdf_corrupted, attempt=1/1, 
error=File cannot be recovered
```

#### Mitigation Complete
```
[MITIGATION COMPLETE] category=insufficient_nodes, success=True, 
attempts=2, time=0.45s
```

#### Stage Wrapper
```
[STAGE WRAPPER] Iniciando wrapper para etapa: STAGE_4
[STAGE WRAPPER] 2 riesgos detectados. Iniciando proceso de mitigación.
[STAGE WRAPPER] Etapa STAGE_4 completada exitosamente en 1.23s
```

#### Degradation Documentation
```
[DEGRADATION DOCUMENTED] missing_budget_data en etapa STAGE_6
```

## Mitigation Results

### MitigationResult Dataclass

```python
@dataclass
class MitigationResult:
    risk: Risk                    # The risk being mitigated
    success: bool                 # Whether mitigation succeeded
    attempts: int                 # Number of attempts made
    error_message: Optional[str]  # Error if failed
    outcome_description: str      # Human-readable outcome
    mitigation_time: float        # Time spent in seconds
    timestamp: str                # ISO format timestamp
```

### Getting Mitigation Report

```python
report = mitigation_layer.get_mitigation_report()

print(f"Total mitigations: {report['total_mitigations']}")
print(f"Success rate: {report['success_rate']:.1%}")
print(f"Average time: {report['average_time']:.2f}s")

# By severity breakdown
for severity, stats in report['by_severity'].items():
    print(f"{severity}: {stats['successful']}/{stats['total']}")

# Detailed events
for detail in report['details']:
    print(f"{detail['category']}: {detail['outcome']}")
```

## Integration with Orchestrator

### Enhanced Orchestrator Pattern

```python
from risk_mitigation_layer import (
    RiskMitigationLayer, create_default_risk_registry
)

class FARFANOrchestratorWithRisks:
    def __init__(self):
        # Initialize risk mitigation
        self.risk_registry = create_default_risk_registry()
        self.mitigation_layer = RiskMitigationLayer(self.risk_registry)
        
        # Register custom risks
        self._register_application_risks()
    
    def _register_application_risks(self):
        """Register application-specific risks"""
        # Add custom risk definitions here
        pass
    
    def process_plan(self, pdf_path, policy_code):
        ctx = PipelineContext(pdf_path=pdf_path, policy_code=policy_code)
        
        try:
            # Each stage wrapped with risk assessment
            ctx = self.mitigation_layer.wrap_stage_execution(
                "STAGE_1_2", self._stage_extract_document, ctx
            )
            
            ctx = self.mitigation_layer.wrap_stage_execution(
                "STAGE_3", self._stage_semantic_analysis, ctx
            )
            
            # ... more stages ...
            
        except (CriticalRiskUnmitigatedException, 
                HighRiskUnmitigatedException) as e:
            logger.error(f"Pipeline aborted: {e}")
            raise
        
        return ctx
```

## Degradation Documentation

When MEDIUM or LOW severity risks are not fully mitigated, the system continues execution but documents the degradation for transparency:

```python
# Degradation is automatically added to context
ctx.degradations = [
    {
        'stage': 'STAGE_6',
        'category': 'missing_budget_data',
        'severity': 'MEDIUM',
        'description': 'No se encontraron datos presupuestarios',
        'mitigation_attempted': True,
        'mitigation_success': False,
        'impact_on_results': 'Results may have reduced quality or completeness',
        'timestamp': '2024-01-15T10:30:45.123456'
    }
]
```

## Risk Score Calculation

Each risk has a calculated score based on probability and impact:

```python
risk_score = probability × impact

# Example:
# probability=0.8, impact=0.9 → score=0.72 (high concern)
# probability=0.2, impact=0.3 → score=0.06 (low concern)
```

Risks are evaluated in order of:
1. Severity (CRITICAL > HIGH > MEDIUM > LOW)
2. Risk score (descending)

## Best Practices

### 1. Define Specific Detectors

```python
# Good: Specific, measurable condition
detector_predicate=lambda ctx: len(ctx.causal_chains) == 0

# Avoid: Vague or unmeasurable
detector_predicate=lambda ctx: ctx.quality < "good"
```

### 2. Implement Effective Mitigations

```python
# Good: Actually attempts to fix the problem
def mitigation_strategy(ctx):
    # Try alternative extraction method
    ctx.causal_chains = extract_chains_fallback(ctx.raw_text)
    return "Applied fallback extraction"

# Avoid: Just logging without action
mitigation_strategy=lambda ctx: "Problem noted"
```

### 3. Set Appropriate Severity

- **CRITICAL**: Pipeline cannot proceed (missing file, corrupted data)
- **HIGH**: Major functionality impaired (no causal chains, missing model)
- **MEDIUM**: Degraded results possible (few sections, limited evidence)
- **LOW**: Minor quality impact (performance concerns, optional features)

### 4. Monitor Mitigation Reports

```python
# After pipeline execution
report = mitigation_layer.get_mitigation_report()

if report['failed'] > 0:
    logger.warning(f"{report['failed']} mitigations failed")
    # Review failures and adjust strategies

if report['success_rate'] < 0.7:
    logger.warning("Low mitigation success rate")
    # Investigate risk definitions
```

## Testing

Run the test suite:

```bash
python test_risk_mitigation.py
```

Run integration examples:

```bash
python example_risk_integration.py
```

## Files

- `risk_mitigation_layer.py` (635 lines): Core implementation
- `test_risk_mitigation.py` (233 lines): Unit tests
- `example_risk_integration.py` (310 lines): Integration examples
- `RISK_MITIGATION_DOCS.md`: This documentation

## API Reference

### RiskRegistry

```python
class RiskRegistry:
    def register_risk(self, stage: str, risk: Risk)
    def get_risks_for_stage(self, stage: str) -> List[Risk]
    def get_all_risks(self) -> Dict[str, List[Risk]]
    def get_statistics(self) -> Dict[str, Any]
```

### RiskMitigationLayer

```python
class RiskMitigationLayer:
    def __init__(self, registry: RiskRegistry)
    
    def assess_stage_risks(self, stage: str, context: Any) -> List[Risk]
    
    def execute_mitigation(self, risk: Risk, context: Any) -> MitigationResult
    
    def wrap_stage_execution(
        self, 
        stage: str,
        stage_function: Callable[[Any], Any],
        context: Any
    ) -> Any
    
    def get_mitigation_report(self) -> Dict[str, Any]
```

### Utility Functions

```python
def create_default_risk_registry() -> RiskRegistry
    """Creates registry with predefined common risks"""
```

## Performance Considerations

- Risk detection adds minimal overhead (~1-5ms per stage)
- Mitigation time varies by strategy complexity
- All timing is logged in mitigation results
- Use `get_mitigation_report()` to analyze performance

## Future Enhancements

Potential improvements:

1. **Adaptive Thresholds**: Learn optimal probability/impact from history
2. **Risk Prediction**: ML model to predict risks before detection
3. **Mitigation Chaining**: Compose multiple strategies
4. **Risk Dependencies**: Model relationships between risks
5. **Dynamic Severity**: Adjust severity based on context
6. **Async Mitigation**: Parallel mitigation for independent risks
7. **Rollback Support**: Undo failed mitigations
8. **Risk Simulation**: Test strategies before deployment
