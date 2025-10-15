# Orchestration Module

This module implements the orchestration layer for the CDAF Framework, providing state machine-based execution control and adaptive learning capabilities.

## Components

### F2.1: PDMOrchestrator (pdm_orchestrator.py)

The `PDMOrchestrator` class is the master orchestrator that executes Phase 0-IV with complete observability.

**Features:**
- **Explicit State Machine**: Tracks analysis state through well-defined states (INITIALIZED, EXTRACTING, BUILDING_DAG, INFERRING_MECHANISMS, VALIDATING, FINALIZING, COMPLETED, FAILED)
- **Backpressure Management**: Uses asyncio queues and semaphores to control concurrency
- **Timeout Enforcement**: Configurable timeouts for worker processes
- **Metrics Collection**: Comprehensive metrics tracking for observability
- **Immutable Audit Logging**: Governance-compliant audit trail with SHA256 hashing
- **Quality Gates**: Enforces quality thresholds at each phase

**Key Classes:**
- `PDMAnalysisState`: Enum defining state machine states
- `PDMOrchestrator`: Main orchestrator class
- `AnalysisResult`: Complete analysis result container
- `MetricsCollector`: Metrics tracking and alerting
- `ImmutableAuditLogger`: Append-only audit log

**Usage:**
```python
from orchestration.pdm_orchestrator import PDMOrchestrator, PDMAnalysisState

config = YourConfigClass()
orchestrator = PDMOrchestrator(config)

# Inject pipeline components
orchestrator.extraction_pipeline = extraction_pipeline
orchestrator.causal_builder = causal_builder
orchestrator.bayesian_engine = bayesian_engine
orchestrator.validator = validator
orchestrator.scorer = scorer

# Run analysis
result = await orchestrator.analyze_plan("path/to/plan.pdf")
print(f"Quality Score: {result.quality_score.overall_score}")
```

### F2.2: AdaptiveLearningLoop (learning_loop.py)

The `AdaptiveLearningLoop` implements Front D.1: Prior Learning from historical failures.

**Features:**
- **Prior History Management**: Stores and manages Bayesian priors for mechanism types
- **Feedback Extraction**: Extracts learning signals from analysis results
- **Prior Decay**: Reduces priors for mechanism types that fail necessity tests
- **Prior Boost**: Increases priors for successful mechanism types
- **Immutable Snapshots**: Maintains audit trail of prior updates
- **Configurable Learning**: Enable/disable learning via configuration

**Key Classes:**
- `AdaptiveLearningLoop`: Main learning loop coordinator
- `PriorHistoryStore`: Persistent storage for priors with snapshot capability
- `FeedbackExtractor`: Extracts feedback from analysis results
- `MechanismPrior`: Prior distribution data structure

**Usage:**
```python
from orchestration.learning_loop import AdaptiveLearningLoop

config = YourConfigClass()
learning_loop = AdaptiveLearningLoop(config)

# After analysis
analysis_result = await orchestrator.analyze_plan("plan.pdf")

# Update priors based on results
learning_loop.extract_and_update_priors(analysis_result)

# Query current prior
alpha = learning_loop.get_current_prior("mechanism_type")

# Get history
history = learning_loop.get_prior_history("mechanism_type")
```

## Configuration

The orchestration module expects a configuration object with the following structure:

```python
@dataclass
class Config:
    # Queue management
    queue_size: int = 10
    max_inflight_jobs: int = 3
    
    # Timeouts
    worker_timeout_secs: int = 300
    
    # Quality thresholds
    min_quality_threshold: float = 0.5
    
    # Learning
    prior_decay_factor: float = 0.9
    
    @dataclass
    class SelfReflection:
        enable_prior_learning: bool = True
        prior_history_path: str = "data/prior_history.json"
    
    self_reflection: SelfReflection = field(default_factory=SelfReflection)
```

## Phase Execution Flow

1. **INITIALIZED**: Orchestrator created, ready to process
2. **EXTRACTING** (Phase I): Extract semantic chunks and tables from PDF
3. **BUILDING_DAG** (Phase II): Construct causal graph from extracted data
4. **INFERRING_MECHANISMS** (Phase III): Run Bayesian mechanism inference in parallel with validation
5. **VALIDATING** (Phase III): Validate graph against axioms and ontology
6. **FINALIZING** (Phase IV): Calculate quality scores and generate recommendations
7. **COMPLETED**: Analysis successfully completed
8. **FAILED**: Analysis failed (timeout or error)

## Metrics and Observability

The orchestrator tracks comprehensive metrics:

- **Extraction metrics**: chunk_count, table_count, quality scores
- **Graph metrics**: node_count, edge_count
- **Mechanism metrics**: prior_decay_rate, hoop_test_fail_count
- **Dimension metrics**: avg_score_D1 through avg_score_D6
- **Pipeline metrics**: duration_seconds, timeout_count, error_count

Metrics can trigger alerts:
- CRITICAL alert when D6 score < 0.55
- Manual review holds for governance compliance

## Audit Trail

Every analysis run generates an immutable audit record containing:
- `run_id`: Unique identifier
- `timestamp`: ISO 8601 timestamp
- `sha256_source`: SHA256 hash of input PDF
- `duration_seconds`: Total execution time
- `final_state`: Final state machine state
- `result_summary`: Key result metrics

Audit records are append-only and stored in JSONL format.

## Testing

Run tests with:
```bash
python3 test_orchestration.py
```

See example usage:
```bash
python3 example_orchestration.py
```

## Integration with Existing Framework

The orchestration module integrates with the existing CDAF framework through dependency injection:

```python
from dereck_beach import ConfigLoader, CDAFFramework
from orchestration import PDMOrchestrator, AdaptiveLearningLoop

# Load existing config
config = ConfigLoader(Path("config.yaml"))

# Create orchestrator
orchestrator = PDMOrchestrator(config)

# Inject existing components from CDAFFramework
framework = CDAFFramework(config_path, output_dir)
orchestrator.extraction_pipeline = framework.pdf_processor
orchestrator.causal_builder = framework.causal_extractor
orchestrator.bayesian_engine = framework.bayesian_mechanism
orchestrator.validator = framework.op_auditor
orchestrator.scorer = framework.reporting_engine

# Create learning loop
learning_loop = AdaptiveLearningLoop(config)

# Run analysis with learning
result = await orchestrator.analyze_plan("plan.pdf")
learning_loop.extract_and_update_priors(result)
```

## Benefits

### Control Total del Flujo
- Explicit state machine provides clear visibility into pipeline execution
- Timeout enforcement prevents runaway processes
- Backpressure prevents resource exhaustion

### Métricas Integradas
- Comprehensive metrics collection at every phase
- Configurable alerting for critical thresholds
- Observable through metrics summary API

### Cumplimiento de Governance Standards
- Immutable audit trail with SHA256 verification
- Manual review holds for human-in-the-loop governance
- Prior history tracking for reproducibility

### Aprendizaje Adaptativo
- Automatic prior adjustment based on historical performance
- Decay for failing mechanism types
- Boost for successful patterns
- Configurable learning rate and decay factor
