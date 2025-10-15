# Harmonic Front 4: Quick Reference Guide

## What is Harmonic Front 4?

Harmonic Front 4 implements **Adaptive Learning and Self-Reflection** in the CDAF framework, enabling the system to:
- Learn from successful and failed causal inferences
- Refine internal priors based on audit results
- Prioritize adaptive mechanisms for structural failures
- Reduce epistemic uncertainty over iterations

## Key Features

### 1. Circular Causality Detection
**What it does**: Detects when policies create circular dependencies (A → B and B → A)

**Why it matters**: Circular causality indicates logical inconsistencies in the causal model

**How to interpret**:
- Check `harmonic_front_4_audit.total_contradictions` in results
- Look for contradictions with `conflict_type: 'circular_causality'`
- Quality grade: "Excelente" if total_contradictions < 5

### 2. Smart Prior Learning
**What it does**: Adjusts mechanism type priors based on which mechanisms succeed/fail

**Why it matters**: Reduces confidence in "miracle" mechanisms that consistently fail validation

**How to interpret**:
- Monitor `prior_history_path` file for prior evolution
- Look for `penalty_factors` in feedback data
- Check `uncertainty_reduction_percent` over iterations

### 3. Priority-Based Recommendations
**What it does**: Elevates structural failures (CAUSAL_INCOHERENCE, TEMPORAL_CONFLICT) to critical priority

**Why it matters**: Ensures immediate attention to fundamental structural issues

**How to interpret**:
- Recommendations now include `priority_score`
- Critical priority = structural failure requiring immediate action
- Sort by priority: crítica > alta > media > baja

### 4. Uncertainty Tracking
**What it does**: Measures and tracks mean mechanism uncertainty across analyses

**Why it matters**: Validates that the system is actually learning and improving

**How to interpret**:
- Goal: ≥5% reduction over 10 sequential PDM analyses
- Check logs for "Uncertainty reduction over 10 iterations"
- Status: success, needs_improvement, or insufficient_data

## Configuration

Enable Harmonic Front 4 in your configuration:

```yaml
self_reflection:
  enable_prior_learning: true      # Turn on adaptive learning
  feedback_weight: 0.1              # How quickly to adapt (0.1 = 10% new, 90% old)
  prior_history_path: './data/priors.json'  # Where to save learning history
  min_documents_for_learning: 5     # Minimum analyses before adapting
```

## Output Files

### Prior History (`priors.json`)
Tracks learning across multiple PDM analyses:
```json
{
  "version": "2.0",
  "harmonic_front": 4,
  "total_iterations": 15,
  "history": [
    {
      "timestamp": "2025-10-15T...",
      "mechanism_type_priors": {...},
      "uncertainty_reduction_percent": 7.2,
      "test_failures": {
        "necessity_failures": 3,
        "sufficiency_failures": 2
      },
      "penalty_factors": {
        "politico": 0.825,
        "mixto": 0.875
      }
    }
  ]
}
```

### Enhanced Contradiction Reports
Now include Harmonic Front 4 metrics:
```json
{
  "harmonic_front_4_audit": {
    "total_contradictions": 8,
    "causal_incoherence_flags": 3,
    "structural_failures": 2,
    "quality_grade": "Bueno"
  }
}
```

## Interpreting Results

### Quality Grades (D6-Q3 Criteria)
- **Excelente**: total_contradictions < 5
- **Bueno**: total_contradictions < 10
- **Regular**: total_contradictions ≥ 10

### Priority Levels
- **Crítica**: Immediate action required (structural failures)
- **Alta**: Important but not structural
- **Media**: Moderate priority
- **Baja**: Low priority, monitor

### Uncertainty Reduction
- **Success**: ≥5% reduction over 10 iterations
- **Needs Improvement**: <5% reduction, system not learning effectively
- **Insufficient Data**: <10 iterations, keep analyzing

## Common Patterns

### Pattern 1: High Circular Causality
**Symptom**: Many CAUSAL_INCOHERENCE contradictions with circular_causality context

**Meaning**: The PDM has logical loops where interventions depend on their own outcomes

**Action**: Review causal chains, eliminate circular dependencies

### Pattern 2: Miracle Mechanism Penalties
**Symptom**: High penalty_factors for 'politico' or 'mixto' types

**Meaning**: Vague political mechanisms consistently failing validation tests

**Action**: Require more specific mechanism descriptions in future PDMs

### Pattern 3: Stagnant Uncertainty
**Symptom**: Uncertainty reduction <5% over 10 iterations

**Meaning**: System not learning, possibly due to low feedback_weight or poor quality data

**Action**: Increase feedback_weight to 0.15-0.2, ensure diverse PDM analyses

## Troubleshooting

### "Prior learning disabled"
- Check `enable_prior_learning: true` in config
- Ensure `self_reflection` section exists

### "Insufficient data for criterion check"
- Need at least 10 PDM analyses for uncertainty tracking
- Current count shown in logs: "X/10 iterations"

### "No uncertainty reduction"
- Increase `feedback_weight` (e.g., from 0.1 to 0.15)
- Ensure analyzing diverse PDMs, not repeating same one
- Check that mechanisms are actually failing tests

## Best Practices

1. **Start Conservative**: Begin with `feedback_weight: 0.1`
2. **Monitor History**: Regularly check `priors.json` for trends
3. **Diversify Analyses**: Analyze different municipalities/regions for better learning
4. **Review Critical Items**: Always address crítica priority recommendations first
5. **Track Progress**: Monitor uncertainty reduction every 5 analyses

## Support

For questions or issues:
- Review full implementation: `HARMONIC_FRONT_4_IMPLEMENTATION.md`
- Check test suite: `/tmp/test_harmonic_front_4.py`
- Examine logs for detailed feedback messages
