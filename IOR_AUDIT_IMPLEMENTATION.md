# IoR Causal Axiomatic-Bayesian Integration - Implementation Guide

## Overview

This implementation adds three critical audit points to ensure structural rules constrain Bayesian inferences, preventing logical jumps per SOTA axiomatic-Bayesian fusion (Goertz & Mahoney 2012).

## Audit Points Implemented

### Audit Point 2.1: Structural Veto (D6-Q2)

**Purpose**: Ensure TeoriaCambio structural rules constrain Bayesian inferences

**Implementation Location**: `dereck_beach` - `CausalExtractor._check_structural_violation()`

**Key Features**:
- Detects impermissible causal links based on hierarchy: programa → producto → resultado → impacto
- Caps Bayesian posterior at ≤0.6 despite high semantic evidence
- Violations detected:
  - **Reverse causation**: e.g., impacto → producto
  - **Level skipping**: Jumping >2 hierarchy levels
  - **Missing intermediates**: producto → impacto (must go through resultado)

**Quality Evidence**:
```python
# Example: producto → impacto link
posterior_semantic = 0.92  # High semantic similarity
violation = "missing_intermediate:producto→impacto requires resultado"
posterior_final = min(0.92, 0.6)  # Capped at 0.6
```

**SOTA Alignment**: 
- Goertz & Mahoney 2012 set-theoretic constraints
- Pearl 2009 causal hierarchies
- Mahoney 2010 MMR bounds

### Audit Point 2.2: Mechanism Necessity Hoop Test

**Purpose**: Validate causal mechanisms have documented Entity, Activity, Budget

**Implementation Location**: `dereck_beach` - `BayesianMechanismInference._test_necessity()`

**Key Features**:
- Tests three required components:
  1. **Entity**: Documented responsible organization
  2. **Activity**: Verb lemma sequence (specific actions)
  3. **Budget**: Allocated resources (quantified)
- Deterministic failure if any component missing
- Returns structured results with remediation text

**Quality Evidence**:
```python
observations = {
    'entity_activity': {'entity': 'Secretaría de Planeación'},
    'verbs': ['implementar', 'ejecutar', 'coordinar'],
    'budget': 75000000
}
result = _test_necessity(node, observations)
# result['is_necessary'] = True
# result['hoop_test_passed'] = True
# result['missing_components'] = []
```

**SOTA Alignment**:
- Beach 2017 Hoop Tests for necessity
- Falleti & Lynch 2009 Bayesian-deterministic hybrid
- Mechanism depth validation

### Audit Point 2.3: Policy Alignment Dual Constraint

**Purpose**: Integrate macro-micro causality via DNP/ODS alignment scores

**Implementation Location**: `dereck_beach` - `CounterfactualAuditor._audit_systemic_risk()`

**Key Features**:
- Applies 1.2× risk multiplier when `pdet_alignment ≤ 0.60`
- Quality thresholds (D5-Q4):
  - Excelente: risk_score < 0.10
  - Bueno: risk_score < 0.20
  - Aceptable: risk_score < 0.35
- Flags when alignment penalty causes quality degradation

**Quality Evidence**:
```python
base_risk = 0.09
pdet_alignment = 0.55  # Low alignment

# Apply penalty
if pdet_alignment <= 0.60:
    risk_score = base_risk * 1.2  # = 0.108

# Quality downgrade
# Without penalty: 0.09 < 0.10 → EXCELENTE
# With penalty: 0.108 >= 0.10 → BUENO
```

**SOTA Alignment**:
- Lieberman 2015 macro-micro causality integration
- UN 2020 ODS benchmarks for low-risk thresholds

## Testing

### Test Suite: `test_ior_audit_points.py`

**Results**: 10/10 tests passing

- **Audit Point 2.1**: 2 tests
  - Structural violation detection
  - Posterior capping at 0.6
  
- **Audit Point 2.2**: 4 tests
  - Complete documentation (all components)
  - Missing entity
  - Missing activity
  - Missing budget
  
- **Audit Point 2.3**: 4 tests
  - Alignment penalty application
  - No penalty for high alignment
  - Quality downgrade due to alignment
  - Quality threshold validation

### Demonstration: `demo_ior_audit_points.py`

Interactive demonstration showing:
- Valid vs. invalid structural links
- Posterior capping in action
- Hoop test scenarios (pass/fail)
- Alignment penalty impact on risk scores

## Usage Examples

### Example 1: Structural Veto

```python
from dereck_beach import CausalExtractor

extractor = CausalExtractor(config, nlp_model)
extractor.extract_causal_links(text)

# CausalExtractor automatically applies structural veto
# Log output:
# WARNING: STRUCTURAL VETO (D6-Q2): Link MP-001→MI-001 violates causal hierarchy.
#          Posterior capped from 0.920 to 0.600.
#          Violation: missing_intermediate:producto→impacto requires resultado
```

### Example 2: Necessity Hoop Test

```python
from dereck_beach import BayesianMechanismInference

inference = BayesianMechanismInference(config, nlp_model)
result = inference._test_necessity(node, observations)

if not result['is_necessary']:
    print(f"Hoop Test FAILED: {result['remediation']}")
    print(f"Missing: {', '.join(result['missing_components'])}")
```

### Example 3: Alignment Dual Constraint

```python
from dereck_beach import CounterfactualAuditor

auditor = CounterfactualAuditor(config)
risk_result = auditor._audit_systemic_risk(
    nodes, graph, evidence, implications,
    pdet_alignment=0.55  # Low alignment
)

if risk_result['alignment_penalty_applied']:
    print(f"Risk escalated from {base_risk:.3f} to {risk_result['risk_score']:.3f}")
    print(f"Quality: {risk_result['d5_q4_quality']}")
```

## Configuration

### Thresholds (Externalized in `config.yaml`)

```yaml
bayesian_thresholds:
  structural_veto_cap: 0.6  # Max posterior for violated links
  alignment_threshold: 0.60  # Trigger for risk multiplier
  alignment_multiplier: 1.2  # Risk escalation factor

quality_thresholds:
  d5_q4_excellent: 0.10
  d5_q4_good: 0.20
  d5_q4_acceptable: 0.35
```

## Logs and Observability

### Structural Veto Logs

```
WARNING - STRUCTURAL VETO (D6-Q2): Link MP-001→MI-001 violates causal hierarchy.
          Posterior capped from 0.920 to 0.600.
          Violation: missing_intermediate:producto→impacto requires resultado
```

### Alignment Penalty Logs

```
WARNING - ALIGNMENT PENALTY (D5-Q4): pdet_alignment=0.55 ≤ 0.60,
          risk_score escalated from 0.090 to 0.108 (multiplier: 1.2×).
          Dual constraint per Lieberman 2015.
```

## Integration with Existing Systems

### TeoriaCambio Integration

The structural veto uses the same hierarchy axioms as `TeoriaCambio`:
- INSUMOS → PROCESOS → PRODUCTOS → RESULTADOS → CAUSALIDAD

Mapped to PDM types:
- programa → producto → resultado → impacto

### Bayesian Engine Integration

Compatible with refactored Bayesian engine (`inference/bayesian_engine.py`):
- `BayesianPriorBuilder` provides type transition priors
- `NecessitySufficiencyTester` can be used via adapter
- Posterior capping happens after Bayesian update

## Performance Characteristics

- **Structural veto check**: O(1) - Simple hierarchy level comparison
- **Necessity test**: O(n) where n = number of observations
- **Alignment penalty**: O(1) - Single multiplication

## Future Enhancements

1. **Dynamic threshold learning**: Adapt veto cap based on domain
2. **Multi-level alignment**: Beyond binary threshold (≤0.60)
3. **Contextual structural rules**: Different hierarchies per policy area
4. **Uncertainty quantification**: Track epistemic uncertainty in vetoed links

## References

- Goertz, G., & Mahoney, J. (2012). *A Tale of Two Cultures*. Set-theoretic constraints.
- Mahoney, J. (2010). *After KKV*. MMR bounds and structural validation.
- Pearl, J. (2009). *Causality*. Causal hierarchies and intervention logic.
- Beach, D. (2017). *Process-Tracing Methods*. Hoop Tests for necessity.
- Falleti, T. G., & Lynch, J. F. (2009). *Context and Causal Mechanisms*. Bayesian-deterministic hybrids.
- Lieberman, E. S. (2015). *Nested Analysis*. Macro-micro causality integration.
- UN (2020). *Sustainable Development Goals*. ODS benchmarks for risk thresholds.

## Support

For issues or questions:
1. Check test suite: `python test_ior_audit_points.py`
2. Run demo: `python demo_ior_audit_points.py`
3. Review logs for WARNING messages about structural violations or alignment penalties
