# Bayesian Inference Implementation - Three AGUJAS

This document describes the implementation of three "agujas" (needles) that transform the FARFAN framework from a deterministic pattern extractor to a Bayesian causal inference engine.

## Overview

The implementation adds three major innovations:

1. **AGUJA I**: Adaptive Bayesian Prior in causal link extraction
2. **AGUJA II**: Hierarchical Bayesian model for mechanism inference
3. **AGUJA III**: Bayesian counterfactual auditing

## AGUJA I: El Prior Informado Adaptativo

**Location**: `CausalExtractor._extract_causal_links()` method

**Purpose**: Transform causal link extraction from deterministic regex matching to Bayesian inference with adaptive priors.

### Key Changes

Instead of assigning a fixed strength (0.8) to causal links, the system now:

1. **Initializes adaptive priors** based on:
   - Semantic distance between nodes (using spaCy embeddings)
   - Historical transition frequencies between goal types
   - Causal language specificity (epistemic certainty)

2. **Performs incremental Bayesian updating**:
   ```
   P(causality | evidence) ∝ P(evidence | causality) × P(causality)
   ```

3. **Calculates composite likelihood** from multiple evidence sources:
   - Temporal coherence (logical verb sequences)
   - Financial consistency (budget alignment)
   - Textual proximity (co-occurrence in context windows)
   - Entity-Activity validation

4. **Tracks convergence** using KL divergence between iterations

### New Methods

- `_calculate_semantic_distance()`: Compute cosine similarity between node embeddings
- `_calculate_type_transition_prior()`: Get prior probabilities for goal type transitions
- `_calculate_language_specificity()`: Assess epistemic certainty of causal keywords
- `_assess_temporal_coherence()`: Validate logical verb sequences
- `_assess_financial_consistency()`: Check budget alignment
- `_calculate_textual_proximity()`: Measure co-occurrence frequencies
- `_initialize_prior()`: Set up Beta distribution priors
- `_calculate_composite_likelihood()`: Combine multiple evidence sources

### Output

Each causal link now includes:
- `posterior_mean`: Bayesian posterior probability
- `posterior_std`: Uncertainty quantification
- `posterior_alpha` / `posterior_beta`: Beta distribution parameters
- `kl_divergence`: Convergence metric
- `converged`: Boolean convergence status
- `evidence_count`: Number of evidence pieces

## AGUJA II: El Modelo Generativo de Mecanismos

**Location**: New `BayesianMechanismInference` class (inserted between `MechanismPartExtractor` and `CausalInferenceSetup`)

**Purpose**: Infer latent causal mechanisms using hierarchical Bayesian modeling.

### Architecture

Three-level hierarchical model:

1. **Level 3 (Hyperprior)**: Distribution of mechanism types (administrativo, técnico, financiero, político)
2. **Level 2 (Prior)**: Typical activity sequences per mechanism type
3. **Level 1 (Likelihood)**: Observed textual evidence

### Latent Variables Inferred

- `z_mechanism`: Mechanism type posterior distribution
- `θ_sequence`: Activity sequence transition probabilities
- `φ_coherence`: Mechanism coherence score

### Key Features

1. **Mechanism Type Inference**: Bayesian updating of mechanism type probabilities based on observed verbs and entities

2. **Activity Sequence Modeling**: Simplified Markov chain inference for activity transitions

3. **Coherence Scoring**: Multi-factor assessment of mechanism plausibility

4. **Validation Tests**:
   - **Sufficiency Test**: Can this mechanism produce the outcome?
   - **Necessity Test**: Is this mechanism unique or are alternatives possible?

5. **Uncertainty Quantification**: Entropy-based epistemic uncertainty across:
   - Mechanism type distribution
   - Sequence completeness
   - Coherence factors

6. **Gap Detection**: Automatic identification of documentation omissions with severity levels

### Methods

- `infer_mechanisms()`: Main entry point for mechanism inference
- `_infer_single_mechanism()`: Infer complete mechanism for one node
- `_extract_observations()`: Extract textual evidence from context
- `_infer_mechanism_type()`: Bayesian update of mechanism type distribution
- `_infer_activity_sequence()`: Markov chain parameter inference
- `_calculate_coherence_factor()`: Multi-component coherence score
- `_test_sufficiency()`: Sufficiency validation
- `_test_necessity()`: Necessity validation
- `_quantify_uncertainty()`: Entropy-based uncertainty metrics
- `_detect_gaps()`: Gap identification with remediation suggestions

### Output

For each product node:
```json
{
  "mechanism_type": {"administrativo": 0.45, "técnico": 0.35, ...},
  "activity_sequence": {
    "expected_sequence": [...],
    "observed_verbs": [...],
    "transition_probabilities": {...},
    "sequence_completeness": 0.75
  },
  "coherence_score": 0.82,
  "sufficiency_test": {"score": 0.85, "is_sufficient": true},
  "necessity_test": {"score": 0.65, "is_necessary": false},
  "uncertainty": {
    "total": 0.35,
    "mechanism_type": 0.40,
    "sequence": 0.30,
    "coherence": 0.18
  },
  "gaps": [...]
}
```

## AGUJA III: El Auditor Contrafactual Bayesiano

**Location**: New method `bayesian_counterfactual_audit()` in `OperationalizationAuditor` class

**Purpose**: Perform counterfactual causal reasoning to detect what should be present but is missing.

### Approach

Uses Pearl's do-calculus framework for counterfactual inference:
```
P(failure | do(omit_E), context) vs P(failure | do(include_E), context)
```

### Three-Layer Audit

1. **Layer 1 - Direct Evidence**:
   - Check for required attributes (baseline, target, entity, budget, mechanism)
   - Calculate failure probability for each omission using historical priors
   - Classify omission severity (critical, high, medium, low)

2. **Layer 2 - Causal Implications**:
   - Infer secondary effects of omissions
   - Example: Missing baseline → P(target_miscalibrated) = 0.73
   - Identify cascade risks to dependent nodes

3. **Layer 3 - Systemic Risk**:
   - Calculate graph-level centrality metrics
   - Assess P(cascade_failure) from accumulated omissions
   - Estimate overall success probability

### Optimal Remediation

Uses Expected Value of Information to prioritize fixes:
```
EVI(remediation) = P(failure_avoided) × Impact / Effort
```

Recommendations are ordered by priority to maximize risk reduction with minimal effort.

### Methods

- `bayesian_counterfactual_audit()`: Main audit orchestrator
- `_build_normative_dag()`: Construct SCM of expected relationships
- `_get_default_historical_priors()`: Default success rates by attribute
- `_audit_direct_evidence()`: Layer 1 audit
- `_audit_causal_implications()`: Layer 2 causal reasoning
- `_audit_systemic_risk()`: Layer 3 system-wide analysis
- `_generate_optimal_remediations()`: EVI-based prioritization

### Output

```json
{
  "direct_evidence": {...},
  "causal_implications": {...},
  "systemic_risk": {
    "risk_score": 0.42,
    "success_probability": 0.68,
    "critical_omissions": [...],
    "completeness": 0.73
  },
  "recommendations": [
    {
      "node_id": "MP-001",
      "omission": "baseline",
      "severity": "critical",
      "expected_value": 0.85,
      "effort": 3,
      "priority": 0.28,
      "recommendation": "..."
    },
    ...
  ],
  "summary": {...}
}
```

## Integration with Main Framework

The three AGUJAS are integrated into the `CDAFFramework.process_document()` pipeline:

1. **Step 2**: Causal hierarchy extraction uses AGUJA I (Bayesian link inference)
2. **Step 4.5**: AGUJA II infers mechanisms after financial traceability
3. **Step 5.5**: AGUJA III performs counterfactual audit after operationalization audit
4. **Step 8**: New `_generate_bayesian_reports()` method creates:
   - `{policy_code}_bayesian_mechanisms.json`
   - `{policy_code}_counterfactual_audit.json`
   - `{policy_code}_bayesian_summary.md`

## Benefits

### From Deterministic to Probabilistic

**Before**: Binary decisions, fixed strengths, no uncertainty
**After**: Probability distributions, quantified uncertainty, confidence intervals

### Causal Reasoning

**Before**: Pattern matching
**After**: Formal causal inference with do-calculus

### Actionable Insights

**Before**: List of errors
**After**: Prioritized recommendations with expected impact

### Learning System

**Before**: Static rules
**After**: Adaptive priors that can incorporate historical data

## Technical Requirements

- NumPy for numerical computation
- SciPy for statistical functions (Beta distribution, KL divergence)
- spaCy for semantic embeddings
- NetworkX for graph analysis (already required)

## Usage Example

```python
from pathlib import Path

# Initialize framework
framework = CDAFFramework(
    config_path=Path("config.yaml"),
    output_dir=Path("results"),
    log_level="INFO"
)

# Process document (now with Bayesian inference)
success = framework.process_document(
    pdf_path=Path("plan.pdf"),
    policy_code="PDM_2024"
)

# Outputs include:
# - results/PDM_2024_bayesian_mechanisms.json
# - results/PDM_2024_counterfactual_audit.json
# - results/PDM_2024_bayesian_summary.md
```

## Future Enhancements

1. **MCMC Sampling**: Replace approximate inference with full MCMC (using PyMC3 or NumPyro)
2. **Historical Learning**: Accumulate priors from multiple documents
3. **Causal Discovery**: Infer DAG structure, not just parameters
4. **Interactive What-If**: Allow users to simulate interventions
5. **Cross-Document Meta-Analysis**: Compare plans in latent mechanism space

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
- Gelman, A. et al. (2013). *Bayesian Data Analysis*
- Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models*

---

*Implementation by: AI Systems Architect*  
*Date: 2025-10-14*  
*Framework Version: 2.0.0*
