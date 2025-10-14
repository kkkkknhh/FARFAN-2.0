# Bayesian Inference Quick Reference

## Quick Start

```python
from pathlib import Path
from dereck_beach import CDAFFramework

# Initialize framework (now with Bayesian inference)
framework = CDAFFramework(
    config_path=Path("config.yaml"),
    output_dir=Path("results"),
    log_level="INFO"
)

# Process document - Bayesian inference runs automatically
framework.process_document(
    pdf_path=Path("plan.pdf"),
    policy_code="PDM_2024"
)

# Check outputs:
# - results/PDM_2024_bayesian_mechanisms.json
# - results/PDM_2024_counterfactual_audit.json
# - results/PDM_2024_bayesian_summary.md
```

## Key Classes & Methods

### AGUJA I: Bayesian Causal Links

**Class**: `CausalExtractor`  
**Method**: `_extract_causal_links(text)`

**What it returns**: Updates `self.causal_chains` with:
```python
{
    'source': 'MP-001',
    'target': 'MR-001',
    'strength': 0.795,              # Posterior mean
    'posterior_mean': 0.795,
    'posterior_std': 0.165,         # Uncertainty
    'kl_divergence': 0.008,         # Convergence metric
    'converged': True,              # Converged?
    'evidence': ['permite', 'contribuye']
}
```

**Helper methods**:
- `_calculate_semantic_distance(source, target)` → float
- `_calculate_type_transition_prior(source, target)` → float
- `_calculate_language_specificity(keyword)` → float
- `_assess_temporal_coherence(source, target)` → float
- `_assess_financial_consistency(source, target)` → float
- `_calculate_textual_proximity(source, target, text)` → float

### AGUJA II: Mechanism Inference

**Class**: `BayesianMechanismInference`  
**Method**: `infer_mechanisms(nodes, text)`

**What it returns**: Dict mapping node_id → mechanism info:
```python
{
    'MP-001': {
        'mechanism_type': {
            'administrativo': 0.50,
            'técnico': 0.25,
            'financiero': 0.15,
            'político': 0.10
        },
        'activity_sequence': {
            'expected_sequence': ['planificar', 'ejecutar', 'evaluar'],
            'observed_verbs': ['planificar', 'coordinar'],
            'sequence_completeness': 0.67
        },
        'coherence_score': 0.82,
        'sufficiency_test': {
            'score': 0.85,
            'is_sufficient': True
        },
        'necessity_test': {
            'score': 0.65,
            'is_necessary': False
        },
        'uncertainty': {
            'total': 0.35,
            'mechanism_type': 0.40,
            'sequence': 0.30,
            'coherence': 0.18
        },
        'gaps': [
            {
                'type': 'insufficient_activities',
                'severity': 'medium',
                'message': '...',
                'suggestion': '...'
            }
        ]
    }
}
```

**Key methods**:
- `infer_mechanisms(nodes, text)` → Dict[str, Dict]
- `_infer_single_mechanism(node, text, all_nodes)` → Dict
- `_infer_mechanism_type(observations)` → Dict[str, float]
- `_quantify_uncertainty(...)` → Dict[str, float]

### AGUJA III: Counterfactual Audit

**Class**: `OperationalizationAuditor`  
**Method**: `bayesian_counterfactual_audit(nodes, graph, historical_data=None)`

**What it returns**:
```python
{
    'direct_evidence': {
        'MP-001': {
            'omissions': ['baseline', 'entity'],
            'omission_probabilities': {
                'baseline': 0.11,
                'entity': 0.06
            },
            'omission_severity': 'critical'
        }
    },
    'causal_implications': {
        'MP-001': {
            'causal_effects': {
                'target_miscalibration': {
                    'probability': 0.73,
                    'description': '...'
                }
            },
            'total_risk': 0.65
        }
    },
    'systemic_risk': {
        'risk_score': 0.42,
        'success_probability': 0.68,
        'critical_omissions': [...]
    },
    'recommendations': [
        {
            'node_id': 'MP-001',
            'omission': 'baseline',
            'severity': 'critical',
            'expected_value': 0.85,
            'effort': 3,
            'priority': 0.28,
            'recommendation': 'Definir línea base...'
        }
    ]
}
```

**Key methods**:
- `bayesian_counterfactual_audit(nodes, graph, ...)` → Dict
- `_build_normative_dag()` → nx.DiGraph
- `_audit_direct_evidence(...)` → Dict
- `_audit_causal_implications(...)` → Dict
- `_audit_systemic_risk(...)` → Dict
- `_generate_optimal_remediations(...)` → List[Dict]

## Accessing Results

### From Graph (after `extract_causal_hierarchy`)
```python
causal_extractor = CausalExtractor(config, nlp)
graph = causal_extractor.extract_causal_hierarchy(text)

# Access Bayesian link data
for source, target in graph.edges():
    edge_data = graph.edges[source, target]
    print(f"{source} → {target}:")
    print(f"  Posterior: {edge_data['posterior_mean']:.3f} ± {edge_data['posterior_std']:.3f}")
    print(f"  Converged: {edge_data['converged']}")
    print(f"  Evidence: {edge_data['evidence_count']} pieces")
```

### From Mechanism Inference
```python
bayesian_mech = BayesianMechanismInference(config, nlp)
mechanisms = bayesian_mech.infer_mechanisms(nodes, text)

for node_id, mech in mechanisms.items():
    print(f"{node_id}:")
    print(f"  Most likely type: {max(mech['mechanism_type'].items(), key=lambda x: x[1])}")
    print(f"  Uncertainty: {mech['uncertainty']['total']:.2f}")
    print(f"  Sufficiency: {mech['sufficiency_test']['score']:.2f}")
```

### From Counterfactual Audit
```python
op_auditor = OperationalizationAuditor(config)
audit = op_auditor.bayesian_counterfactual_audit(nodes, graph)

# Top recommendations
for rec in audit['recommendations'][:5]:
    print(f"Priority {rec['priority']:.2f}: Fix {rec['omission']} in {rec['node_id']}")

# System-wide metrics
print(f"Success probability: {audit['systemic_risk']['success_probability']:.1%}")
print(f"Risk score: {audit['systemic_risk']['risk_score']:.2f}")
```

## Configuration

No special configuration needed. The Bayesian inference uses sensible defaults:

- **KL convergence threshold**: 0.01
- **Prior strength**: 4.0 (equivalent sample size)
- **Historical success rates**: Built-in defaults (can be overridden)
- **Evidence weights**: Tuned based on causal theory

To override historical priors:
```python
custom_priors = {
    'entity_presence_success_rate': 0.95,
    'baseline_presence_success_rate': 0.90,
    # ... etc
}

audit = op_auditor.bayesian_counterfactual_audit(
    nodes, graph, 
    historical_data=custom_priors
)
```

## Interpreting Results

### Posterior Distributions
- **Mean**: Best estimate of causal strength
- **Std**: Uncertainty (higher = less confident)
- **Converged**: True if belief stabilized (KL < 0.01)

### Uncertainty Scores
- **0.0 - 0.3**: Low uncertainty (confident)
- **0.3 - 0.6**: Moderate uncertainty
- **0.6 - 1.0**: High uncertainty (more evidence needed)

### Priority Scores
- Calculated as: Expected Value / Effort
- Higher priority = more impact per unit effort
- Fix highest priority items first

### Success Probability
- Estimated P(plan_succeeds | current_state)
- Based on: Completeness × Base success rate
- < 60%: Critical issues present
- 60-80%: Acceptable with improvements
- > 80%: Well-documented

## Common Patterns

### Pattern 1: Find high-uncertainty mechanisms
```python
mechanisms = bayesian_mech.infer_mechanisms(nodes, text)
high_uncertainty = {
    nid: mech for nid, mech in mechanisms.items()
    if mech['uncertainty']['total'] > 0.6
}
```

### Pattern 2: Identify critical omissions
```python
audit = op_auditor.bayesian_counterfactual_audit(nodes, graph)
critical = [
    rec for rec in audit['recommendations']
    if rec['severity'] == 'critical'
]
```

### Pattern 3: Compare causal link confidence
```python
links = [(s, t, graph.edges[s, t]) for s, t in graph.edges()]
sorted_by_confidence = sorted(
    links, 
    key=lambda x: x[2]['posterior_std']  # Lower std = higher confidence
)
```

## Troubleshooting

**Q: Why are some posterior_stds very high?**  
A: Insufficient evidence. Check `evidence_count` - should be ≥ 3 for stability.

**Q: Why didn't a link converge?**  
A: Conflicting evidence. Inspect individual evidence components to diagnose.

**Q: Why is mechanism uncertainty high?**  
A: Check `gaps` - likely missing entity, activities, or budget information.

**Q: Why is success_probability low?**  
A: Many critical omissions. Check `recommendations` for top fixes.

## Performance Notes

- Bayesian updates are fast: O(evidence_count)
- Semantic distance requires spaCy embeddings (pre-loaded)
- No MCMC sampling = no performance hit
- Typical overhead: ~10-15% vs deterministic version

## Further Reading

- `BAYESIAN_INFERENCE_IMPLEMENTATION.md` - Full technical details
- `demo_bayesian_agujas.py` - Interactive demonstration
- `IMPLEMENTATION_SUMMARY.md` - High-level overview

---

*Quick reference for FARFAN 2.0 Bayesian inference capabilities*
