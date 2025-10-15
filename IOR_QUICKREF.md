# IoR Adaptive Learning Loop - Quick Reference

## Overview
Enable self-reflective capacity for D6-Q4, reducing uncertainty via iterative Bayesian prior updating.

---

## Audit Points at a Glance

| Point | Feature | Criterion | Status |
|-------|---------|-----------|--------|
| 5.1 | Prior Learning | Tracks failed mechanisms; reduces priors | ✅ |
| 5.2 | Prior Decay | >20% decay over 10 sequential analyses | ✅ |
| 5.3 | Immutable Audit | Append-only store with hash verification | ✅ |

---

## Quick Start

### 1. Enable Adaptive Learning (config.yaml)

```yaml
self_reflection:
  enable_prior_learning: true
  feedback_weight: 0.1
  prior_history_path: "logs/prior_history.json"
  min_documents_for_learning: 5
```

### 2. Run Analysis with Learning

```python
from dereck_beach import CDAFFramework

framework = CDAFFramework(
    config_path="config.yaml",
    output_dir="results/",
    log_level="INFO"
)

# Process document - learning happens automatically
success = framework.process_document(
    pdf_path="pdm_2024.pdf",
    policy_code="PDM_2024"
)
```

### 3. Append to Immutable Audit Store

```python
from orchestrator import create_orchestrator

orchestrator = create_orchestrator()

# After analysis
metadata = orchestrator.append_audit_record(
    run_id="pdm_2024_q1_001",
    analysis_results=results,
    source_text=original_text
)

# Verify later
verification = orchestrator.verify_audit_record(
    Path(metadata['record_path'])
)
assert verification['verified'], "Tampering detected!"
```

---

## Key Methods

### ConfigLoader.update_priors_from_feedback()

**Purpose:** Update mechanism type priors based on audit feedback  
**Input:** `feedback_data` with penalty_factors, test_failures  
**Output:** Updated mechanism_type_priors with >20% decay for failures

```python
feedback_data = {
    'penalty_factors': {'politico': 0.85},
    'test_failures': {
        'necessity_failures': 2,
        'sufficiency_failures': 3
    }
}
config.update_priors_from_feedback(feedback_data)
```

### Orchestrator.append_audit_record()

**Purpose:** Store analysis results in immutable append-only store  
**Input:** run_id, analysis_results, source_text  
**Output:** Record metadata with hash, retention date

```python
metadata = orchestrator.append_audit_record(
    run_id="unique_run_id",
    analysis_results={"plan_name": "PDM", ...},
    source_text="Policy document text..."
)
# Returns: {run_id, timestamp, sha256_source, record_hash, retention_until}
```

### Orchestrator.verify_audit_record()

**Purpose:** Verify record hasn't been tampered with  
**Input:** Path to audit record JSON file  
**Output:** Verification result with hash comparison

```python
verification = orchestrator.verify_audit_record(record_path)
if verification['verified']:
    print("✓ Record is immutable")
else:
    print("✗ Tampering detected!")
```

---

## Audit Record Structure

```json
{
  "run_id": "pdm_2024_q1_001",
  "timestamp": "2025-10-15T18:44:07.438366",
  "sha256_source": "258764975e2cb1fa...",
  "retention_until": "2032-10-13T18:44:07.438366",
  "analysis_results": {...},
  "calibration": {...},
  "framework_version": "2.0.0",
  "record_hash": "197a69ee2b8f8605..."
}
```

**Mandatory Fields:**
- `run_id`: Unique identifier
- `timestamp`: ISO format timestamp
- `sha256_source`: Hash of original source text
- `record_hash`: SHA-256 of entire record
- `retention_until`: 7-year retention date

---

## Testing

```bash
# Run all adaptive learning tests
python3 test_adaptive_learning_loop.py

# Expected output:
# ✓ Audit Point 5.3: Immutable Audit Governance - PASSED
# ✓ Audit Point 5.1: Prior Learning Functionality - PASSED
# ✓ Audit Point 5.2: Measurable Prior Decay - PASSED
```

---

## Prior Decay Example

```
Initial prior (politico): 0.150
After 10 failures:       0.119
Decay rate:              20.35% ✅
```

**Decay Formula:**
```python
penalty_factor = 0.95 - (failure_freq * 0.25)  # 0.70-0.95
penalty_weight = feedback_weight * 1.5
new_prior = (1 - penalty_weight) * old_prior + penalty_weight * (old_prior * penalty_factor)
```

---

## Uncertainty Reduction Tracking

```python
uncertainty_check = config.check_uncertainty_reduction_criterion(
    current_uncertainty=0.65
)

# Success if reduction ≥5% over 10 iterations
if uncertainty_check['criterion_met']:
    print(f"✓ Uncertainty reduced by {uncertainty_check['reduction_percent']:.2f}%")
```

---

## SOTA Compliance Checklist

- [x] Feedback loops align with iterative QCA (Ragin 2014)
- [x] Epistemic adaptation automated (Humphreys 2015)
- [x] Prior decay >20% for failing types (Beach 2019)
- [x] Non-repudiable audit governance (Ragin 2014)
- [x] 7-year retention for compliance
- [x] SHA-256 hash verification

---

## Common Parameters

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| feedback_weight | 0.1 | 0.0-1.0 | Weight for prior updates |
| penalty_factor | 0.70-0.95 | Dynamic | Penalty for failures |
| retention_years | 7 | Fixed | Audit retention period |
| min_documents | 5 | ≥1 | Min before learning applies |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Decay <20% | Increase `feedback_weight` to 0.15 |
| Hash fails | Check `sort_keys=True` in JSON dump |
| Learning disabled | Set `enable_prior_learning: true` |
| No updates | Check `min_documents_for_learning` |

---

## File Locations

```
logs/
├── orchestrator/
│   ├── audit_log_PDM_20251015_184407.json       # Regular logs
│   └── audit_store/
│       └── audit_pdm_2024_q1_001_*.json         # Immutable records
└── prior_history.json                           # Prior evolution tracking
```

---

## Performance Metrics

**Target:** Mean mechanism_type uncertainty decreases by ≥5% over 10 sequential analyses

**Measurement:**
```python
initial_entropy = -sum(p * log(p) for p in priors)
final_entropy = -sum(p * log(p) for p in updated_priors)
reduction = (initial_entropy - final_entropy) / initial_entropy * 100
```

---

## Integration Points

### 1. CDAFFramework (dereck_beach)
Auto-updates priors after each analysis if `enable_prior_learning=true`

### 2. Orchestrator (orchestrator.py)
Provides immutable audit storage for all analysis runs

### 3. BayesianMechanismInference
Tracks mechanism uncertainty for feedback extraction

---

## Next Steps

1. ✅ Enable adaptive learning in config
2. ✅ Run multiple analyses to accumulate history
3. ✅ Monitor prior evolution in `prior_history.json`
4. ✅ Verify audit records remain immutable
5. ✅ Check uncertainty reduction criterion (≥5%)

---

## References

- **Audit Point 5.1**: ConfigLoader.update_priors_from_feedback (dereck_beach)
- **Audit Point 5.2**: Prior decay mechanism + test_mechanism_prior_decay
- **Audit Point 5.3**: Orchestrator.append_audit_record + verify_audit_record

**Full Documentation:** See `IOR_ADAPTIVE_LEARNING_LOOP_GUIDE.md`

---

**Version:** FARFAN 2.0  
**Last Updated:** 2025-10-15  
**Status:** ✅ All Audit Points Implemented and Tested
