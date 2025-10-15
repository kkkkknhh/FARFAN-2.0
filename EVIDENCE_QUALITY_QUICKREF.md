# Evidence Quality Auditors - Quick Reference

## Quick Start

```python
from evidence_quality_auditors import run_all_audits, IndicatorMetadata

# Minimal example
text = "Proyecto BPIN 2024000111111 con déficit de 35% alineado con PND y ODS-4"
results = run_all_audits(text=text)

for audit_type, result in results.items():
    print(f"{audit_type}: {result.severity.value} - SOTA: {result.sota_compliance}")
```

## Four Auditors at a Glance

| Auditor | Audit Point | What It Checks | Threshold | SOTA Reference |
|---------|-------------|----------------|-----------|----------------|
| **OperationalizationAuditor** | D3-Q1 | Indicator metadata completeness | ≥80% complete | UN 2020, Gelman 2013 |
| **FinancialTraceabilityAuditor** | D1-Q3, D3-Q3 | BPIN/PPI code traceability | ≥95% confidence | DNP 2023, Waldner 2015 |
| **QuantifiedGapAuditor** | D1-Q2 | Quantified gaps/deficits | ≥70% quantified | Ragin 2008, MMR |
| **SystemicRiskAuditor** | D4-Q5, D5-Q4 | PND/ODS alignment | <0.10 risk score | Pearl 2018 |

## Common Patterns

### Check Indicator Completeness
```python
from evidence_quality_auditors import OperationalizationAuditor, IndicatorMetadata

auditor = OperationalizationAuditor()
indicators = [
    IndicatorMetadata("IND-001", "Nombre", 75.0, 90.0, "DANE", "(A/B)*100")
]
result = auditor.audit_indicators(indicators)
# result.metrics['completeness_ratio'] >= 0.80 → EXCELLENT
```

### Find BPIN/PPI Codes
```python
from evidence_quality_auditors import FinancialTraceabilityAuditor

auditor = FinancialTraceabilityAuditor()
result = auditor.audit_financial_codes(pdm_text)
# Detects: BPIN 2024000111111, PPI-2024001234
```

### Detect Gaps
```python
from evidence_quality_auditors import QuantifiedGapAuditor

auditor = QuantifiedGapAuditor()
result = auditor.audit_quantified_gaps(pdm_text)
# Detects: "déficit de 35%", "brecha de 40%", "vacíos de información"
```

### Check Alignment
```python
from evidence_quality_auditors import SystemicRiskAuditor

auditor = SystemicRiskAuditor()
result = auditor.audit_systemic_risk(pdm_text)
# Detects: PND, ODS-4, ODS-10, calculates risk_score
```

## Severity Levels

```
EXCELLENT        ✓ SOTA Compliant - Exceeds standards
GOOD            ✓ SOTA Compliant - Meets standards  
ACCEPTABLE      ✗ Below SOTA - Needs improvement
REQUIRES_REVIEW ✗ Critical - Urgent action needed
CRITICAL        ✗ Fatal - Missing essential elements
```

## Output Structure

Every audit returns:
```python
result = AuditResult(
    audit_type="D3-Q1_IndicatorMetadata",
    timestamp="2025-10-15T19:00:00",
    severity=AuditSeverity.EXCELLENT,
    findings=[...],              # Detailed findings
    metrics={...},               # Quantitative metrics
    recommendations=[...],       # Actionable advice
    evidence={...},              # SOTA references
    sota_compliance=True         # Boolean flag
)
```

## Integration with Orchestrator

```python
# In orchestrator.py - automatically integrated
orchestrator = AnalyticalOrchestrator()
report = orchestrator.orchestrate_analysis(text, "PDM", "estratégico")

# Evidence quality audits in:
# report['analyze_regulatory_constraints']['outputs']['d1_q5_regulatory_analysis']
```

## Key Metrics

### OperationalizationAuditor
- `completeness_ratio`: 0.0-1.0 (target: ≥0.80)
- `complete_indicators`: Count
- `incomplete_indicators`: Count

### FinancialTraceabilityAuditor
- `total_codes`: BPIN + PPI count
- `high_confidence_ratio`: 0.0-1.0 (target: ≥0.95)
- `low_confidence_codes`: Count

### QuantifiedGapAuditor
- `quantification_ratio`: 0.0-1.0 (target: ≥0.70)
- `subregistro_count`: Count (important for MMR)
- `gap_type_distribution`: Dict by type

### SystemicRiskAuditor
- `risk_score`: 0.0-1.0 (target: <0.10)
- `ods_count`: Number of ODS aligned
- `pnd_alignment`: Boolean

## Testing Quick Check

```bash
# Run all 29 tests (should take <1 second)
python -m unittest test_evidence_quality_auditors -v

# Expected output:
# Ran 29 tests in 0.004s
# OK
```

## Common Issues

### No codes found?
- Check BPIN format: 10-13 digits with optional "BPIN" prefix
- Check PPI format: 6+ digits with "PPI" prefix
- Example: `BPIN 2024000111111` or `PPI-2024001234`

### Low quantification ratio?
- Add numeric values to gaps: "déficit de 35%", "brecha de 1200 cupos"
- Avoid vague statements: "alto déficit" → "déficit de 40%"

### High risk score?
- Add PND reference: "Plan Nacional de Desarrollo"
- Add ODS alignment: "ODS-4", "ODS-10"
- Target: 3+ ODS for best results

### Low indicator completeness?
- Ensure all 4 required fields: linea_base, meta, fuente, formula
- Example: `IndicatorMetadata("ID", "Name", 75.0, 90.0, "DANE", "(A/B)*100")`

## Performance Tips

1. **Pre-extract indicators** from PDM tables for better accuracy
2. **Combine audits** using `run_all_audits()` for efficiency
3. **Cache results** - auditors are deterministic
4. **Customize thresholds** based on municipality context

## SOTA References Quick Lookup

- **UN 2020**: ODS alignment benchmarks → Operationalization
- **Gelman 2013**: Bayesian updating → Operationalization  
- **DNP 2023**: Colombian audit standards → Financial Traceability
- **Waldner 2015**: Fiscal illusions → Financial Traceability
- **Ragin 2008**: QCA calibration → Quantified Gaps
- **Pearl 2018**: Counterfactual rigor → Systemic Risk

## Next Steps

1. See [EVIDENCE_QUALITY_AUDITORS_README.md](EVIDENCE_QUALITY_AUDITORS_README.md) for full documentation
2. Review [evidence_quality_auditors.py](evidence_quality_auditors.py) for implementation
3. Check [test_evidence_quality_auditors.py](test_evidence_quality_auditors.py) for examples
4. Integrate with your PDM analysis pipeline

---

**Need help?** Check the main documentation or run the example:
```bash
python evidence_quality_auditors.py
```
