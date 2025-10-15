# Evidence Quality and Compliance Auditors - Documentation

## Overview

This module implements **Part 3: Evidence Quality and Compliance (Analytical D1, D3, D4, D5 Audit)** as specified in the FARFAN 2.0 framework. It provides four specialized auditors that verify auditable facts against SOTA (State of the Art) compliance standards per DNP frameworks integrated with MMR (Mixed Methods Research).

## Architecture

The module follows the FARFAN 2.0 orchestration principles:

- **Sequential Phase Execution**: Each auditor operates independently but can be orchestrated together
- **Deterministic Behavior**: Same input produces identical outputs
- **Calibration Constants**: Configurable thresholds with sensible defaults
- **Complete Audit Trail**: All findings are logged with evidence
- **Structured Data Contracts**: Standardized `AuditResult` output format

## Four Auditors

### 1. OperationalizationAuditor (D3-Q1)

**Purpose**: Audits indicator ficha técnica completeness

**Check Criteria**:
- Requires full metadata (Línea Base, Meta, Fuente) + formula for ≥80% of product indicators
- Default threshold: 0.80 (configurable)

**Quality Evidence**:
- Cross-checks formulas against PDM tables
- Validates completeness of required fields

**SOTA Performance**:
- Measurability exceeds ODS alignment benchmarks (UN 2020)
- Full metadata enables Bayesian updating (Gelman 2013)

**Usage**:
```python
from evidence_quality_auditors import OperationalizationAuditor, IndicatorMetadata

# Create auditor
auditor = OperationalizationAuditor(metadata_threshold=0.80)

# Define indicators
indicators = [
    IndicatorMetadata(
        codigo="EDU-001",
        nombre="Tasa de cobertura educativa",
        linea_base=75.0,
        meta=90.0,
        fuente="DANE",
        formula="(Matriculados/Población)*100"
    )
]

# Run audit
result = auditor.audit_indicators(indicators, pdm_tables=None)

print(f"Severity: {result.severity.value}")
print(f"SOTA Compliance: {result.sota_compliance}")
print(f"Completeness: {result.metrics['completeness_ratio']:.2%}")
```

**Output Metrics**:
- `total_indicators`: Total number of indicators audited
- `complete_indicators`: Indicators with full metadata
- `incomplete_indicators`: Indicators missing metadata
- `completeness_ratio`: Proportion of complete indicators
- `meets_threshold`: Boolean indicating if threshold is met

**Severity Levels**:
- **Excellent** (≥80%): Full SOTA compliance
- **Good** (70-80%): SOTA compliance
- **Acceptable** (60-70%): Below SOTA
- **Requires Review** (<60%): Critical deficiency

---

### 2. FinancialTraceabilityAuditor (D1-Q3, D3-Q3)

**Purpose**: Audits financial traceability to BPIN/PPI codes

**Check Criteria**:
- Traces to BPIN/PPI codes + dependency
- Penalizes if match confidence <0.95
- Default confidence threshold: 0.95 (configurable)

**Quality Evidence**:
- Reviews fuzzy match incidents
- Verifies codes in PDM vs. logs

**SOTA Performance**:
- High-confidence matching per audit standards (Colombian DNP 2023)
- Reduces fiscal illusions in causal chains (Waldner 2015)

**Usage**:
```python
from evidence_quality_auditors import FinancialTraceabilityAuditor

# Create auditor
auditor = FinancialTraceabilityAuditor(confidence_threshold=0.95)

# Audit PDM text
text = """
Proyecto BPIN 2023000123456 busca mejorar infraestructura.
Plan Plurianual PPI-2023001234 establece prioridades.
"""

result = auditor.audit_financial_codes(text, pdm_tables=None)

print(f"Total codes found: {result.metrics['total_codes']}")
print(f"BPIN codes: {result.metrics['bpin_codes']}")
print(f"PPI codes: {result.metrics['ppi_codes']}")
print(f"High confidence ratio: {result.metrics['high_confidence_ratio']:.2%}")
```

**Code Detection Patterns**:
- **BPIN**: 10-13 digit codes with optional "BPIN" prefix
  - Examples: `BPIN 2023000123456`, `2023000123456`
- **PPI**: 6+ digit codes with "PPI" prefix
  - Examples: `PPI-2023001234`, `PPI 2023001234`

**Confidence Calculation**:
- Base confidence: 0.70
- +0.15 if code type mentioned in context
- +0.05 if investment keywords present (proyecto, inversión, presupuesto, código)

**Output Metrics**:
- `total_codes`: Total BPIN + PPI codes found
- `bpin_codes`: BPIN codes detected
- `ppi_codes`: PPI codes detected
- `high_confidence_codes`: Codes with confidence ≥ threshold
- `low_confidence_codes`: Codes below threshold
- `high_confidence_ratio`: Proportion of high-confidence matches

**Severity Levels**:
- **Critical** (0 codes): No traceability found
- **Excellent** (≥95% high-confidence): Full SOTA compliance
- **Good** (85-95%): SOTA compliance
- **Acceptable** (75-85%): Below SOTA
- **Requires Review** (<75%): Poor traceability

---

### 3. QuantifiedGapAuditor (D1-Q2)

**Purpose**: Audits quantified gap recognition

**Check Criteria**:
- Detects data limitations (vacíos)
- Detects quantified brechas (gap/deficit)
- Identifies subregistro (underreporting)

**Quality Evidence**:
- Pattern-matches structured quantitative claims to PDM text
- Extracts quantification values from context

**SOTA Performance**:
- Quantified baselines boost QCA calibration (Ragin 2008)
- Identifies subregistro for robust MMR

**Usage**:
```python
from evidence_quality_auditors import QuantifiedGapAuditor

# Create auditor
auditor = QuantifiedGapAuditor()

# Audit PDM text
text = """
El municipio presenta un déficit de 35% en cobertura educativa.
Brecha de 1200 cupos en educación preescolar.
Se identifican vacíos de información en zonas rurales.
Problema de subregistro en censos.
"""

result = auditor.audit_quantified_gaps(text, structured_claims=None)

print(f"Total gaps: {result.metrics['total_gaps']}")
print(f"Quantified gaps: {result.metrics['quantified_gaps']}")
print(f"Quantification ratio: {result.metrics['quantification_ratio']:.2%}")
print(f"Subregistro cases: {result.metrics['subregistro_count']}")
```

**Gap Types Detected**:
1. **Vacío**: Data voids or information gaps
   - Pattern: `vacío(s) de información/datos`
   - Example: "vacíos de información en zonas rurales"

2. **Brecha**: Gaps with quantification
   - Pattern: `brecha(s) de X% / X unidades`
   - Example: "brecha de 45% en acceso a salud"

3. **Déficit**: Deficits with quantification
   - Pattern: `déficit de X% / X unidades`
   - Example: "déficit de 35% en cobertura educativa"

4. **Subregistro**: Underreporting issues
   - Pattern: `sub-registro / subregistro`
   - Example: "problema de subregistro en censos"

**Output Metrics**:
- `total_gaps`: Total gaps detected
- `quantified_gaps`: Gaps with numeric quantification
- `unquantified_gaps`: Gaps without quantification
- `quantification_ratio`: Proportion of quantified gaps
- `subregistro_count`: Cases of underreporting
- `gap_type_distribution`: Count by gap type

**Severity Levels**:
- **Excellent** (≥70% quantified): Full SOTA compliance
- **Good** (50-70% quantified): SOTA compliance
- **Acceptable** (<50% quantified): Below SOTA
- **Requires Review** (0 gaps): Missing gap analysis

---

### 4. SystemicRiskAuditor (D4-Q5, D5-Q4)

**Purpose**: Audits systemic risk alignment with PND/ODS

**Check Criteria**:
- Integrates PND/ODS alignment
- risk_score <0.10 for Excellent
- Increases on misalignment
- Default excellent threshold: 0.10 (configurable)

**Quality Evidence**:
- Checks risk_score escalation in CounterfactualAuditor logs
- Detects PND and ODS references

**SOTA Performance**:
- Counterfactual rigor per Pearl (2018)
- Low risk aligns with macro-causal frameworks

**Usage**:
```python
from evidence_quality_auditors import SystemicRiskAuditor

# Create auditor
auditor = SystemicRiskAuditor(excellent_threshold=0.10)

# Audit PDM text
text = """
Proyecto alineado con Plan Nacional de Desarrollo (PND).
Contribuye a ODS-4 (Educación de Calidad) y ODS-10.
"""

result = auditor.audit_systemic_risk(
    text, 
    causal_graph=None, 
    counterfactual_audit=None
)

print(f"PND alignment: {result.metrics['pnd_alignment']}")
print(f"ODS count: {result.metrics['ods_count']}")
print(f"ODS numbers: {result.metrics['ods_numbers']}")
print(f"Risk score: {result.metrics['risk_score']:.3f}")
```

**Alignment Detection**:
- **PND (Plan Nacional de Desarrollo)**:
  - Patterns: `PND`, `Plan Nacional de Desarrollo`
  
- **ODS/SDG (Objetivos de Desarrollo Sostenible)**:
  - Patterns: `ODS-X`, `SDG-X` (where X = 1-17)
  - Example: "ODS-4", "SDG-10"

**Risk Score Calculation**:
- Base risk = 0.0
- +0.15 if no PND alignment
- +0.20 if no ODS alignment
- +0.10 if <3 ODS aligned
- +0.05 per counterfactual audit risk flag
- Capped at 1.0

**Output Metrics**:
- `pnd_alignment`: Boolean - PND mentioned
- `ods_count`: Number of unique ODS detected
- `ods_numbers`: List of ODS numbers (1-17)
- `risk_score`: Calculated risk [0.0, 1.0]
- `misalignment_count`: Number of misalignment reasons
- `meets_excellent_threshold`: Boolean

**Severity Levels**:
- **Excellent** (<0.10 risk): Full SOTA compliance
- **Good** (0.10-0.20 risk): SOTA compliance
- **Acceptable** (0.20-0.35 risk): Below SOTA
- **Requires Review** (>0.35 risk): High systemic risk

---

## Integrated Usage

Run all auditors together:

```python
from evidence_quality_auditors import (
    run_all_audits,
    IndicatorMetadata
)

# Prepare comprehensive PDM text
pdm_text = """
PLAN DE DESARROLLO MUNICIPAL 2024-2027

DIAGNÓSTICO:
El municipio presenta un déficit de 40% en cobertura educativa.
Se identifican vacíos de información en población rural.

ESTRATEGIA:
Proyecto BPIN 2024000111111 para inversión educativa.
Alineado con Plan Nacional de Desarrollo y ODS-4.

INDICADORES:
Tasa de cobertura: LB 60%, Meta 90%, Fuente: SED
Fórmula: (Matriculados/Población) * 100
"""

# Prepare indicators
indicators = [
    IndicatorMetadata(
        codigo="EDU-001",
        nombre="Tasa de cobertura educativa",
        linea_base=60.0,
        meta=90.0,
        fuente="SED",
        formula="(Matriculados/Población) * 100"
    )
]

# Run all audits
results = run_all_audits(
    text=pdm_text,
    indicators=indicators,
    pdm_tables=None,
    structured_claims=None,
    causal_graph=None,
    counterfactual_audit=None
)

# Process results
for audit_type, result in results.items():
    print(f"\n{audit_type.upper()}:")
    print(f"  Severity: {result.severity.value}")
    print(f"  SOTA Compliance: {result.sota_compliance}")
    print(f"  Findings: {len(result.findings)}")
    print(f"  Recommendations: {len(result.recommendations)}")
```

---

## Integration with Orchestrator

The auditors are integrated into `orchestrator.py` in the `_analyze_regulatory_constraints` phase:

```python
# In orchestrator.py - Phase 3
def _analyze_regulatory_constraints(self, statements, text, temporal_conflicts):
    from evidence_quality_auditors import run_all_audits
    
    # Run evidence quality audits
    audit_results = run_all_audits(
        text=text,
        indicators=None,  # Extracted in production
        pdm_tables=None,  # Extracted in production
        structured_claims=None,
        causal_graph=None,
        counterfactual_audit=None
    )
    
    # Aggregate results
    regulatory_analysis = {
        "evidence_quality_audits": {
            audit_type: {
                "severity": result.severity.value,
                "sota_compliance": result.sota_compliance,
                "metrics": result.metrics,
                "recommendation_count": len(result.recommendations)
            }
            for audit_type, result in audit_results.items()
        }
    }
    
    # Determine overall quality
    all_compliant = all(r.sota_compliance for r in audit_results.values())
    regulatory_analysis["d1_q5_quality"] = (
        "excelente" if all_compliant else "bueno" if any(...) else "insuficiente"
    )
```

---

## SOTA Compliance References

Each auditor includes references to state-of-the-art research and standards:

### OperationalizationAuditor
- **UN 2020**: ODS alignment benchmarks for measurability
- **Gelman 2013**: Bayesian updating methodology for adaptive indicators

### FinancialTraceabilityAuditor
- **Colombian DNP 2023**: Official audit standards for project codes
- **Waldner 2015**: Fiscal illusions in causal chains research

### QuantifiedGapAuditor
- **Ragin 2008**: QCA (Qualitative Comparative Analysis) calibration
- **MMR Framework**: Mixed Methods Research for robust analysis

### SystemicRiskAuditor
- **Pearl 2018**: Counterfactual rigor and causal inference
- **Macro-causal frameworks**: PND/ODS alignment reduces systemic risk

---

## Testing

Comprehensive test suite with 29 tests covering:

1. **OperationalizationAuditor Tests** (5 tests)
   - Complete indicators → excellent rating
   - Incomplete indicators → requires review
   - Threshold boundary conditions
   - Empty indicators handling

2. **FinancialTraceabilityAuditor Tests** (5 tests)
   - BPIN code detection
   - PPI code detection
   - Mixed codes
   - No codes → critical severity
   - Confidence calculation

3. **QuantifiedGapAuditor Tests** (7 tests)
   - Deficit detection
   - Brecha detection
   - Vacío detection
   - Subregistro detection
   - Quantification ratio
   - High quantification → excellent
   - No gaps → requires review

4. **SystemicRiskAuditor Tests** (8 tests)
   - PND detection
   - ODS detection
   - SDG alternative format
   - No alignment → high risk
   - Full alignment → excellent
   - Risk score calculation
   - Misalignment reasons

5. **Integration Tests** (2 tests)
   - Run all audits together
   - Comprehensive PDM analysis

6. **SOTA Compliance Tests** (4 tests)
   - Verify all SOTA references present

### Running Tests

```bash
# Run all tests
python -m unittest test_evidence_quality_auditors -v

# Run specific test class
python -m unittest test_evidence_quality_auditors.TestOperationalizationAuditor -v

# Run specific test
python -m unittest test_evidence_quality_auditors.TestOperationalizationAuditor.test_complete_indicators_excellent -v
```

All tests pass with 100% success rate:
```
Ran 29 tests in 0.004s
OK
```

---

## Output Format

All auditors return an `AuditResult` with:

```python
@dataclass
class AuditResult:
    audit_type: str              # e.g., "D3-Q1_IndicatorMetadata"
    timestamp: str               # ISO 8601 format
    severity: AuditSeverity      # EXCELLENT, GOOD, ACCEPTABLE, REQUIRES_REVIEW, CRITICAL
    findings: List[Dict]         # Detailed findings
    metrics: Dict[str, Any]      # Quantitative metrics
    recommendations: List[str]   # Actionable recommendations
    evidence: Dict[str, Any]     # Supporting evidence
    sota_compliance: bool        # SOTA compliance flag
```

---

## Performance Characteristics

- **Deterministic**: Same input always produces same output
- **Fast**: Pattern matching and regex-based detection
- **Lightweight**: No external API calls or heavy dependencies
- **Scalable**: Can process large PDM documents efficiently
- **Configurable**: All thresholds can be customized

---

## Future Enhancements

Potential improvements for production use:

1. **Machine Learning Integration**:
   - Train classifiers for code detection
   - Improve confidence scoring with ML models

2. **Database Integration**:
   - Verify BPIN/PPI codes against official DNP database
   - Cross-reference ODS alignment with national plans

3. **Advanced NLP**:
   - Entity extraction for indicators
   - Relation extraction for gap analysis

4. **Real-time Monitoring**:
   - Dashboard for audit results
   - Alert system for critical findings

5. **Multi-language Support**:
   - Support for indigenous languages
   - Regional dialect handling

---

## License

This module is part of the FARFAN 2.0 framework for municipal development plan analysis.

## Contact

For questions or contributions, please refer to the main FARFAN 2.0 repository.
