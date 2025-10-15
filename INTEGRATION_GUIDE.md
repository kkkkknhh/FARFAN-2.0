# Integration Guide: Axiomatic Validator

## Overview

This guide explains how to integrate the **Axiomatic Validator** into existing PDM evaluation workflows.

## Integration Points

### 1. Integration with Existing Evaluators

The Axiomatic Validator is designed to work alongside existing evaluation code:

```python
# Traditional approach (before F1.3)
from teoria_cambio import TeoriaCambio
from contradiction_deteccion import PolicyContradictionDetectorV2
from dnp_integration import ValidadorDNP

teoria = TeoriaCambio()
detector = PolicyContradictionDetectorV2()
dnp = ValidadorDNP()

# Manual validation with 3 separate calls
resultado_teoria = teoria.validacion_completa(grafo)
resultado_contradicciones = detector.detect(texto, plan_name="PDM")
resultado_dnp = dnp.validar_proyecto_integral(sector, descripcion, indicadores)

# Manual aggregation of results
# ...complex logic to combine results...
```

```python
# Unified approach (F1.3 - Axiomatic Validator)
from validators import AxiomaticValidator, ValidationConfig, PDMOntology, SemanticChunk

config = ValidationConfig()
ontology = PDMOntology()
validator = AxiomaticValidator(config, ontology)

# Single unified validation call
result = validator.validate_complete(grafo, semantic_chunks, financial_data)

# Results already aggregated
if result.is_valid:
    print("All validations passed!")
else:
    print(f"Validation failed: {result.hold_reason}")
```

### 2. Integration with Canonical Notation System

The validator maps directly to canonical notation:

```python
# Canonical notation reference
from canonical_notation import CanonicalID, RubricKey

# D6-Q2: Theory of Change - Causal Order
if result.violaciones_orden:
    canonical_id = CanonicalID.from_string("P1-D6-Q2")
    # Flag for evaluation
    
# D2-Q5: Semantic Coherence
if result.contradiction_density > 0.05:
    canonical_id = CanonicalID.from_string("P1-D2-Q5")
    # Flag for evaluation
    
# D1-Q5: Regulatory Compliance
if result.regulatory_score < 60:
    canonical_id = CanonicalID.from_string("P1-D1-Q5")
    # Flag for evaluation
```

### 3. Integration with Report Generator

```python
from report_generator import ReportGenerator
from validators import AxiomaticValidator

# Run validation
validator = AxiomaticValidator(config, ontology)
result = validator.validate_complete(grafo, chunks)

# Generate report
report_gen = ReportGenerator()
summary = result.get_summary()

# Add validation section to report
report_gen.add_section("Validación Axiomática", {
    'estado': 'VÁLIDO' if result.is_valid else 'INVÁLIDO',
    'validacion_estructural': result.structural_valid,
    'densidad_contradicciones': result.contradiction_density,
    'puntaje_regulatorio': result.regulatory_score,
    'requiere_revision_manual': result.requires_manual_review,
    'razon_retencion': result.hold_reason,
    'fallas_criticas': len([f for f in result.failures if f.severity.value == 'CRITICAL'])
})
```

### 4. Integration with Policy Processor

```python
from policy_processor import PolicyProcessor
from validators import AxiomaticValidator, SemanticChunk

# Process PDM document
processor = PolicyProcessor()
doc = processor.load_pdm("path/to/pdm.pdf")

# Extract semantic chunks
raw_chunks = processor.extract_semantic_chunks(doc)
semantic_chunks = [
    SemanicChunk(
        text=chunk['text'],
        dimension=chunk['dimension'],
        position=chunk['position']
    )
    for chunk in raw_chunks
]

# Extract causal graph
grafo = processor.build_causal_graph(doc)

# Validate
validator = AxiomaticValidator(config, ontology)
result = validator.validate_complete(grafo, semantic_chunks)

# Store validation results
processor.store_validation_results(result)
```

## Workflow Integration Patterns

### Pattern 1: Pre-evaluation Validation Gate

Use the validator as a quality gate before detailed evaluation:

```python
def evaluate_pdm_with_gate(pdm_document):
    """Evaluate PDM with quality gate"""
    
    # Extract data
    grafo = extract_causal_graph(pdm_document)
    chunks = extract_semantic_chunks(pdm_document)
    
    # Validate
    validator = AxiomaticValidator(config, ontology)
    result = validator.validate_complete(grafo, chunks)
    
    # Quality gate
    if not result.is_valid:
        return {
            'status': 'REJECTED',
            'reason': result.hold_reason,
            'failures': result.failures
        }
    
    if result.requires_manual_review:
        return {
            'status': 'PENDING_REVIEW',
            'reason': result.hold_reason,
            'recommendation': 'Manual review required before detailed evaluation'
        }
    
    # Proceed with detailed evaluation
    return perform_detailed_evaluation(pdm_document, result)
```

### Pattern 2: Parallel Validation with Fallback

Run validator in parallel with existing code:

```python
def dual_validation_approach(grafo, chunks):
    """Run both old and new validation for transition period"""
    
    # New unified approach
    validator = AxiomaticValidator(config, ontology)
    new_result = validator.validate_complete(grafo, chunks)
    
    # Old separate approach (for comparison)
    old_result = {
        'teoria': TeoriaCambio().validacion_completa(grafo),
        'contradicciones': PolicyContradictionDetectorV2().detect(text),
        'dnp': ValidadorDNP().validar_proyecto_integral(...)
    }
    
    # Compare and log differences
    if new_result.is_valid != old_result['teoria'].es_valida:
        logger.warning("Validation discrepancy detected")
    
    # Return new result (but keep old for audit)
    return new_result
```

### Pattern 3: Incremental Validation

Use validator for incremental document updates:

```python
class IncrementalValidator:
    """Validator that caches results for efficiency"""
    
    def __init__(self):
        self.validator = AxiomaticValidator(config, ontology)
        self.last_result = None
        self.graph_hash = None
    
    def validate_incremental(self, grafo, chunks, force=False):
        """Only re-validate if graph changed"""
        
        # Compute hash of current graph
        import hashlib
        current_hash = hashlib.md5(
            str(grafo.edges()).encode()
        ).hexdigest()
        
        # Check if validation needed
        if not force and current_hash == self.graph_hash:
            return self.last_result
        
        # Validate
        result = self.validator.validate_complete(grafo, chunks)
        
        # Cache
        self.graph_hash = current_hash
        self.last_result = result
        
        return result
```

## API Integration

### REST API Example

```python
from flask import Flask, request, jsonify
from validators import AxiomaticValidator, ValidationConfig, PDMOntology

app = Flask(__name__)
validator = AxiomaticValidator(ValidationConfig(), PDMOntology())

@app.route('/api/validate', methods=['POST'])
def validate_pdm():
    """API endpoint for PDM validation"""
    
    data = request.json
    
    # Parse input
    grafo = build_graph_from_json(data['graph'])
    chunks = [SemanticChunk(**chunk) for chunk in data['chunks']]
    
    # Validate
    result = validator.validate_complete(grafo, chunks)
    
    # Return response
    return jsonify({
        'valid': result.is_valid,
        'summary': result.get_summary(),
        'failures': [
            {
                'dimension': f.dimension,
                'question': f.question,
                'severity': f.severity.value,
                'impact': f.impact
            }
            for f in result.failures
        ]
    })
```

## Migration Path

### Phase 1: Parallel Operation (Weeks 1-2)
- Deploy Axiomatic Validator alongside existing validators
- Run both systems in parallel
- Compare results and log discrepancies
- Fix any issues found

### Phase 2: Gradual Transition (Weeks 3-4)
- Use Axiomatic Validator as primary
- Keep old validators as fallback
- Monitor production metrics
- Update documentation

### Phase 3: Full Migration (Week 5+)
- Make Axiomatic Validator the sole validator
- Remove redundant code
- Archive old validation logic
- Update all dependent systems

## Configuration Management

### Environment-based Configuration

```python
import os
from validators import ValidationConfig

def get_validator_config():
    """Get configuration based on environment"""
    
    env = os.environ.get('ENV', 'development')
    
    if env == 'production':
        return ValidationConfig(
            dnp_lexicon_version="2025",
            contradiction_threshold=0.05,
            enable_structural_penalty=True,
            enable_human_gating=True
        )
    elif env == 'staging':
        return ValidationConfig(
            dnp_lexicon_version="2025",
            contradiction_threshold=0.07,
            enable_structural_penalty=True,
            enable_human_gating=False  # Auto-approve in staging
        )
    else:  # development
        return ValidationConfig(
            dnp_lexicon_version="2025",
            contradiction_threshold=0.10,
            enable_structural_penalty=False,
            enable_human_gating=False
        )
```

## Monitoring and Metrics

### Key Metrics to Track

```python
def track_validation_metrics(result):
    """Track important validation metrics"""
    
    metrics = {
        'timestamp': result.validation_timestamp,
        'is_valid': result.is_valid,
        'structural_valid': result.structural_valid,
        'contradiction_density': result.contradiction_density,
        'regulatory_score': result.regulatory_score,
        'requires_manual_review': result.requires_manual_review,
        'hold_reason': result.hold_reason,
        'critical_failures': len([f for f in result.failures if f.severity.value == 'CRITICAL']),
        'total_nodes': result.total_nodes,
        'total_edges': result.total_edges
    }
    
    # Send to monitoring system
    # monitoring.send_metrics('validator', metrics)
    
    return metrics
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check Python version: 3.10+ required

2. **Validation Failures**
   - Check `result.failures` for detailed error information
   - Review logs for warnings
   - Verify input data format

3. **Performance Issues**
   - Use incremental validation pattern
   - Consider caching results
   - Profile with larger datasets

## Support

For issues or questions:
- Check the `validators/README.md` for detailed documentation
- Review test files for usage examples
- Run `python3 example_axiomatic_validator.py` for demonstrations
