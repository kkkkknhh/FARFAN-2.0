# Quick Start: Enhanced CDAF Configuration

## What's New?

The CDAF framework has been enhanced with professional-grade features:

1. **Externalized Configuration** - All hardcoded values now in YAML
2. **Schema Validation** - Pydantic validates config at load time
3. **Custom Exceptions** - Better error messages with structured data
4. **Performance Tuning** - Configurable speed/accuracy tradeoffs
5. **Self-Learning** - Optional feedback loops to improve over time

## Installation

```bash
# Install the new dependency
pip install pydantic

# Verify
python -c "import pydantic; print('✓ Pydantic installed')"
```

## Quick Start

### 1. Use the Enhanced Configuration

```bash
# Copy the example config
cp config_example_enhanced.yaml my_config.yaml

# Edit if needed (optional - defaults work fine)
nano my_config.yaml
```

### 2. Run Your Analysis

```python
#!/usr/bin/env python3
from pathlib import Path

# Import the framework (same as before)
exec(open('dereck_beach').read())

# Initialize with your config
framework = CDAFFramework(
    config_path=Path('./my_config.yaml'),
    output_dir=Path('./results'),
    log_level='INFO'
)

# Process a document (same as before)
success = framework.process_document(
    pdf_path=Path('./plan_desarrollo.pdf'),
    policy_code='PDM-2024-001'
)

print("✓ Analysis complete!" if success else "✗ Analysis failed")
```

### 3. Customize Thresholds (Optional)

Edit `my_config.yaml` to tune parameters:

```yaml
# Make Bayesian inference more strict
bayesian_thresholds:
  kl_divergence: 0.005  # Default: 0.01 (lower = stricter)

# Adjust for your domain
mechanism_type_priors:
  administrativo: 0.40  # If admin mechanisms are more common
  tecnico: 0.30
  # ... others
```

## Key Features Explained

### 1. Externalized Thresholds

**Before** (hardcoded):
```python
kl_threshold = 0.01  # In source code
```

**After** (configurable):
```yaml
bayesian_thresholds:
  kl_divergence: 0.01  # In config file
```

**Why**: Tune for different document types without editing code.

### 2. Schema Validation

**What happens**: Config is validated when loaded

```python
# This will fail with clear error message:
bayesian_thresholds:
  kl_divergence: 5.0  # ERROR: must be <= 1.0
```

**Why**: Catch mistakes early, before running expensive analysis.

### 3. Better Errors

**Before**:
```
Error: Error cargando configuración
```

**After**:
```
[CDAF Error] [Stage: config_validation] Configuración inválida
Details: {
  "validation_errors": [
    {
      "field": "kl_divergence",
      "error": "ensure this value is less than or equal to 1.0",
      "type": "value_error.number.not_le"
    }
  ]
}
```

**Why**: Know exactly what's wrong and where.

### 4. Performance Settings

```yaml
performance:
  max_context_length: 500   # Faster, less context
  cache_embeddings: true    # Slower first run, faster reruns
```

**Why**: Trade speed for accuracy based on your needs.

### 5. Self-Learning (Advanced)

```yaml
self_reflection:
  enable_prior_learning: true    # Learn from each document
  feedback_weight: 0.1           # How much to trust new data
  prior_history_path: './priors.json'  # Save learned knowledge
```

**Why**: System improves with each processed document.

## Configuration Reference

### Essential Settings

```yaml
# Bayesian inference strictness
bayesian_thresholds:
  kl_divergence: 0.01            # Lower = stricter convergence
  convergence_min_evidence: 2    # Min evidence before convergence

# Domain knowledge about mechanisms
mechanism_type_priors:
  administrativo: 0.30  # Must sum to 1.0
  tecnico: 0.25
  financiero: 0.20
  politico: 0.15
  mixto: 0.10
```

### Performance Tuning

```yaml
performance:
  max_context_length: 1000    # 500=fast, 2000=accurate
  cache_embeddings: true      # true=fast reruns, false=low memory
```

### Self-Learning (Optional)

```yaml
self_reflection:
  enable_prior_learning: false   # Set true after 5+ documents
  feedback_weight: 0.1           # 0.05-0.2 recommended
```

## Troubleshooting

### "Module pydantic not found"
```bash
pip install pydantic
```

### "Configuración inválida"
Check the error details - they tell you exactly what's wrong:
```json
{
  "field": "bayesian_thresholds.kl_divergence",
  "error": "must be <= 1.0"
}
```

### "Config file not found"
```bash
# Create from example
cp config_example_enhanced.yaml my_config.yaml
```

## Testing

```bash
# Test syntax
python -m py_compile dereck_beach

# Test configuration features
python test_config_enhancements.py

# Test canonical notation (existing)
python test_canonical_notation.py
```

## Migration from Old Code

**Good news**: No migration needed! Old code works as-is.

Old code (still works):
```python
framework = CDAFFramework(
    config_path=Path('./config.yaml'),  # Old config format
    output_dir=Path('./results')
)
```

New features (optional):
```python
# Access new config values
kl_threshold = framework.config.get_bayesian_threshold('kl_divergence')

# Enable self-learning
framework.config.validated_config.self_reflection.enable_prior_learning = True
```

## Examples

### Example 1: Strict Analysis
```yaml
# For high-stakes documents requiring maximum confidence
bayesian_thresholds:
  kl_divergence: 0.005           # Very strict
  convergence_min_evidence: 5    # Need 5+ evidence pieces
```

### Example 2: Fast Analysis
```yaml
# For quick exploratory analysis
performance:
  max_context_length: 500        # Process less text
  cache_embeddings: false        # Don't cache
bayesian_thresholds:
  kl_divergence: 0.05            # Less strict
```

### Example 3: Production with Learning
```yaml
# For production systems processing many documents
performance:
  cache_embeddings: true         # Fast reruns
self_reflection:
  enable_prior_learning: true    # Learn over time
  feedback_weight: 0.15          # Moderate learning rate
  prior_history_path: './data/priors.json'
```

## Further Reading

- **DEREK_ENHANCEMENTS.md** - Complete technical documentation
- **config_example_enhanced.yaml** - Annotated configuration file
- **dereck_beach** - Source code with inline comments

## Support

If you encounter issues:

1. Check error message details (now very descriptive)
2. Validate your config: Python will tell you what's wrong
3. Review config_example_enhanced.yaml for correct format
4. Check DEREK_ENHANCEMENTS.md for architecture details

---

**Happy analyzing! The enhanced CDAF is ready for production use. 🚀**
