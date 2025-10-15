# Float Comparison Replacement Summary

## Overview
This document summarizes the replacement of direct floating-point equality checks with numeric-safe alternatives across the FARFAN 2.0 repository.

## Problem Statement
Direct equality (`==`) and inequality (`!=`) comparisons on floating-point values are numerically unstable due to:
- Binary representation limitations (e.g., 0.1 + 0.1 + 0.1 != 0.3)
- Accumulated rounding errors in calculations
- Platform-specific floating-point implementations

## Solution
Replace all direct float comparisons with tolerance-based alternatives:
- **Scalar comparisons**: `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)`
- **Array comparisons**: `np.isclose(a, b, rtol=1e-9, atol=1e-12)` or `np.allclose(...)`
- **Test assertions**: `assert a == pytest.approx(b, rel=1e-9, abs=1e-12)`

## Default Tolerances
Created `float_tolerance_config.py` with repository-wide defaults:
- **Relative tolerance**: 1e-9 (0.0000001%)
- **Absolute tolerance**: 1e-12

These conservative values balance numerical precision with practical comparison needs.

## Changes Summary

### Files Modified: 13 files, 33 float comparisons replaced

#### Test Files (7 files, 23 comparisons)
1. **test_choreography.py** (10 comparisons)
   - All test assertions now use `pytest.approx`
   - Covers progress calculations, prior/posterior distributions, thresholds

2. **test_circuit_breaker.py** (4 comparisons)
   - Added `import pytest`
   - Score penalty and circuit state comparisons use `pytest.approx`

3. **test_ior_audit.py** (2 comparisons)
   - Confidence score assertions use `pytest.approx`

4. **test_ior_audit_points.py** (2 comparisons)
   - Added `import pytest`
   - Posterior capping and necessity scores use `pytest.approx`

5. **test_observability.py** (1 comparison)
   - Added `import pytest`
   - Gauge metric assertion uses `pytest.approx`

6. **test_part4_ior.py** (2 comparisons)
   - Added `import pytest`
   - Degradation penalty and score comparisons use `pytest.approx`

7. **test_unified_orchestrator.py** (2 comparisons)
   - Inequality check uses negated `pytest.approx`
   - Prior alpha value assertion uses `pytest.approx`

#### Production Code Files (5 files, 10 comparisons)
1. **scoring_framework.py** (3 comparisons)
   - Added `import math`
   - Weight initialization checks use `math.isclose`
   - Score boundary check (1.0) uses `math.isclose`

2. **calibration_validator.py** (1 comparison)
   - Added `import math`
   - Stability rate check (100.0) uses `math.isclose`

3. **example_ior_audit.py** (2 comparisons)
   - Added `import math`
   - Overall compliance checks (100.0) use `math.isclose`

4. **validators/ior_validator.py** (1 comparison)
   - Added `import math`
   - SOTA MMR compliance check uses `math.isclose`

5. **audits/causal_mechanism_auditor.py** (3 comparisons)
   - Added `import math`
   - Necessity score check (1.0) uses `math.isclose`
   - Extraction accuracy check (1.0) uses `math.isclose`
   - Complete extraction count uses `np.isclose` for NumPy array context

#### Configuration File (1 new file)
- **float_tolerance_config.py** (new)
  - Defines `DEFAULT_FLOAT_TOLS`, `REL_TOL`, `ABS_TOL`
  - Provides centralized tolerance configuration

## Pattern Examples

### Before → After

#### Scalar Float Comparison
```python
# Before
if score == 1.0:
    return "excelente"

# After
import math
if math.isclose(score, 1.0, rel_tol=1e-9, abs_tol=1e-12):  # replaced float equality with isclose
    return "excelente"
```

#### Test Assertion
```python
# Before
assert result.score_penalty == 0.05

# After
import pytest
assert result.score_penalty == pytest.approx(0.05, rel=1e-9, abs=1e-12)  # replaced float equality with pytest.approx
```

#### Float Inequality
```python
# Before
assert snapshot.priors['tecnico'] != 5.0

# After
import pytest
assert not (snapshot.priors['tecnico'] == pytest.approx(5.0, rel=1e-9, abs=1e-12))  # replaced float inequality with negated pytest.approx
```

#### Array Comparison (NumPy)
```python
# Before
if r.extraction_accuracy == 1.0

# After
import numpy as np
if np.isclose(r.extraction_accuracy, 1.0, rtol=1e-9, atol=1e-12)  # replaced float equality with isclose
```

## Validation
All changes have been validated:
1. ✅ Import statements added correctly
2. ✅ Syntax is valid (all files compile without errors)
3. ✅ Tolerances are consistent across the codebase
4. ✅ Inline comments document the changes
5. ✅ Edge cases tested (0.1 + 0.1 + 0.1 vs 0.3, 1/3*3 vs 1.0)

## Best Practices
1. **Use math.isclose for scalar floats** in production code
2. **Use pytest.approx in tests** for cleaner assertion syntax
3. **Use np.isclose/np.allclose for arrays** when working with NumPy
4. **Document tolerances** with inline comments referencing DEFAULT_FLOAT_TOLS
5. **Consider domain knowledge** - some comparisons may need tighter/looser tolerances

## Impact
- **Improved numerical stability**: Eliminates false negatives from rounding errors
- **Better maintainability**: Centralized tolerance configuration
- **Clearer intent**: Explicit tolerance values document expected precision
- **Future-proof**: Easier to adjust tolerances globally if needed

## References
- numpy.isclose default: rtol=1e-05, atol=1e-08
- math.isclose default: rel_tol=1e-09, abs_tol=0.0
- pytest.approx default: rel=1e-06, abs=1e-12
- FARFAN 2.0 chosen: rel_tol=1e-09, abs_tol=1e-12 (conservative)
