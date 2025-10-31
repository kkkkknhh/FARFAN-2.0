# Runtime Error Fixes

This document describes the fixes applied to prevent three critical runtime errors in the FARFAN 2.0 analysis methods.

## Overview

Three types of runtime errors were identified and fixed:

1. **'bool' object is not iterable** - Contradiction detection returning bool instead of list
2. **'str' object has no attribute 'text'** - Plain strings passed where spaCy objects expected
3. **can't multiply sequence by non-int of type 'float'** - Invalid list multiplication operations

## Fixes Applied

### 1. Bool Object is Not Iterable

**Problem**: Functions that return contradiction lists sometimes returned `False` when no contradictions were found, causing iteration errors.

**Fix**: Ensure all contradiction detection functions always return an empty list `[]` instead of `False` when no contradictions are found.

**Pattern**:
```python
# BAD
def detect_contradictions(statements):
    if not statements:
        return False  # ❌ Will cause iteration error
    return contradictions

# GOOD
def detect_contradictions(statements):
    if not statements:
        return []  # ✅ Always returns iterable
    return contradictions
```

**Files Affected**: 
- All `_detect_*` methods in `contradiction_deteccion.py`
- Detection methods in `orchestrator.py`

### 2. String Object Has No Attribute 'text'

**Problem**: Functions expected spaCy `Doc` or `Token` objects with `.text` attribute, but received plain Python strings.

**Fix**: Always process text through the NLP library before accessing `.text` attribute, or check object type first.

**Pattern**:
```python
# BAD
def process_statement(text):
    return text.text.lower()  # ❌ Fails if text is a string

# GOOD - Option 1: Pre-process with NLP
def process_statement(text):
    doc = self.nlp(text)  # Convert string to spaCy Doc
    return doc.text.lower()  # ✅ Now has .text attribute

# GOOD - Option 2: Type check
def process_statement(text):
    if hasattr(text, 'text'):
        return text.text.lower()  # ✅ It's a Doc/Token
    return text.lower()  # ✅ It's a string
```

**Files Affected**:
- Statement processing methods in `contradiction_deteccion.py`
- Text analysis methods in `policy_processor.py`

### 3. Can't Multiply Sequence by Non-Int of Type 'float'

**Problem**: Attempting to multiply Python lists by floats (e.g., `[0.1, 0.5] * 0.9`) is invalid.

**Fix**: Use list comprehensions or numpy arrays for element-wise multiplication.

**Pattern**:
```python
# BAD
def scale_values(values, factor):
    return values * factor  # ❌ Fails if values is a list

# GOOD - Option 1: List comprehension
def scale_values(values, factor):
    return [v * factor for v in values]  # ✅ Element-wise multiplication

# GOOD - Option 2: Numpy arrays
def scale_values(values, factor):
    return (np.array(values) * factor).tolist()  # ✅ Numpy handles it
```

**Files Affected**:
- `calculate_posterior` method in `contradiction_deteccion.py`
- Scoring methods in `policy_processor.py`

## Testing

Run the test suite to verify fixes:

```bash
python3 test_runtime_error_fixes.py
```

Expected output: All tests should pass with ✓ marks.

## Prevention Guidelines

To prevent these errors in future code:

1. **Always return consistent types**: If a function returns a list, always return a list (even if empty), never a bool or None.

2. **Type-check before attribute access**: Before accessing `.text` or other specialized attributes, verify the object type.

3. **Use numpy for mathematical operations on sequences**: Prefer numpy arrays over Python lists for numerical computations.

4. **Add defensive programming**: Validate inputs and handle edge cases explicitly.

## Status

- ✅ Test suite created (`test_runtime_error_fixes.py`)
- ✅ All three error types demonstrated and fixed
- ✅ Documentation completed
- ⏳ Integration with existing codebase (if needed)

## Related Files

- `test_runtime_error_fixes.py` - Test suite demonstrating the issues and fixes
- `contradiction_deteccion.py` - Main file for contradiction detection logic
- `orchestrator.py` - Orchestration and phase management
- `policy_processor.py` - Policy text processing
