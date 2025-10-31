# Type Safety Utilities - Quick Start Guide

This guide helps you quickly use the type safety utilities to prevent common runtime errors in FARFAN 2.0.

## Three Error Patterns We Prevent

| Error | When It Happens | How We Fix It |
|:------|:----------------|:--------------|
| **'bool' object is not iterable** | Function returns `False` instead of `[]` | Use `ensure_list()` or `@returns_list` |
| **'str' has no attribute 'text'** | String passed where spaCy object expected | Use `safe_text_extract()` |
| **List * float error** | Python list multiplied by float | Use `safe_scale()` |

## Quick Examples

### Example 1: Ensuring Safe Iteration

```python
from type_safety_utils import ensure_list

# Problem: This crashes if detect_issues returns False
issues = detect_issues(data)
for issue in issues:  # TypeError if issues is False!
    print(issue)

# Solution: Always get a list
issues = ensure_list(detect_issues(data))
for issue in issues:  # Safe! Empty loop if no issues
    print(issue)
```

### Example 2: Using the Decorator

```python
from type_safety_utils import returns_list

@returns_list
def get_contradictions(statements):
    if not statements:
        return False  # Automatically converted to []
    # ... detection logic ...
    return contradictions

# Always safe to iterate
for c in get_contradictions(data):
    process(c)
```

### Example 3: Safe Text Extraction

```python
from type_safety_utils import safe_text_extract

# Works with both strings and spaCy objects
def process_input(text_or_doc):
    text = safe_text_extract(text_or_doc)
    return text.lower()

# Both work!
process_input("Hello")  # String → "hello"
process_input(nlp("Hello")[0])  # Token → "hello"
```

### Example 4: Safe List Scaling

```python
from type_safety_utils import safe_scale

# Problem: This crashes
priors = [0.2, 0.3, 0.5]
result = priors * 0.8  # TypeError!

# Solution: Use safe_scale
priors = [0.2, 0.3, 0.5]
result = safe_scale(priors, 0.8)  # [0.16, 0.24, 0.4]
```

### Example 5: Bayesian Updates

```python
from type_safety_utils import safe_posterior_update

# Safe Bayesian update with optional normalization
prior = [0.2, 0.3, 0.5]
posterior = safe_posterior_update(
    prior, 
    likelihood=0.8, 
    normalize=True
)
# Works with lists, numpy arrays, or scalars!
```

## All Available Functions

### Pattern 1: List Returns
- `ensure_list(value)` - Convert any value to list
- `safe_iterate(collection)` - Safe iteration wrapper
- `@returns_list` - Decorator ensuring list return

### Pattern 2: Text Extraction  
- `safe_text_extract(text_input)` - Extract from string or spaCy object
- `ensure_spacy_doc(text_input, nlp)` - Ensure spaCy Doc object
- `@safe_text_access` - Decorator for safe text access

### Pattern 3: List Multiplication
- `safe_scale(values, factor)` - Scale lists/arrays safely
- `safe_elementwise_op(values, operation)` - Apply operation to each element
- `safe_posterior_update(prior, likelihood, normalize)` - Bayesian update

### Type Checking
- `is_iterable_not_string(obj)` - Check if iterable but not string
- `is_spacy_object(obj)` - Check if spaCy Doc/Span/Token

## When to Use Each Function

| Use Case | Function | Example |
|:---------|:---------|:--------|
| Function might return bool | `ensure_list()` | `claims = ensure_list(get_claims())` |
| Creating list-returning function | `@returns_list` | `@returns_list def detect()` |
| Processing text (string or spaCy) | `safe_text_extract()` | `text = safe_text_extract(input)` |
| Scaling probabilities/weights | `safe_scale()` | `posterior = safe_scale(prior, 0.8)` |
| Bayesian inference | `safe_posterior_update()` | `post = safe_posterior_update(prior, like)` |

## Testing Your Code

Run the test suite to see examples of all patterns:

```bash
python3 test_type_safety_fixes.py
```

Run the utilities self-test:

```bash
python3 type_safety_utils.py
```

## More Information

- **TYPE_SAFETY_GUIDE.md** - Complete reference with detailed examples
- **TYPE_SAFETY_IMPLEMENTATION_SUMMARY.md** - Full implementation report
- **test_type_safety_fixes.py** - Working examples and test cases

## Common Gotchas

❌ **Don't Do This:**
```python
# Returns bool - will crash on iteration
def detect():
    if not data:
        return False
    return []

# Multiplies list by float - TypeError
result = [1, 2, 3] * 0.5

# Assumes token is always spaCy object
return token.text
```

✅ **Do This Instead:**
```python
# Always return list
def detect():
    if not data:
        return []  # Or use @returns_list decorator
    return []

# Use list comprehension or safe_scale
result = [x * 0.5 for x in [1, 2, 3]]
# Or: result = safe_scale([1, 2, 3], 0.5)

# Check type first or use safe_text_extract
return safe_text_extract(token)
```

## Questions?

See TYPE_SAFETY_GUIDE.md for complete documentation with:
- Detailed explanations of each pattern
- More code examples
- Prevention strategies
- Code review checklist
