# Type Safety Best Practices for FARFAN 2.0

This document describes three critical runtime error patterns and how to prevent them in the FARFAN 2.0 codebase.

## Error Pattern 1: 'bool' object is not iterable

### Problem
Functions that should return collections (lists, tuples, etc.) sometimes return boolean values instead, causing iteration errors.

### Example of the Bug
```python
def detect_contradictions(statements):
    if not statements:
        return False  # BUG! Should return []
    
    contradictions = []
    # ... detection logic ...
    return contradictions

# This will crash:
for contradiction in detect_contradictions([]):
    print(contradiction)
# TypeError: 'bool' object is not iterable
```

### Solution
Always return an empty collection instead of False:

```python
def detect_contradictions(statements):
    if not statements:
        return []  # Correct!
    
    contradictions = []
    # ... detection logic ...
    return contradictions
```

### Where to Check
- Any function with names like: `detect_*`, `extract_*`, `get_*`, `find_*`, `collect_*`
- Any function with return type hints like `List[...]`, `Tuple[...]`, etc.
- Check all return statements in these functions

### Prevention Pattern
```python
from typing import List

def detection_function(data) -> List[SomeType]:
    """Always declare return type as List."""
    if not data:
        return []  # Never return False/True/None
    
    results = []
    # ... logic ...
    return results
```

## Error Pattern 2: 'str' object has no attribute 'text'

### Problem
Code expects a spaCy object (Token, Span, Doc) with a `.text` attribute but receives a plain Python string.

### Example of the Bug
```python
def process_sentence(sent):
    # BUG! Assumes sent is always a spaCy Span
    return sent.text.lower()  # Fails if sent is already a string

# This will crash:
text = "Hello world"
result = process_sentence(text)
# AttributeError: 'str' object has no attribute 'text'
```

### Solution
Add type checking or ensure text is always processed through nlp():

```python
def process_sentence(sent):
    # Defensive: check if already a string
    if isinstance(sent, str):
        return sent.lower()
    # Otherwise, it's a spaCy object
    return sent.text.lower()

# Or ensure spaCy processing:
def process_document(text_input, nlp):
    # Always process through spaCy first
    doc = nlp(text_input)  # Creates spaCy Doc object
    for sent in doc.sents:  # sent is Span object
        # Now safe to access .text
        print(sent.text)
```

### Where to Check
- Functions that call `nlp()` and extract `doc.sents`, `doc.ents`, or iterate over `doc`
- Any place accessing `.text`, `.lemma_`, `.pos_`, `.dep_` attributes
- Functions receiving parameters that might be strings OR spaCy objects

### Prevention Pattern
```python
from typing import Union
import spacy

def safe_text_access(text_input: Union[str, spacy.tokens.Span]) -> str:
    """Safely extract text from string or spaCy object."""
    if isinstance(text_input, str):
        return text_input
    # Has .text attribute (Span, Token, Doc)
    return text_input.text
```

## Error Pattern 3: can't multiply sequence by non-int of type 'float'

### Problem
Attempting to multiply a Python list by a float, which is only valid for numpy arrays.

### Example of the Bug
```python
def calculate_posterior(prior_list, weight):
    # BUG! This only works with numpy arrays
    return prior_list * weight  # Fails if prior_list is a list

# This will crash:
priors = [0.1, 0.3, 0.6]
result = calculate_posterior(priors, 0.8)
# TypeError: can't multiply sequence by non-int of type 'float'
```

### Solution
Use list comprehension for Python lists, or ensure you're using numpy:

```python
import numpy as np
from typing import Union, List

def calculate_posterior(priors: Union[List[float], np.ndarray], 
                       weight: float) -> Union[List[float], np.ndarray]:
    """Scale priors by weight, handling both lists and arrays."""
    if isinstance(priors, list):
        # Use list comprehension for Python lists
        return [p * weight for p in priors]
    # numpy array - direct multiplication works
    return priors * weight
```

### Where to Check
- Any math operations on variables named: `priors`, `posteriors`, `weights`, `scores`, `values`
- Operations involving `* float_value` on collections
- Bayesian calculations and probability updates

### Prevention Pattern
```python
# Option 1: Always use numpy
import numpy as np

def calculate_with_numpy(values, scale):
    values_array = np.array(values)  # Convert to numpy
    return values_array * scale      # Safe multiplication

# Option 2: Defensive type checking
def calculate_defensive(values, scale):
    if isinstance(values, list):
        return [v * scale for v in values]
    elif isinstance(values, np.ndarray):
        return values * scale
    else:  # scalar
        return values * scale

# Option 3: Type hints enforce numpy from start
def calculate_typed(values: np.ndarray, scale: float) -> np.ndarray:
    return values * scale  # Caller must pass numpy array
```

## Quick Reference

| Error | Anti-Pattern | Correct Pattern |
|:------|:-------------|:----------------|
| **Bool iteration** | `return False` | `return []` |
| **Text attribute** | `string.text` | `if isinstance(x, str): use x` |
| **List multiplication** | `list * 0.5` | `[item * 0.5 for item in list]` |

## Testing

Run the type safety tests to verify these patterns:

```bash
python3 test_type_safety_fixes.py
```

## Code Review Checklist

Before committing code, verify:

- [ ] All collection-returning functions use `-> List[...]` type hints
- [ ] No function returns `False`/`True` where `[]` is expected
- [ ] All `.text` accesses are on verified spaCy objects
- [ ] No list * float operations (use comprehension or numpy)
- [ ] Type hints are used for function parameters and returns
- [ ] Defensive type checking is added where types might vary

## References

- Python typing: https://docs.python.org/3/library/typing.html
- spaCy objects: https://spacy.io/api/doc
- NumPy arrays: https://numpy.org/doc/stable/reference/arrays.html
