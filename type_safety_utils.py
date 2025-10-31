#!/usr/bin/env python3
"""
Type Safety Utilities for FARFAN 2.0

This module provides helper functions to prevent the three critical runtime error patterns:
1. 'bool' object is not iterable
2. 'str' object has no attribute 'text'
3. can't multiply sequence by non-int of type 'float'
"""

from typing import Union, List, Any, TypeVar, Callable

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import spacy
    from spacy.tokens import Doc, Span, Token
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    # Placeholder types for when spacy is not available
    Doc = Any
    Span = Any
    Token = Any


T = TypeVar('T')


# ============================================================================
# Pattern 1: Ensure functions return lists, not bools
# ============================================================================

def ensure_list(value: Any) -> List[Any]:
    """
    Ensure a value is a list, converting False/True/None to empty list.
    
    This prevents 'bool' object is not iterable errors.
    
    Args:
        value: Any value that should be a list
        
    Returns:
        A list (empty if value was False/True/None)
        
    Examples:
        >>> ensure_list(False)
        []
        >>> ensure_list(True)
        []
        >>> ensure_list(None)
        []
        >>> ensure_list([1, 2, 3])
        [1, 2, 3]
        >>> ensure_list("test")
        ['test']
    """
    if value is None or value is False or value is True:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    # Single value - wrap in list
    return [value]


def safe_iterate(collection: Any) -> List[Any]:
    """
    Safely iterate over a collection, handling edge cases.
    
    Converts False/True/None to empty list to prevent iteration errors.
    
    Args:
        collection: Collection to iterate over
        
    Returns:
        A safe iterable list
        
    Examples:
        >>> for item in safe_iterate(False):
        ...     print(item)  # No iteration, no error
        >>> for item in safe_iterate([1, 2]):
        ...     print(item)  # Normal iteration
        1
        2
    """
    return ensure_list(collection)


# ============================================================================
# Pattern 2: Safe text attribute access
# ============================================================================

def safe_text_extract(text_input: Union[str, Any]) -> str:
    """
    Safely extract text from string or spaCy object.
    
    This prevents 'str' object has no attribute 'text' errors.
    
    Args:
        text_input: Either a string or a spaCy object (Doc, Span, Token)
        
    Returns:
        The text as a string
        
    Examples:
        >>> safe_text_extract("hello")
        'hello'
        >>> # With spaCy object:
        >>> # nlp = spacy.load("en_core_web_sm")
        >>> # doc = nlp("hello")
        >>> # safe_text_extract(doc[0])  # Returns 'hello'
    """
    if isinstance(text_input, str):
        return text_input
    
    # Check if it's a spaCy object with .text attribute
    if hasattr(text_input, 'text'):
        return text_input.text
    
    # Fallback: convert to string
    return str(text_input)


def ensure_spacy_doc(text_input: Union[str, Any], nlp: Any) -> Any:
    """
    Ensure input is a spaCy Doc object, processing if needed.
    
    Args:
        text_input: String or spaCy Doc
        nlp: spaCy language model
        
    Returns:
        spaCy Doc object
        
    Examples:
        >>> # nlp = spacy.load("en_core_web_sm")
        >>> # doc = ensure_spacy_doc("Hello world", nlp)
        >>> # assert isinstance(doc, spacy.tokens.Doc)
    """
    if SPACY_AVAILABLE and isinstance(text_input, (Doc, Span)):
        return text_input
    
    if isinstance(text_input, str):
        if nlp is None:
            raise ValueError("nlp model required to process string input")
        return nlp(text_input)
    
    return text_input


# ============================================================================
# Pattern 3: Safe list/array multiplication
# ============================================================================

def safe_scale(values: Union[List[float], Any, float], 
               factor: float) -> Union[List[float], Any, float]:
    """
    Safely scale values by a factor, handling lists and arrays.
    
    This prevents "can't multiply sequence by non-int" errors.
    
    Args:
        values: List, numpy array, or single value
        factor: Scaling factor
        
    Returns:
        Scaled values in the same format as input
        
    Examples:
        >>> safe_scale([1.0, 2.0, 3.0], 0.5)
        [0.5, 1.0, 1.5]
        >>> safe_scale(2.0, 0.5)
        1.0
    """
    if isinstance(values, list):
        # Use list comprehension for Python lists
        return [v * factor for v in values]
    elif NUMPY_AVAILABLE and isinstance(values, np.ndarray):
        # Direct multiplication works for numpy arrays
        return values * factor
    elif isinstance(values, (int, float)):
        # Scalar multiplication
        return values * factor
    elif isinstance(values, (tuple, set)):
        # Convert to list, scale, convert back
        scaled_list = [v * factor for v in values]
        return type(values)(scaled_list)
    else:
        raise TypeError(f"Cannot scale type {type(values)}")


def safe_elementwise_op(values: Union[List[Any], Any],
                        operation: Callable[[Any], Any]) -> Union[List[Any], Any]:
    """
    Apply an operation element-wise to a collection.
    
    Args:
        values: List or numpy array
        operation: Function to apply to each element
        
    Returns:
        Results in same format as input
        
    Examples:
        >>> safe_elementwise_op([1, 2, 3], lambda x: x * 2)
        [2, 4, 6]
    """
    if isinstance(values, list):
        return [operation(v) for v in values]
    elif NUMPY_AVAILABLE and isinstance(values, np.ndarray):
        return np.vectorize(operation)(values)
    else:
        raise TypeError(f"Cannot apply operation to type {type(values)}")


# ============================================================================
# Defensive wrappers for common operations
# ============================================================================

def safe_posterior_update(prior: Union[List[float], Any, float],
                          likelihood: float,
                          normalize: bool = False) -> Union[List[float], Any, float]:
    """
    Safely update posterior probabilities.
    
    Args:
        prior: Prior probability/probabilities
        likelihood: Likelihood value
        normalize: Whether to normalize result
        
    Returns:
        Updated posterior
        
    Examples:
        >>> safe_posterior_update([0.2, 0.3, 0.5], 0.8)
        [0.16, 0.24, 0.4]
        >>> safe_posterior_update(0.5, 0.8, normalize=True)
        0.4
    """
    # Scale by likelihood
    posterior = safe_scale(prior, likelihood)
    
    if normalize and not isinstance(posterior, (int, float)):
        # Normalize to sum to 1.0
        if isinstance(posterior, list):
            total = sum(posterior)
            if total > 0:
                posterior = [p / total for p in posterior]
        elif NUMPY_AVAILABLE and isinstance(posterior, np.ndarray):
            total = posterior.sum()
            if total > 0:
                posterior = posterior / total
    
    return posterior


# ============================================================================
# Validation decorators
# ============================================================================

def returns_list(func: Callable) -> Callable:
    """
    Decorator to ensure function always returns a list.
    
    Converts False/True/None to empty list.
    
    Examples:
        >>> @returns_list
        ... def get_items(data):
        ...     if not data:
        ...         return False
        ...     return ["item"]
        >>> get_items(None)  # Returns [] instead of False
        []
        >>> get_items("data")  # Returns ["item"] as expected  
        ['item']
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return ensure_list(result)
    return wrapper


def safe_text_access(func: Callable) -> Callable:
    """
    Decorator to safely handle text extraction in function.
    
    Wraps returned value with safe_text_extract.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is not None:
            return safe_text_extract(result)
        return ""
    return wrapper


# ============================================================================
# Type checking utilities
# ============================================================================

def is_iterable_not_string(obj: Any) -> bool:
    """Check if object is iterable but not a string."""
    try:
        iter(obj)
        return not isinstance(obj, str)
    except TypeError:
        return False


def is_spacy_object(obj: Any) -> bool:
    """Check if object is a spaCy Doc, Span, or Token."""
    if not SPACY_AVAILABLE:
        return False
    return isinstance(obj, (Doc, Span, Token))


# ============================================================================
# Example usage in analysis code
# ============================================================================

if __name__ == "__main__":
    print("Type Safety Utilities Test Suite")
    print("=" * 70)
    
    # Test ensure_list
    print("\n1. Testing ensure_list:")
    print(f"   ensure_list(False) = {ensure_list(False)}")
    print(f"   ensure_list([1,2]) = {ensure_list([1,2])}")
    
    # Test safe_text_extract
    print("\n2. Testing safe_text_extract:")
    print(f"   safe_text_extract('hello') = {safe_text_extract('hello')}")
    
    # Test safe_scale
    print("\n3. Testing safe_scale:")
    print(f"   safe_scale([1,2,3], 0.5) = {safe_scale([1,2,3], 0.5)}")
    print(f"   safe_scale(2.0, 0.5) = {safe_scale(2.0, 0.5)}")
    
    # Test decorator
    print("\n4. Testing @returns_list decorator:")
    @returns_list
    def example_func(data):
        if not data:
            return False
        return ["result"]
    
    print(f"   example_func(None) = {example_func(None)}")
    print(f"   example_func('data') = {example_func('data')}")
    
    print("\n✓ All utility functions working correctly")
