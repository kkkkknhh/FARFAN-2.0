"""
Default floating-point comparison tolerances for FARFAN 2.0.

These constants define the default relative and absolute tolerances
for floating-point comparisons across the codebase.

References:
- numpy.isclose default: rtol=1e-05, atol=1e-08
- math.isclose default: rel_tol=1e-09, abs_tol=0.0
- pytest.approx default: rel=1e-06, abs=1e-12

For FARFAN 2.0, we use conservative tolerances that balance
numerical precision with practical comparison needs.
"""

# Default tolerances for floating-point comparisons
DEFAULT_FLOAT_TOLS = {
    "rel_tol": 1e-9,  # Relative tolerance (0.0000001%)
    "abs_tol": 1e-12,  # Absolute tolerance
}

# Export as individual constants for convenience
REL_TOL = DEFAULT_FLOAT_TOLS["rel_tol"]
ABS_TOL = DEFAULT_FLOAT_TOLS["abs_tol"]
