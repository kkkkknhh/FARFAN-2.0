#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest configuration and shared test utilities.
"""

import numpy as np
import pytest


def assert_close(actual, expected, *, rtol=1e-6, atol=1e-9):
    """
    Uniform assertion for floats or numpy arrays used across tests.
    Scalars -> pytest.approx; arrays -> numpy.testing.assert_allclose.
    
    Args:
        actual: The actual value to test
        expected: The expected value
        rtol: Relative tolerance (default: 1e-6)
        atol: Absolute tolerance (default: 1e-9)
    """
    try:
        # numpy scalars/arrays have ndarray-like behavior; use np.asarray
        a = np.asarray(actual)
        b = np.asarray(expected)
    except Exception:
        # fallback to pytest.approx for anything unexpected
        assert actual == pytest.approx(expected, rel=rtol, abs=atol)
        return

    if a.ndim == 0 and b.ndim == 0:
        # scalar
        assert actual == pytest.approx(expected, rel=rtol, abs=atol)
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
