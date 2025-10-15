#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Constants Module
=============================

Immutable calibration constants per SIN_CARRETA doctrine.
All orchestrators must use these values to ensure deterministic behavior.

CRITICAL CONTRACT:
- All constants are frozen (immutable)
- No runtime modification allowed
- Override only via test fixtures with explicit markers
"""

from dataclasses import dataclass
from typing import Final


class FrozenInstanceError(Exception):
    """Raised when attempting to modify frozen calibration constants"""
    pass


@dataclass(frozen=True)
class CalibrationConstants:
    """
    Immutable calibration constants enforced across all orchestrators.
    
    SIN_CARRETA Compliance:
    - Frozen dataclass prevents runtime mutation
    - Type hints enforce contract clarity
    - All values explicitly documented
    
    Usage:
        from infrastructure.calibration_constants import CALIBRATION
        
        if score >= CALIBRATION.COHERENCE_THRESHOLD:
            ...
    """
    
    # ========================================================================
    # COHERENCE THRESHOLDS
    # ========================================================================
    
    COHERENCE_THRESHOLD: Final[float] = 0.7
    """Minimum coherence score for plan quality (0.0-1.0)"""
    
    CAUSAL_INCOHERENCE_LIMIT: Final[int] = 5
    """Maximum allowable causal incoherence contradictions"""
    
    REGULATORY_DEPTH_FACTOR: Final[float] = 1.3
    """Regulatory analysis depth multiplier"""
    
    # ========================================================================
    # SEVERITY THRESHOLDS
    # ========================================================================
    
    CRITICAL_SEVERITY_THRESHOLD: Final[float] = 0.85
    """Contradiction severity threshold for CRITICAL classification"""
    
    HIGH_SEVERITY_THRESHOLD: Final[float] = 0.70
    """Contradiction severity threshold for HIGH classification"""
    
    MEDIUM_SEVERITY_THRESHOLD: Final[float] = 0.50
    """Contradiction severity threshold for MEDIUM classification"""
    
    # ========================================================================
    # AUDIT QUALITY GRADES
    # ========================================================================
    
    EXCELLENT_CONTRADICTION_LIMIT: Final[int] = 5
    """Maximum contradictions for EXCELLENT grade"""
    
    GOOD_CONTRADICTION_LIMIT: Final[int] = 10
    """Maximum contradictions for GOOD grade"""
    
    # ========================================================================
    # BAYESIAN INFERENCE THRESHOLDS
    # ========================================================================
    
    KL_DIVERGENCE_THRESHOLD: Final[float] = 0.01
    """KL divergence threshold for Bayesian convergence"""
    
    CONVERGENCE_MIN_EVIDENCE: Final[int] = 2
    """Minimum evidence count for convergence check"""
    
    PRIOR_ALPHA: Final[float] = 2.0
    """Default alpha parameter for Beta prior distribution"""
    
    PRIOR_BETA: Final[float] = 2.0
    """Default beta parameter for Beta prior distribution"""
    
    LAPLACE_SMOOTHING: Final[float] = 1.0
    """Laplace smoothing parameter for probability estimation"""
    
    # ========================================================================
    # MECHANISM TYPE PRIORS
    # ========================================================================
    
    MECHANISM_PRIOR_ADMINISTRATIVO: Final[float] = 0.30
    """Prior probability for administrative mechanisms"""
    
    MECHANISM_PRIOR_TECNICO: Final[float] = 0.25
    """Prior probability for technical mechanisms"""
    
    MECHANISM_PRIOR_FINANCIERO: Final[float] = 0.20
    """Prior probability for financial mechanisms"""
    
    MECHANISM_PRIOR_POLITICO: Final[float] = 0.15
    """Prior probability for political mechanisms"""
    
    MECHANISM_PRIOR_MIXTO: Final[float] = 0.10
    """Prior probability for mixed mechanisms"""
    
    # ========================================================================
    # PIPELINE CONFIGURATION
    # ========================================================================
    
    MIN_QUALITY_THRESHOLD: Final[float] = 0.5
    """Minimum extraction quality threshold for pipeline continuation"""
    
    WORKER_TIMEOUT_SECS: Final[int] = 300
    """Default worker timeout in seconds (5 minutes)"""
    
    MAX_INFLIGHT_JOBS: Final[int] = 3
    """Maximum concurrent jobs (backpressure limit)"""
    
    QUEUE_SIZE: Final[int] = 10
    """Maximum queue size for job backpressure"""
    
    D6_ALERT_THRESHOLD: Final[float] = 0.55
    """D6 dimension score threshold for critical alerts"""
    
    def __post_init__(self):
        """
        Validate calibration constants on instantiation.
        
        SIN_CARRETA Compliance:
        - Enforce probability sum constraints
        - Validate threshold ordering
        - Ensure non-negative values
        """
        # Validate mechanism priors sum to approximately 1.0
        mechanism_sum = (
            self.MECHANISM_PRIOR_ADMINISTRATIVO +
            self.MECHANISM_PRIOR_TECNICO +
            self.MECHANISM_PRIOR_FINANCIERO +
            self.MECHANISM_PRIOR_POLITICO +
            self.MECHANISM_PRIOR_MIXTO
        )
        if abs(mechanism_sum - 1.0) > 0.01:
            raise ValueError(
                f"Mechanism type priors must sum to 1.0, got {mechanism_sum}"
            )
        
        # Validate severity threshold ordering
        if not (
            self.CRITICAL_SEVERITY_THRESHOLD > 
            self.HIGH_SEVERITY_THRESHOLD > 
            self.MEDIUM_SEVERITY_THRESHOLD
        ):
            raise ValueError("Severity thresholds must be strictly ordered")
        
        # Validate audit grade ordering
        if not self.EXCELLENT_CONTRADICTION_LIMIT < self.GOOD_CONTRADICTION_LIMIT:
            raise ValueError("Audit grade limits must be strictly ordered")
        
        # Validate non-negative constraints
        if any([
            self.COHERENCE_THRESHOLD < 0,
            self.CAUSAL_INCOHERENCE_LIMIT < 0,
            self.REGULATORY_DEPTH_FACTOR < 0,
            self.KL_DIVERGENCE_THRESHOLD < 0,
            self.CONVERGENCE_MIN_EVIDENCE < 1,
            self.MIN_QUALITY_THRESHOLD < 0
        ]):
            raise ValueError("Calibration constants must be non-negative")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

CALIBRATION: Final[CalibrationConstants] = CalibrationConstants()
"""
Global immutable calibration constants singleton.

All orchestrators MUST import and use this singleton:

    from infrastructure.calibration_constants import CALIBRATION
    
    # Correct usage
    if score >= CALIBRATION.COHERENCE_THRESHOLD:
        pass
    
    # FORBIDDEN (will raise FrozenInstanceError)
    CALIBRATION.COHERENCE_THRESHOLD = 0.8  # ❌ COMPILE ERROR
"""


def override_calibration(**kwargs) -> CalibrationConstants:
    """
    Create a new CalibrationConstants instance with overrides.
    
    USE ONLY IN TESTING with explicit markers:
    
        @pytest.mark.override_calibration
        def test_with_custom_threshold():
            custom_cal = override_calibration(COHERENCE_THRESHOLD=0.9)
            orchestrator = MyOrchestrator(calibration=custom_cal)
            ...
    
    Args:
        **kwargs: Calibration constant overrides
        
    Returns:
        New CalibrationConstants instance with overrides
        
    Raises:
        ValueError: If override values are invalid
    """
    # Get current values as dict
    current = {
        field.name: getattr(CALIBRATION, field.name)
        for field in CalibrationConstants.__dataclass_fields__.values()
    }
    
    # Apply overrides
    current.update(kwargs)
    
    # Create new instance (will validate in __post_init__)
    return CalibrationConstants(**current)


def validate_calibration_consistency(modules: list) -> dict:
    """
    Validate that all modules use the same calibration constants.
    
    SIN_CARRETA Enforcement:
    - Scan modules for hardcoded constants
    - Compare against CALIBRATION singleton
    - Report violations
    
    Args:
        modules: List of module objects to validate
        
    Returns:
        Validation report with violations
    """
    violations = []
    
    hardcoded_patterns = [
        (r'COHERENCE_THRESHOLD\s*=\s*([\d.]+)', 'COHERENCE_THRESHOLD'),
        (r'CAUSAL_INCOHERENCE_LIMIT\s*=\s*(\d+)', 'CAUSAL_INCOHERENCE_LIMIT'),
        (r'REGULATORY_DEPTH_FACTOR\s*=\s*([\d.]+)', 'REGULATORY_DEPTH_FACTOR'),
    ]
    
    import re
    import inspect
    
    for module in modules:
        source = inspect.getsource(module)
        for pattern, constant_name in hardcoded_patterns:
            matches = re.findall(pattern, source)
            if matches:
                expected = getattr(CALIBRATION, constant_name)
                for match in matches:
                    if float(match) != expected:
                        violations.append({
                            'module': module.__name__,
                            'constant': constant_name,
                            'found': float(match),
                            'expected': expected
                        })
    
    return {
        'passed': len(violations) == 0,
        'violations': violations,
        'scanned_modules': len(modules)
    }
