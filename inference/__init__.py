"""
Inference module for Bayesian mechanism analysis.

This module provides structured Bayesian inference components with
clear separation of concerns between prior construction, sampling,
and necessity/sufficiency testing.
"""

from .bayesian_engine import (
    BayesianPriorBuilder,
    BayesianSamplingEngine,
    CausalLink,
    ColombianMunicipalContext,
    DocumentEvidence,
    EvidenceChunk,
    InferenceExplainabilityPayload,
    MechanismEvidence,
    MechanismPrior,
    NecessitySufficiencyTester,
    NecessityTestResult,
    PosteriorDistribution,
    SamplingConfig,
)

__all__ = [
    "BayesianPriorBuilder",
    "BayesianSamplingEngine",
    "NecessitySufficiencyTester",
    "MechanismPrior",
    "PosteriorDistribution",
    "NecessityTestResult",
    "MechanismEvidence",
    "EvidenceChunk",
    "SamplingConfig",
    "CausalLink",
    "ColombianMunicipalContext",
    "InferenceExplainabilityPayload",
    "DocumentEvidence",
]
