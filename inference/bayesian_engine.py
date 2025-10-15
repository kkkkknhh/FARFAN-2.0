#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Engine - Refactored Mechanism Inference
F1.2: Refactorización del Motor Bayesiano

Separación cristalina de concerns:
- BayesianPriorBuilder: Construye priors adaptativos basados en evidencia estructural
- BayesianSamplingEngine: Ejecuta MCMC sampling con reproducibilidad garantizada
- NecessitySufficiencyTester: Ejecuta Hoop Tests determinísticos

Objetivo: Eliminar duplicación, consolidar responsabilidades, y cristalizar
la separación de concerns entre extracción, inferencia y auditoría.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CausalLink:
    """Causal link with embeddings and type information"""

    cause_id: str
    effect_id: str
    cause_emb: np.ndarray
    effect_emb: np.ndarray
    cause_type: str
    effect_type: str
    strength: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class ColombianMunicipalContext:
    """Colombian municipal PDM context for conditional independence proxy"""

    overall_pdm_embedding: Optional[np.ndarray] = None
    municipality_name: Optional[str] = None
    year: Optional[int] = None

    # Additional context that could affect mechanisms
    institutional_capacity: Optional[float] = None
    budget_execution_rate: Optional[float] = None


@dataclass
class MechanismEvidence:
    """Evidence for a specific mechanism"""

    type: str  # 'técnico', 'político', 'financiero', 'administrativo', 'mixto'
    verb_sequence: List[str]
    entity: Optional[str] = None
    activity: Optional[str] = None
    budget: Optional[float] = None
    timeline: Optional[str] = None
    confidence: float = 0.0


@dataclass
class EvidenceChunk:
    """Individual evidence chunk with similarity score"""

    chunk_id: str
    text: str
    cosine_similarity: float
    source_page: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    source_chunk_sha256: Optional[str] = None

    def __post_init__(self):
        """Compute SHA256 hash of text content if not provided"""
        if self.source_chunk_sha256 is None and self.text:
            self.source_chunk_sha256 = hashlib.sha256(
                self.text.encode("utf-8")
            ).hexdigest()


@dataclass
class MechanismPrior:
    """Bayesian prior for mechanism inference"""

    alpha: float
    beta: float
    rationale: str
    context_adjusted_strength: float = 0.0
    type_coherence_penalty: float = 0.0
    historical_influence: float = 0.0

    def __post_init__(self):
        """Validate parameters"""
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                f"Alpha and Beta must be positive, got alpha={self.alpha}, beta={self.beta}"
            )


@dataclass
class SamplingConfig:
    """Configuration for MCMC sampling"""

    draws: int = 1000
    chains: int = 4
    sigmoid_tau: float = 1.0  # Temperature parameter for sigmoid calibration
    convergence_threshold: float = 1.1  # Gelman-Rubin R-hat threshold
    timeout_seconds: int = 60


@dataclass
class PosteriorDistribution:
    """Posterior distribution result from Bayesian update"""

    posterior_mean: float
    posterior_std: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    convergence_diagnostic: bool = True
    samples: Optional[np.ndarray] = None

    def get_hdi(self, credible_mass: float = 0.95) -> Tuple[float, float]:
        """Get Highest Density Interval"""
        if self.samples is not None and len(self.samples) > 0:
            sorted_samples = np.sort(self.samples)
            n = len(sorted_samples)
            interval_size = int(np.ceil(credible_mass * n))
            n_intervals = n - interval_size

            if n_intervals <= 0:
                return (sorted_samples[0], sorted_samples[-1])

            # Find interval with minimum width
            interval_widths = (
                sorted_samples[interval_size:] - sorted_samples[:n_intervals]
            )
            min_idx = np.argmin(interval_widths)

            return (sorted_samples[min_idx], sorted_samples[min_idx + interval_size])

        return self.confidence_interval

    @property
    def credible_interval_95(self) -> Tuple[float, float]:
        """Get 95% credible interval (HDI)"""
        return self.get_hdi(credible_mass=0.95)


@dataclass
class NecessityTestResult:
    """Result from necessity hoop test"""

    passed: bool
    missing: List[str]
    severity: Optional[str] = None
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "passed": self.passed,
            "missing": self.missing,
            "severity": self.severity,
            "remediation": self.remediation,
        }


@dataclass
class InferenceExplainabilityPayload:
    """
    IoR (Inference of Relations) Per-Link Explainability Payload.

    Ensures transparent, traceable causal inferences per SOTA process-tracing
    mandates (Beach & Pedersen 2019 on within-case transparency).

    Audit Point 3.1: Full Traceability Payload
    - Every link generates JSON payload with posterior, necessity result,
      snippets, sha256
    - XAI-compliant payloads (Doshi-Velez 2017)
    - Enables replicable MMR inferences, reducing opacity in Bayesian models
      (Gelman 2013)

    Audit Point 3.2: Credibility Reporting
    - QualityScore includes credible_interval_95 and Bayesian metrics
    - Reflects epistemic uncertainty per Humphreys & Jacobs (2015)
    - Avoids point-estimate biases in causal audits
    """

    # Link identification
    cause_id: str
    effect_id: str
    link_type: str  # e.g., "producto→resultado", "resultado→impacto"

    # Bayesian inference results
    posterior_mean: float
    posterior_std: float
    credible_interval_95: Tuple[float, float]
    convergence_diagnostic: bool

    # Necessity test results
    necessity_passed: bool
    necessity_missing: List[str]
    necessity_severity: Optional[str] = None

    # Evidence snippets for transparency
    evidence_snippets: List[Dict[str, Any]] = field(default_factory=list)

    # Source traceability (SHA256 hashes)
    source_chunk_hashes: List[str] = field(default_factory=list)

    # Quality metrics
    evidence_strength: float = 0.0
    epistemic_uncertainty: float = 0.0

    # Timestamp and metadata
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Implements Audit Point 3.1: Full Traceability Payload
        All fields present and matching source hashes for replicability.
        """
        return {
            # Link identification
            "cause_id": self.cause_id,
            "effect_id": self.effect_id,
            "link_type": self.link_type,
            # Bayesian metrics (Audit Point 3.2)
            "posterior_mean": float(self.posterior_mean),
            "posterior_std": float(self.posterior_std),
            "credible_interval_95": [
                float(self.credible_interval_95[0]),
                float(self.credible_interval_95[1]),
            ],
            "convergence_diagnostic": self.convergence_diagnostic,
            # Necessity results
            "necessity_test": {
                "passed": self.necessity_passed,
                "missing_components": self.necessity_missing,
                "severity": self.necessity_severity,
            },
            # Evidence transparency (XAI-compliant)
            "evidence_snippets": self.evidence_snippets,
            "source_chunk_hashes": self.source_chunk_hashes,
            # Quality and uncertainty metrics
            "quality_score": {
                "evidence_strength": float(self.evidence_strength),
                "epistemic_uncertainty": float(self.epistemic_uncertainty),
                "credible_interval_width": float(
                    self.credible_interval_95[1] - self.credible_interval_95[0]
                ),
            },
            # Metadata
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def compute_quality_score(self) -> Dict[str, Any]:
        """
        Compute comprehensive quality score with Bayesian metrics.

        Implements Audit Point 3.2: Credibility Reporting
        Includes credible_interval_95 and epistemic uncertainty.
        """
        # Evidence strength from posterior mean
        self.evidence_strength = self.posterior_mean

        # Epistemic uncertainty from posterior std and interval width
        interval_width = self.credible_interval_95[1] - self.credible_interval_95[0]
        self.epistemic_uncertainty = min(1.0, self.posterior_std + interval_width / 2.0)

        # Overall quality combining evidence and uncertainty
        quality_score = self.evidence_strength * (1.0 - self.epistemic_uncertainty)

        return {
            "evidence_strength": float(self.evidence_strength),
            "epistemic_uncertainty": float(self.epistemic_uncertainty),
            "quality_score": float(quality_score),
            "credible_interval_95": [
                float(self.credible_interval_95[0]),
                float(self.credible_interval_95[1]),
            ],
            "credible_interval_width": float(interval_width),
            "necessity_passed": self.necessity_passed,
        }


class DocumentEvidence:
    """Document evidence for necessity testing"""

    def __init__(self):
        self.entities: Dict[str, List[str]] = {}
        self.activities: Dict[Tuple[str, str], List[str]] = {}
        self.budgets: Dict[str, float] = {}
        self.timelines: Dict[str, str] = {}

    def has_entity(self, cause_id: str) -> bool:
        """Check if entity is documented for cause"""
        return cause_id in self.entities and len(self.entities[cause_id]) > 0

    def has_activity_sequence(self, cause_id: str, effect_id: str) -> bool:
        """Check if activity sequence is documented"""
        return (cause_id, effect_id) in self.activities and len(
            self.activities[(cause_id, effect_id)]
        ) > 0

    def has_budget_trace(self, cause_id: str) -> bool:
        """Check if budget is allocated"""
        return cause_id in self.budgets and self.budgets[cause_id] > 0

    def has_timeline(self, cause_id: str) -> bool:
        """Check if timeline is specified"""
        return cause_id in self.timelines and len(self.timelines[cause_id]) > 0


# ============================================================================
# AGUJA I: Bayesian Prior Builder
# ============================================================================


class BayesianPriorBuilder:
    """
    Construye priors adaptativos basados en evidencia estructural.

    Implementa AGUJA I: Adaptive Prior con:
    - Semantic distance (cause-effect embedding similarity)
    - Hierarchical type transition (producto→resultado→impacto)
    - Mechanism type coherence (técnico/político/financiero)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}

        # Prior history for mechanism types (can be loaded from file)
        self.prior_history: Dict[str, List[Tuple[float, float]]] = {}

        # Structural penalty factor (Governance Trigger 3)
        self.structural_penalty_factor: float = 1.0
        self.structural_violations: List[Tuple[str, str]] = []

        # Type transition priors (hierarchical structure)
        self.type_transitions = {
            ("producto", "resultado"): 0.8,
            ("producto", "producto"): 0.6,
            ("resultado", "impacto"): 0.7,
            ("producto", "impacto"): 0.4,
            ("resultado", "resultado"): 0.5,
        }

        # Mechanism type verb signatures
        self.mechanism_type_verbs = {
            "técnico": [
                "implementar",
                "diseñar",
                "construir",
                "desarrollar",
                "ejecutar",
            ],
            "político": ["concertar", "negociar", "aprobar", "promulgar", "acordar"],
            "financiero": ["asignar", "transferir", "ejecutar", "auditar", "reportar"],
            "administrativo": [
                "planificar",
                "coordinar",
                "gestionar",
                "supervisar",
                "controlar",
            ],
            "mixto": ["articular", "integrar", "coordinar", "colaborar"],
        }

    def build_mechanism_prior(
        self,
        link: CausalLink,
        mechanism_evidence: MechanismEvidence,
        context: ColombianMunicipalContext,
    ) -> MechanismPrior:
        """
        Implementa AGUJA I: Adaptive Prior con:
        - Semantic distance (cause-effect embedding similarity)
        - Hierarchical type transition (producto→resultado→impacto)
        - Mechanism type coherence (técnico/político/financiero)

        Front B.3: Conditional Independence Proxy
        Front C.2: Mechanism Type Validation
        """
        # Front B.3: Conditional Independence Proxy
        context_adjusted_strength = self._apply_independence_proxy(
            link.cause_emb, link.effect_emb, context.overall_pdm_embedding
        )

        # Front C.2: Mechanism Type Validation
        type_penalty = self._validate_mechanism_type_coherence(
            mechanism_evidence.verb_sequence, link.cause_type, link.effect_type
        )

        # Calculate semantic distance
        semantic_distance = self._calculate_semantic_distance(
            link.cause_emb, link.effect_emb
        )

        # Get hierarchical type transition prior
        type_transition_prior = self._get_type_transition_prior(
            link.cause_type, link.effect_type
        )

        # Compute beta parameters
        alpha, beta = self._compute_beta_params(
            base_strength=context_adjusted_strength,
            type_coherence=type_penalty,
            semantic_distance=semantic_distance,
            type_transition=type_transition_prior,
            historical_priors=self.prior_history.get(mechanism_evidence.type, []),
        )

        rationale = (
            f"Prior based on: context_strength={context_adjusted_strength:.3f}, "
            f"type_coherence={type_penalty:.3f}, semantic_dist={semantic_distance:.3f}, "
            f"type_transition={type_transition_prior:.3f}"
        )

        return MechanismPrior(
            alpha=alpha,
            beta=beta,
            rationale=rationale,
            context_adjusted_strength=context_adjusted_strength,
            type_coherence_penalty=type_penalty,
            historical_influence=len(
                self.prior_history.get(mechanism_evidence.type, [])
            )
            / 100.0,
        )

    def _apply_independence_proxy(
        self,
        cause_emb: np.ndarray,
        effect_emb: np.ndarray,
        context_emb: Optional[np.ndarray],
    ) -> float:
        """
        Front B.3: Conditional Independence Proxy

        Adjusts link strength based on conditional independence with respect to
        overall PDM context. If cause and effect are conditionally independent
        given the context, the link strength should be reduced.
        """
        if context_emb is None:
            # No context available, return raw similarity
            return 1.0 - cosine(cause_emb, effect_emb)

        # Calculate partial correlation approximation
        # P(effect|cause, context) vs P(effect|context)

        cause_effect_sim = 1.0 - cosine(cause_emb, effect_emb)
        cause_context_sim = 1.0 - cosine(cause_emb, context_emb)
        effect_context_sim = 1.0 - cosine(effect_emb, context_emb)

        # If cause and effect both correlate strongly with context,
        # they might be conditionally independent
        if cause_context_sim > 0.7 and effect_context_sim > 0.7:
            # Reduce strength by correlation with context
            adjustment = 1.0 - (cause_context_sim * effect_context_sim)
            return cause_effect_sim * adjustment

        return cause_effect_sim

    def _validate_mechanism_type_coherence(
        self, verb_sequence: List[str], cause_type: str, effect_type: str
    ) -> float:
        """
        Front C.2: Mechanism Type Validation

        Validates that mechanism type is coherent with the types of
        cause and effect nodes and the verb sequence used.

        Returns a coherence score [0, 1] where 1 is fully coherent.
        """
        if not verb_sequence:
            return 0.5  # Neutral when no verbs

        # Score each verb against mechanism types
        type_scores = {mech_type: 0.0 for mech_type in self.mechanism_type_verbs}

        for verb in verb_sequence:
            verb_lower = verb.lower()
            for mech_type, typical_verbs in self.mechanism_type_verbs.items():
                if verb_lower in typical_verbs:
                    type_scores[mech_type] += 1.0

        # Normalize by verb count
        if verb_sequence:
            for mech_type in type_scores:
                type_scores[mech_type] /= len(verb_sequence)

        # Expected mechanism types based on cause/effect types
        expected_types = self._get_expected_mechanism_types(cause_type, effect_type)

        # Calculate coherence as max score among expected types
        coherence = max(
            [type_scores.get(et, 0.0) for et in expected_types], default=0.5
        )

        return coherence

    def _get_expected_mechanism_types(
        self, cause_type: str, effect_type: str
    ) -> List[str]:
        """Get expected mechanism types for cause-effect pair"""
        # Simplified heuristic - can be made more sophisticated
        if cause_type == "producto":
            if effect_type == "resultado":
                return ["técnico", "administrativo"]
            elif effect_type == "impacto":
                return ["político", "mixto"]
        elif cause_type == "resultado":
            if effect_type == "impacto":
                return ["político", "mixto"]

        return ["administrativo", "mixto"]  # Default

    def _calculate_semantic_distance(
        self, cause_emb: np.ndarray, effect_emb: np.ndarray
    ) -> float:
        """Calculate semantic distance (1 - cosine similarity)"""
        return cosine(cause_emb, effect_emb)

    def _get_type_transition_prior(self, cause_type: str, effect_type: str) -> float:
        """Get hierarchical type transition prior"""
        return self.type_transitions.get((cause_type, effect_type), 0.5)

    def _compute_beta_params(
        self,
        base_strength: float,
        type_coherence: float,
        semantic_distance: float,
        type_transition: float,
        historical_priors: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        Compute Beta distribution parameters (alpha, beta) from evidence.

        Uses a combination of:
        - Current evidence (base_strength, coherence, etc.)
        - Historical priors (if available)
        """
        # Combine evidence into overall strength estimate
        evidence_strength = (
            base_strength * 0.4
            + type_coherence * 0.3
            + (1.0 - semantic_distance) * 0.2
            + type_transition * 0.1
        )

        # Clamp to [0, 1]
        evidence_strength = max(0.0, min(1.0, evidence_strength))

        # Determine prior strength (equivalent sample size)
        if historical_priors:
            # Use average of historical priors
            avg_alpha = np.mean([p[0] for p in historical_priors])
            avg_beta = np.mean([p[1] for p in historical_priors])
            prior_strength = avg_alpha + avg_beta
        else:
            # Default prior strength
            prior_strength = 4.0  # Equivalent to 4 observations

        # Convert evidence strength to alpha/beta
        # For Beta distribution: mean = alpha / (alpha + beta)
        # So: alpha = mean * (alpha + beta)
        #     beta = (1 - mean) * (alpha + beta)

        alpha = evidence_strength * prior_strength
        beta = (1.0 - evidence_strength) * prior_strength

        # Ensure minimum values
        alpha = max(0.5, alpha)
        beta = max(0.5, beta)
        
        # Apply structural penalty if active (Governance Trigger 3)
        if self.structural_penalty_factor < 1.0:
            alpha, beta = self._apply_penalty_to_prior(alpha, beta)
            self.logger.debug(
                "Applied structural penalty: alpha=%.3f, beta=%.3f (factor=%.3f)",
                alpha, beta, self.structural_penalty_factor
            )

        return alpha, beta
    
    def apply_structural_penalty(
        self, 
        penalty_factor: float, 
        violations: List[Tuple[str, str]]
    ):
        """
        Apply structural penalty to prior builder (Governance Trigger 3).
        
        Called by AxiomaticValidator._apply_structural_penalty to propagate
        structural violation penalties to Bayesian posteriors.
        
        Args:
            penalty_factor: Multiplier in [0, 1] to scale priors
            violations: List of (source, target) edge violations
        """
        self.structural_penalty_factor = penalty_factor
        self.structural_violations = violations
        
        self.logger.info(
            "Applied structural penalty factor: %.3f for %d violations",
            penalty_factor, len(violations)
        )
    
    def _apply_penalty_to_prior(
        self,
        alpha: float,
        beta: float
    ) -> Tuple[float, float]:
        """
        Apply structural penalty to computed Beta parameters.
        
        Reduces alpha (success parameter) and increases beta (failure parameter)
        proportionally to penalty factor.
        
        Args:
            alpha: Original alpha parameter
            beta: Original beta parameter
            
        Returns:
            (penalized_alpha, penalized_beta)
        """
        if self.structural_penalty_factor >= 1.0:
            return alpha, beta
        
        # Penalty reduces success probability
        # New mean should be: original_mean * penalty_factor
        # For Beta: mean = alpha / (alpha + beta)
        
        original_mean = alpha / (alpha + beta)
        penalized_mean = original_mean * self.structural_penalty_factor
        
        # Keep total strength constant
        total_strength = alpha + beta
        
        # Recalculate alpha and beta
        penalized_alpha = penalized_mean * total_strength
        penalized_beta = (1.0 - penalized_mean) * total_strength
        
        # Ensure minimum values
        penalized_alpha = max(0.5, penalized_alpha)
        penalized_beta = max(0.5, penalized_beta)
        
        return penalized_alpha, penalized_beta


# ============================================================================
# AGUJA II: Bayesian Sampling Engine
# ============================================================================


class BayesianSamplingEngine:
    """
    Ejecuta MCMC sampling con reproducibilidad garantizada.

    AGUJA II: Bayesian Update con:
    - Calibrated likelihood (Front B.2)
    - Convergence diagnostics (Gelman-Rubin)
    - Confidence interval extraction

    NOTE: This is a simplified implementation using conjugate Beta-Binomial
    instead of full PyMC MCMC, as PyMC is not available in the environment.
    """

    def __init__(self, seed: int = 42):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_rng_complete(seed)

        # Metrics for observability
        self.metrics = {
            "posterior.nonconvergent_count": 0,
            "posterior.sampling_errors": 0,
        }

    def _initialize_rng_complete(self, seed: int):
        """Initialize RNG for reproducibility"""
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

    def sample_mechanism_posterior(
        self,
        prior: MechanismPrior,
        evidence: List[EvidenceChunk],
        config: SamplingConfig,
    ) -> PosteriorDistribution:
        """
        AGUJA II: Bayesian Update con:
        - Calibrated likelihood (Front B.2)
        - Convergence diagnostics (Gelman-Rubin)
        - Confidence interval extraction

        Simplified conjugate Beta-Binomial update instead of full MCMC.
        """
        if not evidence:
            # No evidence, return prior as posterior
            return self._prior_as_posterior(prior, config)

        # Calibrated likelihood using sigmoid transformation
        total_likelihood = 0.0
        evidence_count = 0

        for ev in evidence:
            likelihood = self._similarity_to_probability(
                ev.cosine_similarity, tau=config.sigmoid_tau
            )
            total_likelihood += likelihood
            evidence_count += 1

        # Beta-Binomial conjugate update
        # Prior: Beta(alpha, beta)
        # Likelihood: Binomial with success probability from evidence
        # Posterior: Beta(alpha + successes, beta + failures)

        # Treat high similarity as "success"
        successes = sum(
            1
            for ev in evidence
            if self._similarity_to_probability(ev.cosine_similarity, config.sigmoid_tau)
            > 0.5
        )
        failures = evidence_count - successes

        posterior_alpha = prior.alpha + successes
        posterior_beta = prior.beta + failures

        # Sample from posterior Beta distribution
        samples = self.rng.beta(posterior_alpha, posterior_beta, size=config.draws)

        # Calculate statistics
        posterior_mean = samples.mean()
        posterior_std = samples.std()

        # Convergence check (simplified - in real MCMC would use Gelman-Rubin)
        convergence_ok = self._check_convergence_simple(samples, config)

        if not convergence_ok:
            self.metrics["posterior.nonconvergent_count"] += 1
            self.logger.warning("Posterior sampling did not converge")

        # Extract HDI
        confidence_interval = self._extract_hdi(samples, 0.95)

        return PosteriorDistribution(
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            confidence_interval=confidence_interval,
            convergence_diagnostic=convergence_ok,
            samples=samples,
        )

    def _prior_as_posterior(
        self, prior: MechanismPrior, config: SamplingConfig
    ) -> PosteriorDistribution:
        """Convert prior to posterior when no evidence"""
        samples = self.rng.beta(prior.alpha, prior.beta, size=config.draws)

        return PosteriorDistribution(
            posterior_mean=samples.mean(),
            posterior_std=samples.std(),
            confidence_interval=self._extract_hdi(samples, 0.95),
            convergence_diagnostic=True,
            samples=samples,
        )

    def _similarity_to_probability(
        self, cosine_similarity: float, tau: float = 1.0
    ) -> float:
        """
        Front B.2: Calibrated likelihood

        Convert cosine similarity to probability using sigmoid with temperature.
        tau controls the steepness of the conversion.
        """
        # Sigmoid: p = 1 / (1 + exp(-x/tau))
        # Map similarity [0, 1] to [-5, 5] for sigmoid
        x = (cosine_similarity - 0.5) * 10
        prob = 1.0 / (1.0 + np.exp(-x / tau))
        return prob

    def _check_convergence_simple(
        self, samples: np.ndarray, config: SamplingConfig
    ) -> bool:
        """
        Simplified convergence check.

        In full MCMC, would use Gelman-Rubin diagnostic across chains.
        Here we just check if variance is reasonable.
        """
        if len(samples) < 10:
            return False

        # Check if variance is not too high (samples should cluster)
        variance = np.var(samples)

        # For Beta distribution on [0, 1], variance > 0.1 is quite high
        return variance < 0.1

    def _extract_hdi(
        self, samples: np.ndarray, credible_mass: float = 0.95
    ) -> Tuple[float, float]:
        """Extract Highest Density Interval"""
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)
        interval_size = int(np.ceil(credible_mass * n))
        n_intervals = n - interval_size

        if n_intervals <= 0:
            return (sorted_samples[0], sorted_samples[-1])

        # Find interval with minimum width
        interval_widths = sorted_samples[interval_size:] - sorted_samples[:n_intervals]
        min_idx = np.argmin(interval_widths)

        return (sorted_samples[min_idx], sorted_samples[min_idx + interval_size])

    def create_explainability_payload(
        self,
        link: CausalLink,
        posterior: PosteriorDistribution,
        evidence: List[EvidenceChunk],
        necessity_result: Optional[NecessityTestResult] = None,
        timestamp: Optional[str] = None,
    ) -> InferenceExplainabilityPayload:
        """
        Create IoR (Inference of Relations) explainability payload for a causal link.

        Implements Audit Point 3.1 and 3.2:
        - Full traceability with posterior, necessity results, snippets, SHA256
        - XAI-compliant payload structure (Doshi-Velez 2017)
        - Credibility reporting with Bayesian metrics and uncertainty

        Args:
            link: Causal link being analyzed
            posterior: Posterior distribution from Bayesian update
            evidence: List of evidence chunks used in inference
            necessity_result: Result from necessity test (optional)
            timestamp: ISO format timestamp (optional)

        Returns:
            InferenceExplainabilityPayload with full traceability
        """
        from datetime import datetime

        # Extract evidence snippets (top 5 by similarity)
        evidence_snippets = []
        sorted_evidence = sorted(
            evidence, key=lambda e: e.cosine_similarity, reverse=True
        )
        for ev in sorted_evidence[:5]:
            snippet = {
                "chunk_id": ev.chunk_id,
                "text": ev.text[:200] + "..." if len(ev.text) > 200 else ev.text,
                "cosine_similarity": float(ev.cosine_similarity),
                "source_page": ev.source_page,
                "sha256": ev.source_chunk_sha256,
            }
            evidence_snippets.append(snippet)

        # Collect all source chunk hashes
        source_hashes = [
            ev.source_chunk_sha256
            for ev in evidence
            if ev.source_chunk_sha256 is not None
        ]

        # Create necessity result if not provided
        if necessity_result is None:
            necessity_result = NecessityTestResult(
                passed=True, missing=[], severity=None, remediation=None
            )

        # Construct payload
        payload = InferenceExplainabilityPayload(
            cause_id=link.cause_id,
            effect_id=link.effect_id,
            link_type=f"{link.cause_type}→{link.effect_type}",
            posterior_mean=posterior.posterior_mean,
            posterior_std=posterior.posterior_std,
            credible_interval_95=posterior.credible_interval_95,
            convergence_diagnostic=posterior.convergence_diagnostic,
            necessity_passed=necessity_result.passed,
            necessity_missing=necessity_result.missing,
            necessity_severity=necessity_result.severity,
            evidence_snippets=evidence_snippets,
            source_chunk_hashes=source_hashes,
            timestamp=timestamp or datetime.utcnow().isoformat() + "Z",
        )

        # Compute quality score
        payload.compute_quality_score()

        return payload


# ============================================================================
# AGUJA III: Necessity & Sufficiency Tester
# ============================================================================


class NecessitySufficiencyTester:
    """
    Ejecuta Hoop Tests determinísticos (Front C.3).

    Tests necessity and sufficiency of causal mechanisms based on
    documented evidence.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def test_necessity(
        self, link: CausalLink, document_evidence: DocumentEvidence
    ) -> NecessityTestResult:
        """
        Hoop Test: ¿Existen los componentes necesarios documentados?
        - Entity (responsable)
        - Activity (verbo de acción)
        - Budget (presupuesto asignado)
        - Timeline (cronograma)

        Front C.3: Deterministic failure on missing components
        """
        missing_components = []

        if not document_evidence.has_entity(link.cause_id):
            missing_components.append("entity")

        if not document_evidence.has_activity_sequence(link.cause_id, link.effect_id):
            missing_components.append("activity")

        if not document_evidence.has_budget_trace(link.cause_id):
            missing_components.append("budget")

        if not document_evidence.has_timeline(link.cause_id):
            missing_components.append("timeline")

        # Deterministic failure (Front C.3)
        if missing_components:
            return NecessityTestResult(
                passed=False,
                missing=missing_components,
                severity="critical" if len(missing_components) >= 3 else "moderate",
                remediation=self._generate_hoop_failure_text(missing_components, link),
            )

        return NecessityTestResult(
            passed=True, missing=[], severity=None, remediation=None
        )

    def test_sufficiency(
        self,
        link: CausalLink,
        document_evidence: DocumentEvidence,
        mechanism_evidence: MechanismEvidence,
    ) -> NecessityTestResult:
        """
        Sufficiency test: Are the documented components sufficient to
        produce the claimed effect?

        Checks:
        - Budget adequacy
        - Entity capacity
        - Activity completeness
        """
        missing_components = []

        # Check budget adequacy
        if mechanism_evidence.budget is not None:
            if mechanism_evidence.budget < 1000000:  # Threshold (1M COP)
                missing_components.append("adequate_budget")

        # Check entity capacity (simplified - would need more context)
        if mechanism_evidence.entity and len(mechanism_evidence.entity) < 10:
            # Very short entity name might indicate placeholder
            missing_components.append("qualified_entity")

        # Check activity completeness (need at least 3 activities)
        if len(mechanism_evidence.verb_sequence) < 3:
            missing_components.append("complete_activity_sequence")

        if missing_components:
            return NecessityTestResult(
                passed=False,
                missing=missing_components,
                severity="moderate",
                remediation=self._generate_sufficiency_failure_text(
                    missing_components, link
                ),
            )

        return NecessityTestResult(
            passed=True, missing=[], severity=None, remediation=None
        )

    def _generate_hoop_failure_text(
        self, missing_components: List[str], link: CausalLink
    ) -> str:
        """Generate remediation text for hoop test failure"""
        component_descriptions = {
            "entity": "entidad responsable",
            "activity": "secuencia de actividades",
            "budget": "presupuesto asignado",
            "timeline": "cronograma de ejecución",
        }

        missing_desc = ", ".join(
            [component_descriptions.get(c, c) for c in missing_components]
        )

        return (
            f"El mecanismo causal entre {link.cause_id} y {link.effect_id} "
            f"falla el Hoop Test. Componentes faltantes: {missing_desc}. "
            f"Se requiere documentar estos componentes necesarios para validar "
            f"la cadena causal."
        )

    def _generate_sufficiency_failure_text(
        self, missing_components: List[str], link: CausalLink
    ) -> str:
        """Generate remediation text for sufficiency test failure"""
        component_descriptions = {
            "adequate_budget": "presupuesto adecuado",
            "qualified_entity": "entidad calificada",
            "complete_activity_sequence": "secuencia completa de actividades",
        }

        missing_desc = ", ".join(
            [component_descriptions.get(c, c) for c in missing_components]
        )

        return (
            f"El mecanismo causal entre {link.cause_id} y {link.effect_id} "
            f"podría no ser suficiente. Componentes insuficientes: {missing_desc}. "
            f"Se recomienda fortalecer estos aspectos para garantizar el logro "
            f"del efecto esperado."
        )
