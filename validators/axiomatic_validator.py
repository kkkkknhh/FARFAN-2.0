#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Axiomatic Validator - Unified Validation System
================================================

Unified validator for Phase III-B and III-C.
Ensures that the causal graph meets both structural AND semantic axioms.

This module provides a single point of validation with explicit execution order
and automatic governance triggers.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

# Conditional imports for runtime vs testing
if TYPE_CHECKING:
    import networkx as nx

    from contradiction_deteccion import PolicyContradictionDetectorV2, PolicyDimension
    from dnp_integration import ResultadoValidacionDNP, ValidadorDNP
    from teoria_cambio import TeoriaCambio, ValidacionResultado

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class SemanticChunk:
    """Represents a semantic chunk of text from the PDM"""

    text: str
    dimension: str = "ESTRATEGICO"
    position: Tuple[int, int] = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedTable:
    """Represents a financial/data table extracted from the PDM"""

    title: str
    headers: List[str]
    rows: List[List[Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDMOntology:
    """
    Encapsulates PDM ontology data.
    Contains canonical chain and other ontological structures.
    """

    canonical_chain: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    policy_areas: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize with default canonical chain if not provided"""
        if not self.canonical_chain:
            self.canonical_chain = [
                "INSUMOS",
                "PROCESOS",
                "PRODUCTOS",
                "RESULTADOS",
                "CAUSALIDAD",
            ]
        if not self.dimensions:
            self.dimensions = [
                "D1_DIAGNOSTICO",
                "D2_DISENO",
                "D3_PRODUCTOS",
                "D4_RESULTADOS",
                "D5_IMPACTOS",
                "D6_TEORIA_CAMBIO",
            ]


@dataclass
class ValidationConfig:
    """Configuration for axiomatic validation"""

    dnp_lexicon_version: str = "2025"
    es_municipio_pdet: bool = False
    contradiction_threshold: float = 0.05
    enable_structural_penalty: bool = True
    enable_human_gating: bool = True


class ValidationDimension(Enum):
    """Validation dimensions mapped to canonical notation"""

    D1_DIAGNOSTICO = "D1"
    D2_DISENO = "D2"
    D3_PRODUCTOS = "D3"
    D4_RESULTADOS = "D4"
    D5_IMPACTOS = "D5"
    D6_TEORIA_CAMBIO = "D6"


class ValidationSeverity(Enum):
    """Severity levels for validation failures"""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ValidationFailure:
    """Represents a validation failure"""

    dimension: str
    question: str
    severity: ValidationSeverity
    evidence: Any
    impact: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AxiomaticValidationResult:
    """
    Comprehensive result of axiomatic validation.

    Contains results from:
    1. Structural validation (Theory of Change) → D6-Q1/Q2
    2. Semantic validation (Contradictions) → D2-Q5, D6-Q3
    3. Regulatory validation (DNP Compliance) → D1-Q5, D4-Q5
    """

    # Overall validation status
    is_valid: bool = True

    # Structural validation (Phase III-B)
    structural_valid: bool = True
    violaciones_orden: List[Tuple[str, str]] = field(default_factory=list)
    categorias_faltantes: List[str] = field(default_factory=list)
    caminos_completos: List[List[str]] = field(default_factory=list)
    structural_penalty_factor: float = 1.0

    # Semantic validation (Phase III-C)
    contradiction_density: float = 0.0
    contradictions: List[Dict[str, Any]] = field(default_factory=list)

    # Regulatory validation (Phase III-D)
    regulatory_score: float = 0.0
    dnp_compliance: Optional[Any] = None  # ResultadoValidacionDNP when available
    bpin_indicators: List[str] = field(default_factory=list)

    # Score mappings to canonical notation
    score_mappings: Dict[str, float] = field(default_factory=dict)

    # Failures and recommendations
    failures: List[ValidationFailure] = field(default_factory=list)

    # Governance triggers
    requires_manual_review: bool = False
    hold_reason: Optional[str] = None

    # Metadata
    validation_timestamp: str = ""
    total_edges: int = 0
    total_nodes: int = 0

    def add_critical_failure(
        self,
        dimension: str,
        question: str,
        evidence: Any,
        impact: str,
        recommendations: Optional[List[str]] = None,
    ):
        """Add a critical failure to the results"""
        self.is_valid = False
        failure = ValidationFailure(
            dimension=dimension,
            question=question,
            severity=ValidationSeverity.CRITICAL,
            evidence=evidence,
            impact=impact,
            recommendations=recommendations or [],
        )
        self.failures.append(failure)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results"""
        return {
            "is_valid": self.is_valid,
            "structural_valid": self.structural_valid,
            "contradiction_density": self.contradiction_density,
            "regulatory_score": self.regulatory_score,
            "critical_failures": len(
                [f for f in self.failures if f.severity == ValidationSeverity.CRITICAL]
            ),
            "requires_manual_review": self.requires_manual_review,
            "hold_reason": self.hold_reason,
        }


# ============================================================================
# AXIOMATIC VALIDATOR
# ============================================================================


class AxiomaticValidator:
    """
    Validador unificado para Phase III-B y III-C.
    Garantiza que el grafo cumple axiomas estructurales Y semánticos.

    This validator provides a unified interface to:
    1. TeoriaCambio (Structural validation - D6-Q1/Q2)
    2. PolicyContradictionDetectorV2 (Semantic validation - D2-Q5, D6-Q3)
    3. ValidadorDNP (Regulatory validation - D1-Q5, D4-Q5)

    Benefits:
    - Single point of validation
    - Explicit execution order
    - Automatic governance triggers
    """

    def __init__(self, config: ValidationConfig, ontology: PDMOntology):
        """
        Initialize the axiomatic validator

        Args:
            config: ValidationConfig with settings
            ontology: PDMOntology with canonical structures
        """
        self.config = config
        self.ontology = ontology

        # Initialize component validators (lazy import to avoid circular dependencies)
        try:
            from contradiction_deteccion import PolicyContradictionDetectorV2
            from dnp_integration import ValidadorDNP
            from teoria_cambio import TeoriaCambio

            self.teoria_cambio = TeoriaCambio()
            self.contradiction_detector = PolicyContradictionDetectorV2()
            self.dnp_validator = ValidadorDNP(
                es_municipio_pdet=config.es_municipio_pdet
            )
        except ImportError as e:
            logger.warning("Could not import component validators: %s", e)
            self.teoria_cambio = None
            self.contradiction_detector = None
            self.dnp_validator = None

        logger.info("AxiomaticValidator initialized with config: %s", config)

    def validate_complete(
        self,
        causal_graph: "nx.DiGraph",
        semantic_chunks: List[SemanticChunk],
        financial_data: Optional[List[ExtractedTable]] = None,
        prior_builder: Optional[Any] = None,
    ) -> AxiomaticValidationResult:
        """
        Ejecuta validación completa en orden de severidad:
        1. STRUCTURAL validation (TeoriaCambio) → D6-Q1/Q2 scores
        2. SEMANTIC validation (PolicyContradictionDetectorV2) → D2-Q5, D6-Q3 scores
        3. REGULATORY validation (ValidadorDNP) → D1-Q5, D4-Q5 scores with BPIN

        Governance Triggers:
        - contradiction_density > 0.05 → manual review flag
        - D6 scores < 0.55 → block progression at quality gate
        - structural violations → penalty factors to Bayesian posteriors

        Args:
            causal_graph: The causal graph to validate
            semantic_chunks: List of semantic chunks from the PDM
            financial_data: Optional financial/data tables
            prior_builder: Optional BayesianPriorBuilder for applying penalties

        Returns:
            AxiomaticValidationResult with comprehensive validation results and score mappings
        """
        from datetime import datetime

        results = AxiomaticValidationResult()
        results.validation_timestamp = datetime.now().isoformat()
        results.total_edges = causal_graph.number_of_edges()
        results.total_nodes = causal_graph.number_of_nodes()

        logger.info(
            "Starting complete validation with %d nodes and %d edges",
            results.total_nodes,
            results.total_edges,
        )

        # ====================================================================
        # 1. STRUCTURAL VALIDATION: TeoriaCambio → D6-Q1/Q2
        # ====================================================================
        logger.info("Phase 1: STRUCTURAL validation (TeoriaCambio)")
        logger.info(
            "  → Mapping to canonical scores: D6-Q1 (completeness), D6-Q2 (causal order)"
        )

        structural = self._validate_structural(causal_graph)
        results.structural_valid = structural.es_valida
        results.violaciones_orden = structural.violaciones_orden
        # Handle categorias_faltantes which may be strings or CategoriaCausal enums
        results.categorias_faltantes = [
            cat.name if hasattr(cat, "name") else str(cat)
            for cat in structural.categorias_faltantes
        ]
        results.caminos_completos = structural.caminos_completos

        # Map to D6-Q1 (completeness) and D6-Q2 (causal order)
        d6_q1_score = self._calculate_d6_q1_score(structural)
        d6_q2_score = self._calculate_d6_q2_score(structural)

        logger.info("  D6-Q1 score (completeness): %.3f", d6_q1_score)
        logger.info("  D6-Q2 score (causal order): %.3f", d6_q2_score)

        # Store score mappings
        results.score_mappings = {"D6-Q1": d6_q1_score, "D6-Q2": d6_q2_score}

        # Governance Trigger 2: D6 scores below 0.55 block progression
        if d6_q1_score < 0.55 or d6_q2_score < 0.55:
            results.requires_manual_review = True
            results.hold_reason = "D6_SCORE_BELOW_THRESHOLD"
            logger.warning(
                "⚠ GOVERNANCE TRIGGER 2: D6 scores below 0.55 - blocking progression"
            )
            results.add_critical_failure(
                dimension="D6",
                question="Q1" if d6_q1_score < 0.55 else "Q2",
                evidence={"d6_q1": d6_q1_score, "d6_q2": d6_q2_score},
                impact="Theory of Change quality gate failed",
                recommendations=[
                    "Strengthen causal model completeness and ordering before proceeding"
                ],
            )

        if structural.violaciones_orden:
            logger.warning(
                "Found %d structural violations", len(structural.violaciones_orden)
            )
            results.add_critical_failure(
                dimension="D6",
                question="Q2",
                evidence=structural.violaciones_orden,
                impact="Saltos lógicos detectados en orden causal",
                recommendations=structural.sugerencias,
            )

            # Governance Trigger 3: Apply penalty factors to Bayesian posteriors
            if self.config.enable_structural_penalty:
                penalty_factor = self._apply_structural_penalty(
                    causal_graph, structural.violaciones_orden, prior_builder
                )
                results.structural_penalty_factor = penalty_factor
                logger.info(
                    "  Applied structural penalty factor: %.3f to Bayesian posteriors",
                    penalty_factor,
                )

        # ====================================================================
        # 2. SEMANTIC VALIDATION: PolicyContradictionDetectorV2 → D2-Q5, D6-Q3
        # ====================================================================
        logger.info("Phase 2: SEMANTIC validation (PolicyContradictionDetectorV2)")
        logger.info(
            "  → Mapping to canonical scores: D2-Q5 (design contradictions), D6-Q3 (causal coherence)"
        )

        contradictions = self._validate_semantic(causal_graph, semantic_chunks)
        results.contradictions = contradictions

        # Calculate contradiction density
        if results.total_edges > 0:
            results.contradiction_density = len(contradictions) / results.total_edges
        else:
            results.contradiction_density = 0.0

        logger.info(
            "  Contradiction density: %.4f (threshold: %.4f)",
            results.contradiction_density,
            self.config.contradiction_threshold,
        )

        # Map to D2-Q5 (design contradictions) and D6-Q3 (causal coherence)
        d2_q5_score = self._calculate_d2_q5_score(contradictions, semantic_chunks)
        d6_q3_score = self._calculate_d6_q3_score(
            results.contradiction_density, contradictions
        )

        logger.info("  D2-Q5 score (design coherence): %.3f", d2_q5_score)
        logger.info("  D6-Q3 score (causal coherence): %.3f", d6_q3_score)

        results.score_mappings["D2-Q5"] = d2_q5_score
        results.score_mappings["D6-Q3"] = d6_q3_score

        # Governance Trigger 1: contradiction_density > 0.05 flags for manual review
        if results.contradiction_density > self.config.contradiction_threshold:
            if self.config.enable_human_gating:
                results.requires_manual_review = True
                results.hold_reason = "HIGH_CONTRADICTION_DENSITY"
                logger.warning(
                    "⚠ GOVERNANCE TRIGGER 1: Contradiction density %.4f > %.4f - flagging for manual review",
                    results.contradiction_density,
                    self.config.contradiction_threshold,
                )
                results.add_critical_failure(
                    dimension="D2",
                    question="Q5",
                    evidence={
                        "density": results.contradiction_density,
                        "count": len(contradictions),
                    },
                    impact="High contradiction density detected",
                    recommendations=[
                        "Review policy contradictions before proceeding",
                        "Reconcile conflicting statements in design phase",
                    ],
                )

        # Governance Trigger 2: Check D6-Q3 threshold
        if d6_q3_score < 0.55:
            results.requires_manual_review = True
            results.hold_reason = "D6_SCORE_BELOW_THRESHOLD"
            logger.warning(
                "⚠ GOVERNANCE TRIGGER 2: D6-Q3 score %.3f < 0.55 - blocking progression",
                d6_q3_score,
            )

        # ====================================================================
        # 3. REGULATORY VALIDATION: ValidadorDNP → D1-Q5, D4-Q5 with BPIN
        # ====================================================================
        logger.info("Phase 3: REGULATORY validation (ValidadorDNP)")
        logger.info(
            "  → Mapping to canonical scores: D1-Q5 (DNP diagnostic compliance), D4-Q5 (results validation)"
        )

        dnp_results = self._validate_regulatory(semantic_chunks, financial_data)
        results.dnp_compliance = dnp_results
        results.regulatory_score = dnp_results.score_total if dnp_results else 0.0

        logger.info("  Regulatory compliance score: %.2f/100", results.regulatory_score)

        # Map to D1-Q5 (diagnostic compliance) and D4-Q5 (results validation)
        d1_q5_score = self._calculate_d1_q5_score(dnp_results)
        d4_q5_score = self._calculate_d4_q5_score(dnp_results)

        logger.info("  D1-Q5 score (DNP diagnostic): %.3f", d1_q5_score)
        logger.info("  D4-Q5 score (results validation): %.3f", d4_q5_score)

        results.score_mappings["D1-Q5"] = d1_q5_score
        results.score_mappings["D4-Q5"] = d4_q5_score

        # Integrate BPIN validation results
        if dnp_results and hasattr(dnp_results, "indicadores_mga_usados"):
            results.bpin_indicators = dnp_results.indicadores_mga_usados
            logger.info(
                "  BPIN validation: %d MGA indicators validated",
                len(dnp_results.indicadores_mga_usados),
            )

        # ====================================================================
        # FINAL VALIDATION STATUS
        # ====================================================================
        # Final validation status
        results.is_valid = (
            results.structural_valid
            and not results.requires_manual_review
            and results.regulatory_score >= 60.0
            and all(
                score >= 0.55
                for key, score in results.score_mappings.items()
                if "D6" in key
            )
        )

        logger.info("=" * 80)
        logger.info(
            "Validation complete. Overall status: %s",
            "VALID" if results.is_valid else "INVALID",
        )
        logger.info("  Structural: %s", "PASS" if results.structural_valid else "FAIL")
        logger.info("  Contradiction density: %.4f", results.contradiction_density)
        logger.info("  Regulatory score: %.2f/100", results.regulatory_score)
        logger.info("  Manual review required: %s", results.requires_manual_review)
        logger.info("=" * 80)

        return results

    def _validate_structural(self, causal_graph: "nx.DiGraph") -> "ValidacionResultado":
        """
        Execute structural validation using TeoriaCambio

        Args:
            causal_graph: The causal graph to validate

        Returns:
            ValidacionResultado from TeoriaCambio
        """
        try:
            if self.teoria_cambio is None:
                raise ImportError("TeoriaCambio not available")
            resultado = self.teoria_cambio.validacion_completa(causal_graph)
            logger.debug(
                "Structural validation: valid=%s, violations=%d",
                resultado.es_valida,
                len(resultado.violaciones_orden),
            )
            return resultado
        except Exception as e:
            logger.error("Error in structural validation: %s", e)
            # Return empty result on error
            from teoria_cambio import ValidacionResultado

            return ValidacionResultado(es_valida=False)

    def _validate_semantic(
        self, causal_graph: "nx.DiGraph", semantic_chunks: List[SemanticChunk]
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic validation using PolicyContradictionDetectorV2

        Args:
            causal_graph: The causal graph
            semantic_chunks: Semantic chunks from the PDM

        Returns:
            List of detected contradictions
        """
        contradictions = []

        try:
            if self.contradiction_detector is None:
                raise ImportError("PolicyContradictionDetectorV2 not available")

            # Import PolicyDimension locally
            from contradiction_deteccion import PolicyDimension

            # Process each semantic chunk
            for chunk in semantic_chunks:
                # Map dimension string to PolicyDimension enum
                dimension = self._map_dimension(chunk.dimension)

                # Detect contradictions in this chunk
                result = self.contradiction_detector.detect(
                    text=chunk.text, plan_name="PDM", dimension=dimension
                )

                # Extract contradictions from result
                if "contradictions" in result:
                    contradictions.extend(result["contradictions"])

            logger.debug(
                "Semantic validation found %d contradictions", len(contradictions)
            )

        except Exception as e:
            logger.error("Error in semantic validation: %s", e)

        return contradictions

    def _validate_regulatory(
        self,
        semantic_chunks: List[SemanticChunk],
        financial_data: Optional[List[ExtractedTable]] = None,
    ) -> Optional["ResultadoValidacionDNP"]:
        """
        Execute regulatory validation using ValidadorDNP

        Args:
            semantic_chunks: Semantic chunks from the PDM
            financial_data: Optional financial data tables

        Returns:
            ResultadoValidacionDNP or None if validation fails
        """
        try:
            if self.dnp_validator is None:
                raise ImportError("ValidadorDNP not available")

            # Import ResultadoValidacionDNP locally
            from dnp_integration import ResultadoValidacionDNP

            # For now, we'll create a simple aggregated validation
            # In a real implementation, this would be more sophisticated
            # Combine all semantic chunks
            combined_text = " ".join([chunk.text for chunk in semantic_chunks])

            # Extract basic project information
            # This is a simplified version - real implementation would be more complex
            resultado = ResultadoValidacionDNP(
                es_municipio_pdet=self.config.es_municipio_pdet
            )

            # Run a basic validation
            # In practice, you'd validate individual projects/programs
            # For now, we'll return a basic result
            resultado.score_total = 70.0  # Placeholder

            logger.debug("Regulatory validation score: %.2f", resultado.score_total)

            return resultado

        except Exception as e:
            logger.error("Error in regulatory validation: %s", e)
            return None

    def _apply_structural_penalty(
        self,
        causal_graph: "nx.DiGraph",
        violations: List[Tuple[str, str]],
        prior_builder: Optional[Any] = None,
    ) -> float:
        """
        Apply structural penalty for violations (Governance Trigger 3)

        Penalty factor applied to Bayesian posteriors via prior_builder:
        - 1 violation: 0.9x penalty
        - 2-5 violations: 0.8x penalty
        - 6-10 violations: 0.6x penalty
        - >10 violations: 0.4x penalty

        Also marks edges as suspect and reduces confidence scores.

        Args:
            causal_graph: The causal graph
            violations: List of edge violations (source, target)
            prior_builder: BayesianPriorBuilder instance to apply penalties

        Returns:
            penalty_factor: Multiplier applied to Bayesian posteriors
        """
        num_violations = len(violations)
        logger.info("Applying structural penalty for %d violations", num_violations)

        # Calculate penalty factor based on violation count
        if num_violations == 0:
            penalty_factor = 1.0
        elif num_violations == 1:
            penalty_factor = 0.9
        elif num_violations <= 5:
            penalty_factor = 0.8
        elif num_violations <= 10:
            penalty_factor = 0.6
        else:
            penalty_factor = 0.4

        logger.info(
            "  Penalty factor: %.2f (based on %d violations)",
            penalty_factor,
            num_violations,
        )

        # Apply penalty to prior_builder if provided
        if prior_builder is not None:
            if hasattr(prior_builder, "apply_structural_penalty"):
                prior_builder.apply_structural_penalty(penalty_factor, violations)
                logger.info("  Applied penalty factor to BayesianPriorBuilder")
            elif hasattr(prior_builder, "structural_penalty_factor"):
                prior_builder.structural_penalty_factor = penalty_factor
                logger.info("  Set structural_penalty_factor on BayesianPriorBuilder")

        # Mark edges with penalty metadata
        for source, target in violations:
            if causal_graph.has_edge(source, target):
                # Mark edge with penalty metadata
                causal_graph.edges[source, target]["penalty"] = True
                causal_graph.edges[source, target]["violation_reason"] = (
                    "structural_order"
                )
                causal_graph.edges[source, target]["penalty_factor"] = penalty_factor

                # Reduce confidence if present
                if "confidence" in causal_graph.edges[source, target]:
                    original = causal_graph.edges[source, target]["confidence"]
                    causal_graph.edges[source, target]["confidence"] = (
                        original * penalty_factor
                    )
                    logger.debug(
                        "Reduced edge confidence %s->%s: %.2f -> %.2f",
                        source,
                        target,
                        original,
                        original * penalty_factor,
                    )

        return penalty_factor

    def _map_dimension(self, dimension_str: str) -> "PolicyDimension":
        """
        Map dimension string to PolicyDimension enum

        Args:
            dimension_str: Dimension string (e.g., "ESTRATEGICO", "DIAGNOSTICO")

        Returns:
            PolicyDimension enum value
        """
        # Import PolicyDimension locally
        try:
            from contradiction_deteccion import PolicyDimension
        except ImportError:
            logger.warning("PolicyDimension not available, using fallback")
            # Create a fallback enum if import fails
            from enum import Enum

            class PolicyDimension(Enum):
                DIAGNOSTICO = "diagnóstico"
                ESTRATEGICO = "estratégico"
                PROGRAMATICO = "programático"
                FINANCIERO = "plan plurianual de inversiones"
                SEGUIMIENTO = "seguimiento y evaluación"
                TERRITORIAL = "ordenamiento territorial"

        dimension_map = {
            "DIAGNOSTICO": PolicyDimension.DIAGNOSTICO,
            "ESTRATEGICO": PolicyDimension.ESTRATEGICO,
            "PROGRAMATICO": PolicyDimension.PROGRAMATICO,
            "FINANCIERO": PolicyDimension.FINANCIERO,
            "SEGUIMIENTO": PolicyDimension.SEGUIMIENTO,
            "TERRITORIAL": PolicyDimension.TERRITORIAL,
        }

        return dimension_map.get(dimension_str.upper(), PolicyDimension.ESTRATEGICO)

    # ========================================================================
    # SCORE MAPPING METHODS: Map validator outputs to canonical notation
    # ========================================================================

    def _calculate_d6_q1_score(self, structural_result: "ValidacionResultado") -> float:
        """
        Calculate D6-Q1 score (Theory of Change completeness)

        Based on:
        - Presence of all canonical categories (INSUMOS → CAUSALIDAD)
        - Number of complete paths from INSUMOS to CAUSALIDAD

        Args:
            structural_result: ValidacionResultado from TeoriaCambio

        Returns:
            Score in [0, 1] range
        """
        # Component 1: Category completeness (50% weight)
        expected_categories = 5  # INSUMOS, PROCESOS, PRODUCTOS, RESULTADOS, CAUSALIDAD
        missing_count = len(structural_result.categorias_faltantes)
        category_score = max(
            0.0, (expected_categories - missing_count) / expected_categories
        )

        # Component 2: Complete paths (50% weight)
        path_count = len(structural_result.caminos_completos)
        path_score = min(1.0, path_count / 3.0)  # Normalize: 3+ paths = perfect score

        # Weighted combination
        d6_q1 = 0.5 * category_score + 0.5 * path_score

        return d6_q1

    def _calculate_d6_q2_score(self, structural_result: "ValidacionResultado") -> float:
        """
        Calculate D6-Q2 score (Causal order validity)

        Based on:
        - Number of order violations
        - Severity of violations (backward jumps)

        Args:
            structural_result: ValidacionResultado from TeoriaCambio

        Returns:
            Score in [0, 1] range
        """
        num_violations = len(structural_result.violaciones_orden)

        # Perfect score if no violations
        if num_violations == 0:
            return 1.0

        # Penalize based on violation count
        # Sigmoid-like decay: score = 1 / (1 + k * violations)
        k = 0.3  # Decay constant
        d6_q2 = 1.0 / (1.0 + k * num_violations)

        return d6_q2

    def _calculate_d2_q5_score(
        self, contradictions: List[Dict[str, Any]], semantic_chunks: List[SemanticChunk]
    ) -> float:
        """
        Calculate D2-Q5 score (Design phase contradiction-free)

        Based on:
        - Number of contradictions in design/strategic dimension
        - Severity of contradictions

        Args:
            contradictions: List of detected contradictions
            semantic_chunks: Semantic chunks analyzed

        Returns:
            Score in [0, 1] range
        """
        # Filter contradictions in design/strategic dimensions
        design_contradictions = [
            c
            for c in contradictions
            if "dimension" in c
            and c["dimension"] in ["ESTRATEGICO", "PROGRAMATICO", "DIAGNOSTICO"]
        ]

        num_design_contradictions = len(design_contradictions)
        total_design_chunks = sum(
            1
            for chunk in semantic_chunks
            if chunk.dimension in ["ESTRATEGICO", "PROGRAMATICO", "DIAGNOSTICO"]
        )

        if total_design_chunks == 0:
            return 0.5  # No design content = neutral score

        # Calculate contradiction rate
        contradiction_rate = num_design_contradictions / max(1, total_design_chunks)

        # Score decreases with contradiction rate
        # Sigmoid: score = 1 / (1 + k * rate)
        k = 10.0  # Decay constant
        d2_q5 = 1.0 / (1.0 + k * contradiction_rate)

        return d2_q5

    def _calculate_d6_q3_score(
        self, contradiction_density: float, contradictions: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate D6-Q3 score (Causal coherence / no logical contradictions)

        Based on:
        - Overall contradiction density
        - Severity of causal contradictions

        Args:
            contradiction_density: Ratio of contradictions to edges
            contradictions: List of detected contradictions

        Returns:
            Score in [0, 1] range
        """
        # Component 1: Density-based score (70% weight)
        # Linear decay from density 0 (perfect) to 0.15 (very poor)
        density_score = max(0.0, 1.0 - contradiction_density / 0.15)

        # Component 2: Severity-based score (30% weight)
        causal_contradictions = [
            c
            for c in contradictions
            if "type" in c and "causal" in str(c.get("type", "")).lower()
        ]
        severity_factor = (
            len(causal_contradictions) / max(1, len(contradictions))
            if contradictions
            else 0.0
        )
        severity_score = 1.0 - severity_factor

        # Weighted combination
        d6_q3 = 0.7 * density_score + 0.3 * severity_score

        return d6_q3

    def _calculate_d1_q5_score(
        self, dnp_result: Optional["ResultadoValidacionDNP"]
    ) -> float:
        """
        Calculate D1-Q5 score (Diagnostic aligned with DNP standards)

        Based on:
        - DNP competency validation
        - Diagnostic dimension compliance

        Args:
            dnp_result: ResultadoValidacionDNP from ValidadorDNP

        Returns:
            Score in [0, 1] range
        """
        if dnp_result is None:
            return 0.0

        # Component 1: Competency validation (60% weight)
        competency_score = 1.0 if dnp_result.cumple_competencias else 0.0

        # Component 2: Overall DNP score normalized (40% weight)
        dnp_normalized = dnp_result.score_total / 100.0

        # Weighted combination
        d1_q5 = 0.6 * competency_score + 0.4 * dnp_normalized

        return d1_q5

    def _calculate_d4_q5_score(
        self, dnp_result: Optional["ResultadoValidacionDNP"]
    ) -> float:
        """
        Calculate D4-Q5 score (Results validated with MGA indicators and BPIN)

        Based on:
        - MGA indicator compliance
        - BPIN validation (indicator count and quality)

        Args:
            dnp_result: ResultadoValidacionDNP from ValidadorDNP

        Returns:
            Score in [0, 1] range
        """
        if dnp_result is None:
            return 0.0

        # Component 1: MGA compliance (50% weight)
        mga_score = (
            1.0
            if dnp_result.cumple_mga
            else 0.5
            if dnp_result.indicadores_mga_usados
            else 0.0
        )

        # Component 2: BPIN indicator count (30% weight)
        num_indicators = len(dnp_result.indicadores_mga_usados)
        indicator_score = min(1.0, num_indicators / 5.0)  # 5+ indicators = perfect

        # Component 3: Overall DNP score (20% weight)
        dnp_normalized = dnp_result.score_total / 100.0

        # Weighted combination
        d4_q5 = 0.5 * mga_score + 0.3 * indicator_score + 0.2 * dnp_normalized

        return d4_q5

    # ========================================================================
    # CROSS-VALIDATION: GNN graph contradictions vs Bayesian implicit contradictions
    # ========================================================================

    def validate_contradiction_consistency(
        self,
        gnn_contradictions: List[Dict[str, Any]],
        bayesian_contradictions: List[Dict[str, Any]],
        semantic_chunks: List[SemanticChunk],
    ) -> Dict[str, Any]:
        """
        Cross-validate GNN-detected graph contradictions against Bayesian-inferred
        implicit contradictions.

        Identifies:
        - Overlap: Contradictions detected by both methods (high confidence)
        - GNN-only: Explicit graph structure contradictions
        - Bayesian-only: Implicit semantic/probabilistic contradictions
        - Conflicts: Cases where methods disagree

        Args:
            gnn_contradictions: Contradictions from GraphNeuralReasoningEngine
            bayesian_contradictions: Contradictions from BayesianCausalInference
            semantic_chunks: Source semantic chunks for reference

        Returns:
            Dict with cross-validation results and consistency metrics
        """
        logger.info("Cross-validating GNN and Bayesian contradiction detection")

        # Extract statement pairs from each method
        gnn_pairs = self._extract_contradiction_pairs(gnn_contradictions, "gnn")
        bayesian_pairs = self._extract_contradiction_pairs(
            bayesian_contradictions, "bayesian"
        )

        # Find overlaps and differences
        overlap = gnn_pairs & bayesian_pairs
        gnn_only = gnn_pairs - bayesian_pairs
        bayesian_only = bayesian_pairs - gnn_only

        # Calculate consistency metrics
        total_unique = len(gnn_pairs | bayesian_pairs)
        overlap_rate = len(overlap) / total_unique if total_unique > 0 else 0.0

        # High confidence: both methods agree
        high_confidence = [
            self._reconstruct_contradiction(
                pair, gnn_contradictions, bayesian_contradictions
            )
            for pair in overlap
        ]

        # Medium confidence: one method only
        medium_confidence_gnn = [
            self._find_contradiction_by_pair(pair, gnn_contradictions)
            for pair in gnn_only
        ]
        medium_confidence_bayesian = [
            self._find_contradiction_by_pair(pair, bayesian_contradictions)
            for pair in bayesian_only
        ]

        logger.info(
            "  Overlap: %d contradictions (%.1f%% consistency)",
            len(overlap),
            overlap_rate * 100,
        )
        logger.info("  GNN-only: %d contradictions (explicit graph)", len(gnn_only))
        logger.info(
            "  Bayesian-only: %d contradictions (implicit semantic)", len(bayesian_only)
        )

        return {
            "consistency_rate": overlap_rate,
            "total_unique_contradictions": total_unique,
            "high_confidence": high_confidence,
            "gnn_explicit": medium_confidence_gnn,
            "bayesian_implicit": medium_confidence_bayesian,
            "overlap_count": len(overlap),
            "gnn_only_count": len(gnn_only),
            "bayesian_only_count": len(bayesian_only),
            "recommendations": self._generate_cross_validation_recommendations(
                overlap_rate, len(gnn_only), len(bayesian_only)
            ),
        }

    def _extract_contradiction_pairs(
        self, contradictions: List[Dict[str, Any]], method: str
    ) -> Set[Tuple[str, str]]:
        """
        Extract normalized statement pairs from contradiction list

        Args:
            contradictions: List of contradiction dicts
            method: 'gnn' or 'bayesian' to determine extraction strategy

        Returns:
            Set of (stmt_id_1, stmt_id_2) tuples (normalized order)
        """
        pairs = set()

        for contradiction in contradictions:
            # Extract IDs based on method-specific structure
            if method == "gnn":
                # GNN contradictions: tuple format (stmt_a, stmt_b, score, attention)
                if isinstance(contradiction, tuple) and len(contradiction) >= 2:
                    stmt_a = getattr(contradiction[0], "text", str(contradiction[0]))[
                        :50
                    ]
                    stmt_b = getattr(contradiction[1], "text", str(contradiction[1]))[
                        :50
                    ]
                    pair = tuple(sorted([stmt_a, stmt_b]))
                    pairs.add(pair)
            elif method == "bayesian":
                # Bayesian contradictions: dict with statement_a, statement_b
                if isinstance(contradiction, dict):
                    stmt_a = contradiction.get("statement_a", {})
                    stmt_b = contradiction.get("statement_b", {})
                    text_a = (
                        stmt_a.get("text", str(stmt_a))[:50]
                        if isinstance(stmt_a, dict)
                        else str(stmt_a)[:50]
                    )
                    text_b = (
                        stmt_b.get("text", str(stmt_b))[:50]
                        if isinstance(stmt_b, dict)
                        else str(stmt_b)[:50]
                    )
                    pair = tuple(sorted([text_a, text_b]))
                    pairs.add(pair)

        return pairs

    def _reconstruct_contradiction(
        self,
        pair: Tuple[str, str],
        gnn_contradictions: List[Dict[str, Any]],
        bayesian_contradictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Reconstruct full contradiction info from both methods

        Args:
            pair: Statement pair (normalized)
            gnn_contradictions: GNN contradiction list
            bayesian_contradictions: Bayesian contradiction list

        Returns:
            Combined contradiction dict with both method's evidence
        """
        gnn_match = self._find_contradiction_by_pair(pair, gnn_contradictions)
        bayesian_match = self._find_contradiction_by_pair(pair, bayesian_contradictions)

        return {
            "pair": pair,
            "gnn_evidence": gnn_match,
            "bayesian_evidence": bayesian_match,
            "confidence": "HIGH",
            "detection_methods": ["GNN", "Bayesian"],
        }

    def _find_contradiction_by_pair(
        self, pair: Tuple[str, str], contradictions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find contradiction matching the given pair

        Args:
            pair: Statement pair to match
            contradictions: List to search

        Returns:
            Matching contradiction dict or None
        """
        for contradiction in contradictions:
            # Extract pair from contradiction (method-agnostic)
            if isinstance(contradiction, tuple):
                stmt_a = str(contradiction[0])[:50] if len(contradiction) > 0 else ""
                stmt_b = str(contradiction[1])[:50] if len(contradiction) > 1 else ""
            elif isinstance(contradiction, dict):
                stmt_a = str(contradiction.get("statement_a", ""))[:50]
                stmt_b = str(contradiction.get("statement_b", ""))[:50]
            else:
                continue

            check_pair = tuple(sorted([stmt_a, stmt_b]))
            if check_pair == pair:
                return (
                    contradiction
                    if isinstance(contradiction, dict)
                    else {
                        "statement_a": (
                            contradiction[0] if len(contradiction) > 0 else None
                        ),
                        "statement_b": (
                            contradiction[1] if len(contradiction) > 1 else None
                        ),
                        "score": contradiction[2] if len(contradiction) > 2 else 0.0,
                    }
                )

        return None

    def _generate_cross_validation_recommendations(
        self, consistency_rate: float, gnn_only_count: int, bayesian_only_count: int
    ) -> List[str]:
        """
        Generate recommendations based on cross-validation results

        Args:
            consistency_rate: Overlap rate between methods
            gnn_only_count: Contradictions detected by GNN only
            bayesian_only_count: Contradictions detected by Bayesian only

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if consistency_rate < 0.3:
            recommendations.append(
                "LOW CONSISTENCY: GNN and Bayesian methods show <30% agreement. "
                "Review contradiction detection parameters and thresholds."
            )
        elif consistency_rate < 0.5:
            recommendations.append(
                "MODERATE CONSISTENCY: Consider manual review of method-specific contradictions."
            )
        else:
            recommendations.append(
                "HIGH CONSISTENCY: Strong agreement between methods validates contradiction detection."
            )

        if gnn_only_count > bayesian_only_count * 2:
            recommendations.append(
                "GNN detecting significantly more explicit graph contradictions. "
                "Verify graph structure and edge semantics."
            )
        elif bayesian_only_count > gnn_only_count * 2:
            recommendations.append(
                "Bayesian detecting significantly more implicit contradictions. "
                "Consider strengthening graph representation to capture semantic relationships."
            )

        if gnn_only_count + bayesian_only_count > 20:
            recommendations.append(
                f"HIGH TOTAL CONTRADICTION COUNT ({gnn_only_count + bayesian_only_count}). "
                "Prioritize high-confidence overlaps for resolution."
            )

        return recommendations
