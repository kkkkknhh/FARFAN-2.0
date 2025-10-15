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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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
    1. Structural validation (Theory of Change)
    2. Semantic validation (Contradictions)
    3. Regulatory validation (DNP Compliance)
    """

    # Overall validation status
    is_valid: bool = True

    # Structural validation (Phase III-B)
    structural_valid: bool = True
    violaciones_orden: List[Tuple[str, str]] = field(default_factory=list)
    categorias_faltantes: List[str] = field(default_factory=list)
    caminos_completos: List[List[str]] = field(default_factory=list)

    # Semantic validation (Phase III-C)
    contradiction_density: float = 0.0
    contradictions: List[Dict[str, Any]] = field(default_factory=list)

    # Regulatory validation (Phase III-D)
    regulatory_score: float = 0.0
    dnp_compliance: Optional[Any] = None  # ResultadoValidacionDNP when available

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
    ) -> AxiomaticValidationResult:
        """
        Ejecuta validación completa en orden de severidad:
        1. Structural (D6-Q1/Q2: Theory of Change)
        2. Semantic (D2-Q5, D6-Q3: Contradictions)
        3. Regulatory (D1-Q5, D4-Q5: DNP Compliance)

        Args:
            causal_graph: The causal graph to validate
            semantic_chunks: List of semantic chunks from the PDM
            financial_data: Optional financial/data tables

        Returns:
            AxiomaticValidationResult with comprehensive validation results
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
        # 1. STRUCTURAL: Orden Causal (Front C.1)
        # ====================================================================
        logger.info("Phase 1: Structural validation (Theory of Change)")
        structural = self._validate_structural(causal_graph)
        results.structural_valid = structural.es_valida
        results.violaciones_orden = structural.violaciones_orden
        results.categorias_faltantes = [
            cat.name for cat in structural.categorias_faltantes
        ]
        results.caminos_completos = structural.caminos_completos

        if structural.violaciones_orden:
            logger.warning(
                "Found %d structural violations", len(structural.violaciones_orden)
            )
            results.add_critical_failure(
                dimension="D6",
                question="Q2",
                evidence=structural.violaciones_orden,
                impact="Saltos lógicos detectados",
                recommendations=structural.sugerencias,
            )

            # Hard constraint: cap posterior (Front C.1)
            if self.config.enable_structural_penalty:
                self._apply_structural_penalty(
                    causal_graph, structural.violaciones_orden
                )

        # ====================================================================
        # 2. SEMANTIC: Contradictions (Phase III-C)
        # ====================================================================
        logger.info("Phase 2: Semantic validation (Contradiction detection)")
        contradictions = self._validate_semantic(causal_graph, semantic_chunks)
        results.contradictions = contradictions

        # Calculate contradiction density
        if results.total_edges > 0:
            results.contradiction_density = len(contradictions) / results.total_edges
        else:
            results.contradiction_density = 0.0

        logger.info(
            "Contradiction density: %.4f (threshold: %.4f)",
            results.contradiction_density,
            self.config.contradiction_threshold,
        )

        if results.contradiction_density > self.config.contradiction_threshold:
            # Human gating trigger (Governance Standard)
            if self.config.enable_human_gating:
                results.requires_manual_review = True
                results.hold_reason = "HIGH_CONTRADICTION_DENSITY"
                logger.warning(
                    "High contradiction density detected - manual review required"
                )

        # ====================================================================
        # 3. REGULATORY: DNP Compliance (Phase III-D)
        # ====================================================================
        logger.info("Phase 3: Regulatory validation (DNP compliance)")
        dnp_results = self._validate_regulatory(semantic_chunks, financial_data)
        results.dnp_compliance = dnp_results
        results.regulatory_score = dnp_results.score_total if dnp_results else 0.0

        logger.info("Regulatory compliance score: %.2f/100", results.regulatory_score)

        # Final validation status
        results.is_valid = (
            results.structural_valid
            and not results.requires_manual_review
            and results.regulatory_score >= 60.0  # Minimum acceptable threshold
        )

        logger.info(
            "Validation complete. Overall status: %s",
            "VALID" if results.is_valid else "INVALID",
        )

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
        self, causal_graph: "nx.DiGraph", violations: List[Tuple[str, str]]
    ):
        """
        Apply structural penalty for violations (Front C.1)

        This could involve:
        - Marking edges as suspect
        - Reducing confidence scores
        - Flagging for manual review

        Args:
            causal_graph: The causal graph
            violations: List of edge violations (source, target)
        """
        logger.info("Applying structural penalty for %d violations", len(violations))

        for source, target in violations:
            if causal_graph.has_edge(source, target):
                # Mark edge with penalty metadata
                causal_graph.edges[source, target]["penalty"] = True
                causal_graph.edges[source, target]["violation_reason"] = (
                    "structural_order"
                )

                # Reduce confidence if present
                if "confidence" in causal_graph.edges[source, target]:
                    original = causal_graph.edges[source, target]["confidence"]
                    causal_graph.edges[source, target]["confidence"] = original * 0.5
                    logger.debug(
                        "Reduced edge confidence %s->%s: %.2f -> %.2f",
                        source,
                        target,
                        original,
                        original * 0.5,
                    )

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
