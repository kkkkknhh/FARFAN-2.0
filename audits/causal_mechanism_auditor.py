#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal Mechanism Rigor Auditor
Part 2: Analytical D3, D6 Audit

Implements SOTA process-tracing audits per Beach & Pedersen 2019:
- Audit Point 2.1: Mechanism Necessity Check (D3-Q5)
- Audit Point 2.2: Root Cause Mapping (D2-Q3)
- Audit Point 2.3: Causal Proportionality (D6-Q2)
- Audit Point 2.4: Explicit Activity Logic (D2-Q2)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

# ============================================================================
# Data Structures
# ============================================================================


class QualityGrade(Enum):
    """Quality grades for audit results"""

    EXCELENTE = "Excelente"
    BUENO = "Bueno"
    REGULAR = "Regular"
    INSUFICIENTE = "Insuficiente"


@dataclass
class MechanismNecessityResult:
    """
    Result from Audit Point 2.1: Mechanism Necessity Check (D3-Q5)

    Per Beach & Pedersen 2019, mechanism necessity requires:
    - Entity (responsible entity)
    - Activity (specific actions)
    - Budget (resource allocation)
    - Timeline (temporal specification)
    """

    link_id: str
    is_necessary: bool
    necessity_score: float  # 0.0 to 1.0
    missing_components: List[str]
    quality_grade: QualityGrade
    evidence: Dict[str, Any]
    remediation: Optional[str] = None

    # Micro-foundations check
    has_entity: bool = False
    has_activity: bool = False
    has_budget: bool = False
    has_timeline: bool = False

    # Cross-reference to PDM text
    entity_mentions: List[str] = field(default_factory=list)
    activity_mentions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "link_id": self.link_id,
            "is_necessary": self.is_necessary,
            "necessity_score": self.necessity_score,
            "missing_components": self.missing_components,
            "quality_grade": self.quality_grade.value,
            "micro_foundations": {
                "has_entity": self.has_entity,
                "has_activity": self.has_activity,
                "has_budget": self.has_budget,
                "has_timeline": self.has_timeline,
            },
            "evidence": self.evidence,
            "entity_mentions": self.entity_mentions,
            "activity_mentions": self.activity_mentions,
            "remediation": self.remediation,
        }


@dataclass
class RootCauseMappingResult:
    """
    Result from Audit Point 2.2: Root Cause Mapping (D2-Q3)

    Cross-dimensional linkage from D2 activities to D1 root causes
    via linguistic markers (e.g., "para abordar la causa").
    """

    activity_id: str
    root_causes: List[str]
    linguistic_markers: List[str]
    mapping_confidence: float  # 0.0 to 1.0
    coherence_score: float  # Target: >95%
    quality_grade: QualityGrade
    linkage_phrases: List[Tuple[str, str, str]]  # (marker, d2_node, d1_node)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "activity_id": self.activity_id,
            "root_causes": self.root_causes,
            "linguistic_markers": self.linguistic_markers,
            "mapping_confidence": self.mapping_confidence,
            "coherence_score": self.coherence_score,
            "quality_grade": self.quality_grade.value,
            "linkage_phrases": [
                {"marker": m, "d2_node": d2, "d1_node": d1}
                for m, d2, d1 in self.linkage_phrases
            ],
        }


@dataclass
class CausalProportionalityResult:
    """
    Result from Audit Point 2.3: Causal Proportionality (D6-Q2)

    Detects/penalizes logical jumps (salto lógico).
    Caps posterior ≤0.6 for impossible transitions (Product → Impact).
    """

    link_id: str
    source_type: str
    target_type: str
    is_proportional: bool
    has_logical_jump: bool
    posterior_capped: bool
    original_posterior: float
    adjusted_posterior: float
    quality_grade: QualityGrade
    violation_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "link_id": self.link_id,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "is_proportional": self.is_proportional,
            "has_logical_jump": self.has_logical_jump,
            "posterior_capped": self.posterior_capped,
            "original_posterior": self.original_posterior,
            "adjusted_posterior": self.adjusted_posterior,
            "quality_grade": self.quality_grade.value,
            "violation_details": self.violation_details,
        }


@dataclass
class ActivityLogicResult:
    """
    Result from Audit Point 2.4: Explicit Activity Logic (D2-Q2)

    Extracts Instrument, Target Population, Causal Logic
    (porque genera, mecanismo) for key activities.
    """

    activity_id: str
    instrument: Optional[str]
    target_population: Optional[str]
    causal_logic: Optional[str]
    extraction_accuracy: float  # Target: 100%
    quality_grade: QualityGrade
    matched_rationale: Optional[str] = None
    missing_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "activity_id": self.activity_id,
            "instrument": self.instrument,
            "target_population": self.target_population,
            "causal_logic": self.causal_logic,
            "extraction_accuracy": self.extraction_accuracy,
            "quality_grade": self.quality_grade.value,
            "matched_rationale": self.matched_rationale,
            "missing_components": self.missing_components,
        }


# ============================================================================
# Main Auditor
# ============================================================================


class CausalMechanismAuditor:
    """
    Enforces non-miraculous links via mechanism necessity
    per SOTA process-tracing (Beach & Pedersen 2019)
    """

    # Linguistic markers for root cause linkage
    ROOT_CAUSE_MARKERS = [
        "para abordar la causa",
        "para atender la causa",
        "con el fin de resolver",
        "con el propósito de solucionar",
        "dirigido a resolver",
        "orientado a atender",
        "busca solucionar",
        "pretende abordar",
    ]

    # Causal logic markers for activity logic
    CAUSAL_LOGIC_MARKERS = [
        "porque genera",
        "porque produce",
        "ya que permite",
        "dado que facilita",
        "mecanismo",
        "mediante el cual",
        "a través del cual",
        "por medio de",
    ]

    # Instrument markers
    INSTRUMENT_MARKERS = [
        "mediante",
        "a través de",
        "por medio de",
        "utilizando",
        "con el instrumento",
        "con la herramienta",
        "con el mecanismo",
    ]

    # Target population markers
    TARGET_POPULATION_MARKERS = [
        "dirigido a",
        "orientado a",
        "beneficia a",
        "atiende a",
        "población objetivo",
        "población beneficiaria",
        "grupo poblacional",
    ]

    # Impossible transitions (logical jumps)
    IMPOSSIBLE_TRANSITIONS = [
        ("producto", "impacto"),  # Product → Impact
        ("programa", "impacto"),  # Program → Impact
    ]

    # Maximum posterior for impossible transitions
    MAX_POSTERIOR_IMPOSSIBLE = 0.6

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    # ========================================================================
    # Audit Point 2.1: Mechanism Necessity Check (D3-Q5)
    # ========================================================================

    def audit_mechanism_necessity(
        self,
        graph: nx.DiGraph,
        text: str,
        inferred_mechanisms: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, MechanismNecessityResult]:
        """
        Audit Point 2.1: Mechanism Necessity Check (D3-Q5)

        Check Criteria: BayesianMechanismInference._test_necessity requires
        micro-foundations (Entity, Activity, Budget, Timeline).
        "Excelente" only if necessity_test['is_necessary'] True for ≥80% links.

        Args:
            graph: Causal graph with nodes and edges
            text: Full PDM text for cross-reference
            inferred_mechanisms: Optional dict of inferred mechanisms from Bayesian inference

        Returns:
            Dict mapping link_id to MechanismNecessityResult
        """
        self.logger.info("Starting Mechanism Necessity Check (D3-Q5)...")

        results = {}
        total_links = 0
        necessary_links = 0

        for source, target in graph.edges():
            link_id = f"{source}→{target}"
            total_links += 1

            # Get node data
            source_node = graph.nodes[source]
            target_node = graph.nodes[target]
            edge_data = graph.edges[source, target]

            # Check micro-foundations
            has_entity = self._check_entity(source_node, text, source)
            has_activity = self._check_activity(source_node, edge_data, text, source)
            has_budget = self._check_budget(source_node)
            has_timeline = self._check_timeline(source_node, text, source)

            # Extract evidence mentions from text
            entity_mentions = self._extract_entity_mentions(source, text)
            activity_mentions = self._extract_activity_mentions(source, text)

            # Calculate necessity score
            components_present = sum(
                [has_entity, has_activity, has_budget, has_timeline]
            )
            necessity_score = components_present / 4.0

            # Determine missing components
            missing_components = []
            if not has_entity:
                missing_components.append("entity")
            if not has_activity:
                missing_components.append("activity")
            if not has_budget:
                missing_components.append("budget")
            if not has_timeline:
                missing_components.append("timeline")

            # Is necessary if all components present
            is_necessary = math.isclose(necessity_score, 1.0, rel_tol=1e-9, abs_tol=1e-12)  # replaced float equality with isclose (tolerance from DEFAULT_FLOAT_TOLS)
            if is_necessary:
                necessary_links += 1

            # Determine quality grade
            quality_grade = self._get_necessity_quality_grade(necessity_score)

            # Generate remediation
            remediation = (
                self._generate_necessity_remediation(link_id, missing_components)
                if missing_components
                else None
            )

            # Create result
            result = MechanismNecessityResult(
                link_id=link_id,
                is_necessary=is_necessary,
                necessity_score=necessity_score,
                missing_components=missing_components,
                quality_grade=quality_grade,
                evidence={
                    "source_node": source,
                    "target_node": target,
                    "components_present": components_present,
                    "total_components": 4,
                },
                remediation=remediation,
                has_entity=has_entity,
                has_activity=has_activity,
                has_budget=has_budget,
                has_timeline=has_timeline,
                entity_mentions=entity_mentions,
                activity_mentions=activity_mentions,
            )

            results[link_id] = result

        # Calculate overall quality grade
        necessity_rate = necessary_links / total_links if total_links > 0 else 0.0
        overall_grade = (
            QualityGrade.EXCELENTE
            if necessity_rate >= 0.8
            else (
                QualityGrade.BUENO
                if necessity_rate >= 0.6
                else (
                    QualityGrade.REGULAR
                    if necessity_rate >= 0.4
                    else QualityGrade.INSUFICIENTE
                )
            )
        )

        self.logger.info(
            f"Mechanism Necessity Check complete: {necessary_links}/{total_links} "
            f"links necessary ({necessity_rate * 100:.1f}%) - Grade: {overall_grade.value}"
        )

        return results

    def _check_entity(self, node_data: Dict[str, Any], text: str, node_id: str) -> bool:
        """Check if entity is documented for node"""
        # Check for responsible_entity field
        if node_data.get("responsible_entity"):
            return True

        # Check for entity_activity
        ea = node_data.get("entity_activity")
        if ea and isinstance(ea, dict):
            if ea.get("entity"):
                return True

        return False

    def _check_activity(
        self,
        node_data: Dict[str, Any],
        edge_data: Dict[str, Any],
        text: str,
        node_id: str,
    ) -> bool:
        """Check if activity sequence is documented"""
        # Check for entity_activity
        ea = node_data.get("entity_activity")
        if ea and isinstance(ea, dict):
            if ea.get("activity") or ea.get("verb_lemma"):
                return True

        # Check edge for causal logic/keyword
        if edge_data.get("logic") or edge_data.get("keyword"):
            return True

        return False

    def _check_budget(self, node_data: Dict[str, Any]) -> bool:
        """Check if budget is allocated"""
        budget = node_data.get("financial_allocation")
        return budget is not None and budget > 0

    def _check_timeline(
        self, node_data: Dict[str, Any], text: str, node_id: str
    ) -> bool:
        """Check if timeline is specified"""
        # Look for temporal indicators near node mention
        temporal_patterns = [
            r"\d{4}",  # Year
            r"\d{1,2}\s*(?:año|años|mes|meses)",  # Duration
            r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)",
            r"(?:corto|mediano|largo)\s*plazo",
            r"cronograma",
            r"periodo",
            r"etapa",
        ]

        # Search in node context
        node_pattern = re.escape(node_id)
        for match in re.finditer(node_pattern, text, re.IGNORECASE):
            context_start = max(0, match.start() - 200)
            context_end = min(len(text), match.end() + 200)
            context = text[context_start:context_end]

            for pattern in temporal_patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    return True

        return False

    def _extract_entity_mentions(self, node_id: str, text: str) -> List[str]:
        """Extract entity mentions from text"""
        mentions = []
        entity_pattern = r"entidad\s+(?:responsable|ejecutora)"

        node_pattern = re.escape(node_id)
        for match in re.finditer(node_pattern, text, re.IGNORECASE):
            context_start = max(0, match.start() - 200)
            context_end = min(len(text), match.end() + 200)
            context = text[context_start:context_end]

            for entity_match in re.finditer(entity_pattern, context, re.IGNORECASE):
                mentions.append(entity_match.group())

        return mentions

    def _extract_activity_mentions(self, node_id: str, text: str) -> List[str]:
        """Extract activity mentions from text"""
        mentions = []
        activity_pattern = (
            r"(?:realiza|ejecuta|implementa|desarrolla)\s+(?:actividad|acción|proyecto)"
        )

        node_pattern = re.escape(node_id)
        for match in re.finditer(node_pattern, text, re.IGNORECASE):
            context_start = max(0, match.start() - 200)
            context_end = min(len(text), match.end() + 200)
            context = text[context_start:context_end]

            for activity_match in re.finditer(activity_pattern, context, re.IGNORECASE):
                mentions.append(activity_match.group())

        return mentions

    def _get_necessity_quality_grade(self, score: float) -> QualityGrade:
        """Determine quality grade for necessity score"""
        if score >= 0.9:
            return QualityGrade.EXCELENTE
        elif score >= 0.75:
            return QualityGrade.BUENO
        elif score >= 0.5:
            return QualityGrade.REGULAR
        else:
            return QualityGrade.INSUFICIENTE

    def _generate_necessity_remediation(
        self, link_id: str, missing_components: List[str]
    ) -> str:
        """Generate remediation text for necessity failures"""
        component_names = {
            "entity": "entidad responsable",
            "activity": "secuencia de actividades",
            "budget": "presupuesto asignado",
            "timeline": "cronograma de ejecución",
        }

        missing_names = [component_names.get(c, c) for c in missing_components]
        missing_str = ", ".join(missing_names)

        return (
            f"El mecanismo causal {link_id} falla el test de necesidad. "
            f"Componentes faltantes: {missing_str}. "
            f"Se requiere documentar estos componentes para validar "
            f"la cadena causal conforme a Beach & Pedersen 2019."
        )

    # ========================================================================
    # Audit Point 2.2: Root Cause Mapping (D2-Q3)
    # ========================================================================

    def audit_root_cause_mapping(
        self, graph: nx.DiGraph, text: str
    ) -> Dict[str, RootCauseMappingResult]:
        """
        Audit Point 2.2: Root Cause Mapping (D2-Q3)

        Check Criteria: Cross-dimensional links map D2 activities to D1 root causes
        via linguistic markers (e.g., "para abordar la causa").

        Quality Evidence: Search initial_processor_causal_policy logs for linkage phrases;
        validate against D1 nodes. Coherence score >95%.

        Args:
            graph: Causal graph with dimensional information
            text: Full PDM text

        Returns:
            Dict mapping activity_id to RootCauseMappingResult
        """
        self.logger.info("Starting Root Cause Mapping audit (D2-Q3)...")

        results = {}

        # Identify D1 (diagnostic) and D2 (design) nodes
        d1_nodes = []
        d2_nodes = []

        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            node_type = node_data.get("type", "")

            # Heuristic: 'programa' nodes are often D1 (diagnostic)
            # 'producto' nodes are often D2 (design/intervention)
            if (
                node_type == "programa"
                or "diagnostic" in node_data.get("text", "").lower()
            ):
                d1_nodes.append(node_id)
            elif node_type == "producto":
                d2_nodes.append(node_id)

        # For each D2 activity, find linkages to D1 root causes
        for d2_node in d2_nodes:
            activity_id = d2_node

            root_causes = []
            linguistic_markers = []
            linkage_phrases = []

            # Search for linguistic markers linking D2 to D1
            for marker in self.ROOT_CAUSE_MARKERS:
                pattern = re.compile(
                    rf"{re.escape(d2_node)}\s+{re.escape(marker)}\s+({'|'.join(re.escape(d1) for d1 in d1_nodes)})",
                    re.IGNORECASE,
                )

                for match in pattern.finditer(text):
                    d1_node = match.group(1)
                    if d1_node in d1_nodes:
                        root_causes.append(d1_node)
                        linguistic_markers.append(marker)
                        linkage_phrases.append((marker, d2_node, d1_node))

            # Calculate mapping confidence
            mapping_confidence = (
                min(1.0, len(root_causes) / max(1, len(d1_nodes))) if d1_nodes else 0.0
            )

            # Calculate coherence score (based on consistency of linkages)
            # High coherence if multiple consistent linkages exist
            unique_markers = len(set(linguistic_markers))
            unique_root_causes = len(set(root_causes))

            if unique_root_causes > 0:
                # Perfect coherence if all linkages point to same root cause
                coherence_score = 1.0 - (unique_root_causes - 1) * 0.1
                coherence_score = max(0.0, coherence_score)
            else:
                coherence_score = 0.0

            # Determine quality grade based on coherence score
            if coherence_score >= 0.95:
                quality_grade = QualityGrade.EXCELENTE
            elif coherence_score >= 0.80:
                quality_grade = QualityGrade.BUENO
            elif coherence_score >= 0.60:
                quality_grade = QualityGrade.REGULAR
            else:
                quality_grade = QualityGrade.INSUFICIENTE

            result = RootCauseMappingResult(
                activity_id=activity_id,
                root_causes=list(set(root_causes)),
                linguistic_markers=list(set(linguistic_markers)),
                mapping_confidence=mapping_confidence,
                coherence_score=coherence_score,
                quality_grade=quality_grade,
                linkage_phrases=linkage_phrases,
            )

            results[activity_id] = result

        # Calculate overall coherence
        if results:
            avg_coherence = np.mean([r.coherence_score for r in results.values()])
            self.logger.info(
                f"Root Cause Mapping complete: {len(results)} activities mapped, "
                f"average coherence: {avg_coherence * 100:.1f}%"
            )
        else:
            self.logger.warning("No D2 activities found for root cause mapping")

        return results

    # ========================================================================
    # Audit Point 2.3: Causal Proportionality (D6-Q2)
    # ========================================================================

    def audit_causal_proportionality(
        self, graph: nx.DiGraph
    ) -> Dict[str, CausalProportionalityResult]:
        """
        Audit Point 2.3: Causal Proportionality (D6-Q2)

        Check Criteria: Detects/penalizes logical jumps (salto lógico);
        caps posterior ≤0.6 for impossible transitions (Product → Impact).

        Quality Evidence: Force high-semantic invalid link; confirm capped output
        in Bayesian logs.

        Args:
            graph: Causal graph with type information and posteriors

        Returns:
            Dict mapping link_id to CausalProportionalityResult
        """
        self.logger.info("Starting Causal Proportionality audit (D6-Q2)...")

        results = {}
        total_links = 0
        capped_links = 0

        for source, target in graph.edges():
            link_id = f"{source}→{target}"
            total_links += 1

            source_data = graph.nodes[source]
            target_data = graph.nodes[target]
            edge_data = graph.edges[source, target]

            source_type = source_data.get("type", "unknown")
            target_type = target_data.get("type", "unknown")

            # Get original posterior
            original_posterior = edge_data.get(
                "posterior_mean", edge_data.get("strength", 0.5)
            )

            # Check for impossible transitions
            is_impossible = (source_type, target_type) in self.IMPOSSIBLE_TRANSITIONS
            has_logical_jump = is_impossible

            # Cap posterior if impossible
            posterior_capped = False
            adjusted_posterior = original_posterior

            if is_impossible and original_posterior > self.MAX_POSTERIOR_IMPOSSIBLE:
                adjusted_posterior = self.MAX_POSTERIOR_IMPOSSIBLE
                posterior_capped = True
                capped_links += 1

                # Update graph with capped value
                graph.edges[source, target]["posterior_mean"] = adjusted_posterior
                graph.edges[source, target]["strength"] = adjusted_posterior
                graph.edges[source, target]["proportionality_capped"] = True

            # Determine if proportional
            is_proportional = not has_logical_jump

            # Quality grade
            if is_proportional:
                quality_grade = QualityGrade.EXCELENTE
            elif posterior_capped:
                quality_grade = QualityGrade.REGULAR
            else:
                quality_grade = QualityGrade.INSUFICIENTE

            # Violation details
            violation_details = None
            if has_logical_jump:
                violation_details = (
                    f"Salto lógico detectado: {source_type} → {target_type}. "
                    f"Posterior original {original_posterior:.3f} "
                    f"{'capado a' if posterior_capped else 'debería ser ≤'} "
                    f"{self.MAX_POSTERIOR_IMPOSSIBLE:.1f} (Pearl 2009, Mahoney 2010)"
                )

            result = CausalProportionalityResult(
                link_id=link_id,
                source_type=source_type,
                target_type=target_type,
                is_proportional=is_proportional,
                has_logical_jump=has_logical_jump,
                posterior_capped=posterior_capped,
                original_posterior=original_posterior,
                adjusted_posterior=adjusted_posterior,
                quality_grade=quality_grade,
                violation_details=violation_details,
            )

            results[link_id] = result

        self.logger.info(
            f"Causal Proportionality audit complete: {capped_links}/{total_links} "
            f"links capped for logical jumps"
        )

        return results

    # ========================================================================
    # Audit Point 2.4: Explicit Activity Logic (D2-Q2)
    # ========================================================================

    def audit_activity_logic(
        self, graph: nx.DiGraph, text: str
    ) -> Dict[str, ActivityLogicResult]:
        """
        Audit Point 2.4: Explicit Activity Logic (D2-Q2)

        Check Criteria: Extracts Instrument, Target Population, Causal Logic
        (porque genera, mecanismo) for key activities.

        Quality Evidence: Sample activities; match extractions to PDM rationale clauses.
        Target: 100% extraction accuracy via NLP tuned to QCA patterns.

        Args:
            graph: Causal graph with activity nodes
            text: Full PDM text

        Returns:
            Dict mapping activity_id to ActivityLogicResult
        """
        self.logger.info("Starting Activity Logic audit (D2-Q2)...")

        results = {}

        # Focus on 'producto' nodes (key activities)
        activity_nodes = [
            node_id
            for node_id in graph.nodes()
            if graph.nodes[node_id].get("type") == "producto"
        ]

        for activity_id in activity_nodes:
            # Search for activity in text
            activity_pattern = re.escape(activity_id)

            instrument = None
            target_population = None
            causal_logic = None
            matched_rationale = None

            for match in re.finditer(activity_pattern, text, re.IGNORECASE):
                context_start = max(0, match.start() - 500)
                context_end = min(len(text), match.end() + 500)
                context = text[context_start:context_end]

                # Extract instrument
                for marker in self.INSTRUMENT_MARKERS:
                    inst_pattern = rf"{marker}\s+([^.;]{10, 100})"
                    inst_match = re.search(inst_pattern, context, re.IGNORECASE)
                    if inst_match:
                        instrument = inst_match.group(1).strip()
                        break

                # Extract target population
                for marker in self.TARGET_POPULATION_MARKERS:
                    target_pattern = rf"{marker}\s+([^.;]{10, 100})"
                    target_match = re.search(target_pattern, context, re.IGNORECASE)
                    if target_match:
                        target_population = target_match.group(1).strip()
                        break

                # Extract causal logic
                for marker in self.CAUSAL_LOGIC_MARKERS:
                    logic_pattern = rf"{marker}\s+([^.;]{10, 200})"
                    logic_match = re.search(logic_pattern, context, re.IGNORECASE)
                    if logic_match:
                        causal_logic = logic_match.group(1).strip()
                        matched_rationale = context[:300]
                        break

                # Break after first match with extractions
                if instrument or target_population or causal_logic:
                    break

            # Calculate extraction accuracy
            components_found = sum(
                [
                    instrument is not None,
                    target_population is not None,
                    causal_logic is not None,
                ]
            )
            extraction_accuracy = components_found / 3.0

            # Determine missing components
            missing_components = []
            if not instrument:
                missing_components.append("instrument")
            if not target_population:
                missing_components.append("target_population")
            if not causal_logic:
                missing_components.append("causal_logic")

            # Quality grade (target 100%)
            if math.isclose(extraction_accuracy, 1.0, rel_tol=1e-9, abs_tol=1e-12):  # replaced float equality with isclose (tolerance from DEFAULT_FLOAT_TOLS)
                quality_grade = QualityGrade.EXCELENTE
            elif extraction_accuracy >= 0.66:
                quality_grade = QualityGrade.BUENO
            elif extraction_accuracy >= 0.33:
                quality_grade = QualityGrade.REGULAR
            else:
                quality_grade = QualityGrade.INSUFICIENTE

            result = ActivityLogicResult(
                activity_id=activity_id,
                instrument=instrument,
                target_population=target_population,
                causal_logic=causal_logic,
                extraction_accuracy=extraction_accuracy,
                quality_grade=quality_grade,
                matched_rationale=matched_rationale,
                missing_components=missing_components,
            )

            results[activity_id] = result

        # Calculate overall accuracy
        if results:
            avg_accuracy = np.mean([r.extraction_accuracy for r in results.values()])
            self.logger.info(
                f"Activity Logic audit complete: {len(results)} activities analyzed, "
                f"average extraction accuracy: {avg_accuracy * 100:.1f}%"
            )
        else:
            self.logger.warning("No activity nodes found for logic extraction")

        return results

    # ========================================================================
    # Comprehensive Audit Report
    # ========================================================================

    def generate_comprehensive_audit(
        self,
        graph: nx.DiGraph,
        text: str,
        inferred_mechanisms: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive audit report for all four audit points

        Returns:
            Dict with all audit results and summary statistics
        """
        self.logger.info("Generating comprehensive causal mechanism rigor audit...")

        # Run all four audits
        necessity_results = self.audit_mechanism_necessity(
            graph, text, inferred_mechanisms
        )
        root_cause_results = self.audit_root_cause_mapping(graph, text)
        proportionality_results = self.audit_causal_proportionality(graph)
        activity_logic_results = self.audit_activity_logic(graph, text)

        # Calculate summary statistics
        summary = {
            "audit_point_2_1_mechanism_necessity": {
                "total_links": len(necessity_results),
                "necessary_links": sum(
                    1 for r in necessity_results.values() if r.is_necessary
                ),
                "necessity_rate": (
                    sum(1 for r in necessity_results.values() if r.is_necessary)
                    / len(necessity_results)
                    if necessity_results
                    else 0.0
                ),
                "average_score": (
                    np.mean([r.necessity_score for r in necessity_results.values()])
                    if necessity_results
                    else 0.0
                ),
                "quality_distribution": self._count_quality_grades(
                    necessity_results.values()
                ),
            },
            "audit_point_2_2_root_cause_mapping": {
                "total_activities": len(root_cause_results),
                "mapped_activities": sum(
                    1 for r in root_cause_results.values() if r.root_causes
                ),
                "average_coherence": (
                    np.mean([r.coherence_score for r in root_cause_results.values()])
                    if root_cause_results
                    else 0.0
                ),
                "meets_target": (
                    np.mean([r.coherence_score for r in root_cause_results.values()])
                    > 0.95
                    if root_cause_results
                    else False
                ),
                "quality_distribution": self._count_quality_grades(
                    root_cause_results.values()
                ),
            },
            "audit_point_2_3_causal_proportionality": {
                "total_links": len(proportionality_results),
                "proportional_links": sum(
                    1 for r in proportionality_results.values() if r.is_proportional
                ),
                "capped_links": sum(
                    1 for r in proportionality_results.values() if r.posterior_capped
                ),
                "logical_jumps": sum(
                    1 for r in proportionality_results.values() if r.has_logical_jump
                ),
                "quality_distribution": self._count_quality_grades(
                    proportionality_results.values()
                ),
            },
            "audit_point_2_4_activity_logic": {
                "total_activities": len(activity_logic_results),
                "complete_extractions": sum(
                    1
                    for r in activity_logic_results.values()
                    if np.isclose(r.extraction_accuracy, 1.0, rtol=1e-9, atol=1e-12)  # replaced float equality with isclose (tolerance from DEFAULT_FLOAT_TOLS)
                ),
                "average_accuracy": (
                    np.mean(
                        [r.extraction_accuracy for r in activity_logic_results.values()]
                    )
                    if activity_logic_results
                    else 0.0
                ),
                "meets_target": (
                    np.mean(
                        [r.extraction_accuracy for r in activity_logic_results.values()]
                    )
                    >= 1.0
                    if activity_logic_results
                    else False
                ),
                "quality_distribution": self._count_quality_grades(
                    activity_logic_results.values()
                ),
            },
        }

        self.logger.info("Comprehensive audit complete")

        return {
            "summary": summary,
            "audit_point_2_1_necessity": {
                k: v.to_dict() for k, v in necessity_results.items()
            },
            "audit_point_2_2_root_cause_mapping": {
                k: v.to_dict() for k, v in root_cause_results.items()
            },
            "audit_point_2_3_proportionality": {
                k: v.to_dict() for k, v in proportionality_results.items()
            },
            "audit_point_2_4_activity_logic": {
                k: v.to_dict() for k, v in activity_logic_results.items()
            },
        }

    def _count_quality_grades(self, results) -> Dict[str, int]:
        """Count quality grade distribution"""
        counts = {grade.value: 0 for grade in QualityGrade}
        for result in results:
            counts[result.quality_grade.value] += 1
        return counts
