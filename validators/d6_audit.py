#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D6 Audit Module - Structural Coherence and Adaptive Learning
=============================================================

Implements Part 4 of the FARFAN 2.0 audit framework, enforcing axiomatic
Theory of Change validation and self-correction per SOTA adaptive causality
(Bennett 2015 on learning cycles).

Audit Points:
- D6-Q1: Axiomatic Validation (5 elements + empty violations)
- D6-Q3: Inconsistency Recognition (<5 causal_incoherence flags)
- D6-Q4: Adaptive M&E System (correction/feedback mechanisms)
- D1-Q5, D6-Q5: Contextual Restrictions (≥3 restriction types)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging
import json

if TYPE_CHECKING:
    import networkx as nx
    from teoria_cambio import TeoriaCambio, ValidacionResultado, CategoriaCausal
    from contradiction_deteccion import PolicyContradictionDetectorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class D6Q1AxiomaticResult:
    """Result for D6-Q1: Axiomatic Validation"""
    has_five_elements: bool
    elements_present: List[str]
    elements_missing: List[str]
    violaciones_orden_empty: bool
    violaciones_orden_count: int
    caminos_completos_exist: bool
    caminos_completos_count: int
    quality_grade: str  # Excelente, Bueno, Regular
    evidence: Dict[str, Any]
    recommendations: List[str]


@dataclass
class D6Q3InconsistencyResult:
    """Result for D6-Q3: Inconsistency Recognition"""
    causal_incoherence_count: int
    total_contradictions: int
    flags_below_limit: bool  # < 5 causal_incoherence
    has_pilot_testing: bool
    pilot_testing_mentions: List[str]
    quality_grade: str  # Excelente, Bueno, Regular
    evidence: Dict[str, Any]
    recommendations: List[str]


@dataclass
class D6Q4AdaptiveMEResult:
    """Result for D6-Q4: Adaptive M&E System"""
    has_correction_mechanism: bool
    has_feedback_mechanism: bool
    mechanism_types_tracked: List[str]
    prior_updates_detected: bool
    learning_loop_evidence: Dict[str, Any]
    uncertainty_reduction: Optional[float]
    quality_grade: str  # Excelente, Bueno, Regular
    evidence: Dict[str, Any]
    recommendations: List[str]


@dataclass
class D1Q5D6Q5RestrictionsResult:
    """Result for D1-Q5, D6-Q5: Contextual Restrictions"""
    restriction_types_detected: List[str]
    restriction_count: int
    meets_minimum_threshold: bool  # ≥ 3 restrictions
    temporal_consistency: bool
    legal_constraints: List[str]
    budgetary_constraints: List[str]
    temporal_constraints: List[str]
    competency_constraints: List[str]
    quality_grade: str  # Excelente, Bueno, Regular
    evidence: Dict[str, Any]
    recommendations: List[str]


@dataclass
class D6AuditReport:
    """Comprehensive D6 Audit Report"""
    timestamp: str
    plan_name: str
    dimension: str
    
    # Individual audit results
    d6_q1_axiomatic: D6Q1AxiomaticResult
    d6_q3_inconsistency: D6Q3InconsistencyResult
    d6_q4_adaptive_me: D6Q4AdaptiveMEResult
    d1_q5_d6_q5_restrictions: D1Q5D6Q5RestrictionsResult
    
    # Overall assessment
    overall_quality: str
    meets_sota_standards: bool
    critical_issues: List[str]
    actionable_recommendations: List[str]
    
    # Metadata
    audit_metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# D6 AUDIT ORCHESTRATOR
# ============================================================================

class D6AuditOrchestrator:
    """
    Orchestrates the D6 audit process, integrating:
    - TeoriaCambio (D6-Q1 structural validation)
    - PolicyContradictionDetectorV2 (D6-Q3 inconsistency recognition)
    - Adaptive learning metrics (D6-Q4)
    - Regulatory constraint analysis (D1-Q5, D6-Q5)
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize D6 audit orchestrator
        
        Args:
            log_dir: Directory for audit logs (default: ./logs/d6_audit)
        """
        self.log_dir = log_dir or Path("./logs/d6_audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component modules (lazy loading)
        self.teoria_cambio = None
        self.contradiction_detector = None
        
        logger.info(f"D6 Audit Orchestrator initialized. Logs: {self.log_dir}")
    
    def execute_full_audit(
        self,
        causal_graph: 'nx.DiGraph',
        text: str,
        plan_name: str = "PDM",
        dimension: str = "estratégico",
        contradiction_results: Optional[Dict[str, Any]] = None,
        prior_history: Optional[List[Dict[str, Any]]] = None
    ) -> D6AuditReport:
        """
        Execute complete D6 audit with all four audit points
        
        Args:
            causal_graph: NetworkX DiGraph with teoria_cambio structure
            text: Full PDM text for analysis
            plan_name: Name of the plan being audited
            dimension: Dimension being analyzed
            contradiction_results: Optional pre-computed contradiction analysis
            prior_history: Optional prior update history for learning loop
            
        Returns:
            D6AuditReport with comprehensive audit results
        """
        logger.info(f"Starting D6 audit for {plan_name} - {dimension}")
        timestamp = datetime.now().isoformat()
        
        # ====================================================================
        # D6-Q1: Axiomatic Validation
        # ====================================================================
        logger.info("Executing D6-Q1: Axiomatic Validation")
        d6_q1_result = self._audit_d6_q1_axiomatic_validation(causal_graph)
        
        # ====================================================================
        # D6-Q3: Inconsistency Recognition
        # ====================================================================
        logger.info("Executing D6-Q3: Inconsistency Recognition")
        d6_q3_result = self._audit_d6_q3_inconsistency_recognition(
            text, plan_name, dimension, contradiction_results
        )
        
        # ====================================================================
        # D6-Q4: Adaptive M&E System
        # ====================================================================
        logger.info("Executing D6-Q4: Adaptive M&E System")
        d6_q4_result = self._audit_d6_q4_adaptive_me_system(
            contradiction_results, prior_history
        )
        
        # ====================================================================
        # D1-Q5, D6-Q5: Contextual Restrictions
        # ====================================================================
        logger.info("Executing D1-Q5, D6-Q5: Contextual Restrictions")
        d1_q5_d6_q5_result = self._audit_d1_q5_d6_q5_contextual_restrictions(
            text, contradiction_results
        )
        
        # ====================================================================
        # Overall Assessment
        # ====================================================================
        overall_quality, meets_sota, critical_issues, recommendations = \
            self._calculate_overall_assessment(
                d6_q1_result, d6_q3_result, d6_q4_result, d1_q5_d6_q5_result
            )
        
        # ====================================================================
        # Compile Final Report
        # ====================================================================
        report = D6AuditReport(
            timestamp=timestamp,
            plan_name=plan_name,
            dimension=dimension,
            d6_q1_axiomatic=d6_q1_result,
            d6_q3_inconsistency=d6_q3_result,
            d6_q4_adaptive_me=d6_q4_result,
            d1_q5_d6_q5_restrictions=d1_q5_d6_q5_result,
            overall_quality=overall_quality,
            meets_sota_standards=meets_sota,
            critical_issues=critical_issues,
            actionable_recommendations=recommendations,
            audit_metadata={
                'total_nodes': causal_graph.number_of_nodes(),
                'total_edges': causal_graph.number_of_edges(),
                'text_length': len(text),
                'audit_timestamp': timestamp
            }
        )
        
        # Save audit report
        self._save_audit_report(report)
        
        logger.info(f"D6 audit completed. Overall quality: {overall_quality}")
        return report
    
    def _audit_d6_q1_axiomatic_validation(
        self,
        causal_graph: 'nx.DiGraph'
    ) -> D6Q1AxiomaticResult:
        """
        D6-Q1: Axiomatic Validation
        
        Check Criteria:
        - TeoriaCambio confirms five elements (INSUMOS-PROCESOS-PRODUCTOS-RESULTADOS-CAUSALIDAD)
        - violaciones_orden empty
        
        Quality Evidence:
        - Run validacion_completa
        - Inspect empty violation list
        
        SOTA Performance:
        - Structural validity matches set-theoretic chains (Goertz 2017)
        - Full elements enable deep inference
        """
        # Initialize TeoriaCambio if not already done
        if self.teoria_cambio is None:
            try:
                from teoria_cambio import TeoriaCambio, CategoriaCausal
                self.teoria_cambio = TeoriaCambio()
            except ImportError as e:
                logger.error(f"Cannot import TeoriaCambio: {e}")
                # Return fallback result
                return D6Q1AxiomaticResult(
                    has_five_elements=False,
                    elements_present=[],
                    elements_missing=['INSUMOS', 'PROCESOS', 'PRODUCTOS', 'RESULTADOS', 'CAUSALIDAD'],
                    violaciones_orden_empty=False,
                    violaciones_orden_count=0,
                    caminos_completos_exist=False,
                    caminos_completos_count=0,
                    quality_grade='Regular',
                    evidence={'error': str(e)},
                    recommendations=['Install teoria_cambio module']
                )
        
        # Run validacion_completa
        from teoria_cambio import CategoriaCausal
        validacion = self.teoria_cambio.validacion_completa(causal_graph)
        
        # Extract elements present
        required_elements = ['INSUMOS', 'PROCESOS', 'PRODUCTOS', 'RESULTADOS', 'CAUSALIDAD']
        elements_present = [
            elem for elem in required_elements
            if elem not in [cat.name for cat in validacion.categorias_faltantes]
        ]
        elements_missing = [cat.name for cat in validacion.categorias_faltantes]
        
        # Check criteria
        has_five_elements = len(elements_present) == 5
        violaciones_orden_empty = len(validacion.violaciones_orden) == 0
        violaciones_orden_count = len(validacion.violaciones_orden)
        caminos_completos_exist = len(validacion.caminos_completos) > 0
        caminos_completos_count = len(validacion.caminos_completos)
        
        # Quality grading
        if has_five_elements and violaciones_orden_empty and caminos_completos_exist:
            quality_grade = 'Excelente'
        elif has_five_elements and violaciones_orden_count <= 2:
            quality_grade = 'Bueno'
        else:
            quality_grade = 'Regular'
        
        # Evidence
        evidence = {
            'es_valida': validacion.es_valida,
            'violaciones_orden': validacion.violaciones_orden,
            'caminos_completos': validacion.caminos_completos,
            'categorias_faltantes': [cat.name for cat in validacion.categorias_faltantes],
            'sugerencias': validacion.sugerencias
        }
        
        # Recommendations
        recommendations = []
        if not has_five_elements:
            recommendations.append(
                f"Agregar elementos faltantes: {', '.join(elements_missing)}"
            )
        if not violaciones_orden_empty:
            recommendations.append(
                f"Corregir {violaciones_orden_count} violaciones de orden causal"
            )
        if not caminos_completos_exist:
            recommendations.append(
                "Establecer al menos un camino completo de INSUMOS a CAUSALIDAD"
            )
        
        recommendations.extend(validacion.sugerencias)
        
        return D6Q1AxiomaticResult(
            has_five_elements=has_five_elements,
            elements_present=elements_present,
            elements_missing=elements_missing,
            violaciones_orden_empty=violaciones_orden_empty,
            violaciones_orden_count=violaciones_orden_count,
            caminos_completos_exist=caminos_completos_exist,
            caminos_completos_count=caminos_completos_count,
            quality_grade=quality_grade,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _audit_d6_q3_inconsistency_recognition(
        self,
        text: str,
        plan_name: str,
        dimension: str,
        contradiction_results: Optional[Dict[str, Any]] = None
    ) -> D6Q3InconsistencyResult:
        """
        D6-Q3: Inconsistency Recognition
        
        Check Criteria:
        - Flags <5 causal_incoherence
        - Rewards pilot/testing plans
        
        Quality Evidence:
        - Count flags in PolicyContradictionDetectorV2
        - Search "plan piloto" in detections
        
        SOTA Performance:
        - Self-reflection per MMR (Lieberman 2015)
        - Low flags indicate Bayesian-tested assumptions
        """
        # Get contradiction results
        if contradiction_results is None:
            # Run contradiction detection
            if self.contradiction_detector is None:
                try:
                    from contradiction_deteccion import (
                        PolicyContradictionDetectorV2,
                        PolicyDimension
                    )
                    self.contradiction_detector = PolicyContradictionDetectorV2()
                except ImportError as e:
                    logger.error(f"Cannot import PolicyContradictionDetectorV2: {e}")
                    # Return fallback result
                    return D6Q3InconsistencyResult(
                        causal_incoherence_count=0,
                        total_contradictions=0,
                        flags_below_limit=True,
                        has_pilot_testing=False,
                        pilot_testing_mentions=[],
                        quality_grade='Regular',
                        evidence={'error': str(e)},
                        recommendations=['Install contradiction_deteccion module']
                    )
            
            # Map dimension string to PolicyDimension
            from contradiction_deteccion import PolicyDimension
            dimension_map = {
                'diagnóstico': PolicyDimension.DIAGNOSTICO,
                'estratégico': PolicyDimension.ESTRATEGICO,
                'programático': PolicyDimension.PROGRAMATICO,
                'financiero': PolicyDimension.FINANCIERO,
                'seguimiento': PolicyDimension.SEGUIMIENTO,
                'territorial': PolicyDimension.TERRITORIAL
            }
            policy_dimension = dimension_map.get(
                dimension.lower(), PolicyDimension.ESTRATEGICO
            )
            
            contradiction_results = self.contradiction_detector.detect(
                text=text,
                plan_name=plan_name,
                dimension=policy_dimension
            )
        
        # Extract causal_incoherence count
        causal_incoherence_count = 0
        total_contradictions = contradiction_results.get('total_contradictions', 0)
        
        # Check harmonic_front_4_audit if available
        if 'harmonic_front_4_audit' in contradiction_results:
            causal_incoherence_count = contradiction_results['harmonic_front_4_audit'].get(
                'causal_incoherence_flags', 0
            )
        else:
            # Count from contradictions list
            contradictions = contradiction_results.get('contradictions', [])
            for contradiction in contradictions:
                if contradiction.get('contradiction_type') == 'CAUSAL_INCOHERENCE':
                    causal_incoherence_count += 1
        
        # Check for pilot/testing mentions
        pilot_patterns = [
            'plan piloto', 'proyecto piloto', 'prueba piloto',
            'fase de prueba', 'implementación piloto', 'pilotaje'
        ]
        pilot_testing_mentions = []
        for pattern in pilot_patterns:
            if pattern.lower() in text.lower():
                # Find context around mention
                import re
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    pilot_testing_mentions.append(context)
        
        has_pilot_testing = len(pilot_testing_mentions) > 0
        
        # Check criteria
        flags_below_limit = causal_incoherence_count < 5
        
        # Quality grading
        if flags_below_limit and has_pilot_testing:
            quality_grade = 'Excelente'
        elif flags_below_limit:
            quality_grade = 'Bueno'
        else:
            quality_grade = 'Regular'
        
        # Evidence
        evidence = {
            'causal_incoherence_count': causal_incoherence_count,
            'total_contradictions': total_contradictions,
            'flags_below_limit': flags_below_limit,
            'has_pilot_testing': has_pilot_testing,
            'pilot_mentions_count': len(pilot_testing_mentions),
            'contradiction_density': contradiction_results.get('coherence_metrics', {}).get(
                'contradiction_density', 0.0
            )
        }
        
        # Recommendations
        recommendations = []
        if not flags_below_limit:
            recommendations.append(
                f"Reducir incoherencias causales (actual: {causal_incoherence_count}, "
                f"objetivo: <5). Revisar cadenas causales circulares."
            )
        if not has_pilot_testing:
            recommendations.append(
                "Incorporar planes piloto o fases de prueba para validar supuestos causales"
            )
        if causal_incoherence_count > 0:
            recommendations.append(
                "Revisar relaciones causales usando análisis bayesiano para reducir incertidumbre"
            )
        
        return D6Q3InconsistencyResult(
            causal_incoherence_count=causal_incoherence_count,
            total_contradictions=total_contradictions,
            flags_below_limit=flags_below_limit,
            has_pilot_testing=has_pilot_testing,
            pilot_testing_mentions=pilot_testing_mentions,
            quality_grade=quality_grade,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _audit_d6_q4_adaptive_me_system(
        self,
        contradiction_results: Optional[Dict[str, Any]] = None,
        prior_history: Optional[List[Dict[str, Any]]] = None
    ) -> D6Q4AdaptiveMEResult:
        """
        D6-Q4: Adaptive M&E System
        
        Check Criteria:
        - Describes correction/feedback
        - Updates mechanism_type_priors from failures
        
        Quality Evidence:
        - Track prior changes in ConfigLoader post-failures
        
        SOTA Performance:
        - Learning loops reduce uncertainty (Humphreys 2015)
        - Adapts like iterative QCA
        """
        # Check for correction mechanism in contradiction results
        has_correction_mechanism = False
        has_feedback_mechanism = False
        mechanism_types_tracked = []
        prior_updates_detected = False
        learning_loop_evidence = {}
        uncertainty_reduction = None
        
        # Check contradiction results for adaptive mechanisms
        if contradiction_results is not None:
            # Check for recommendations (correction mechanism)
            recommendations = contradiction_results.get('recommendations', [])
            if recommendations:
                has_correction_mechanism = True
                learning_loop_evidence['recommendations_count'] = len(recommendations)
            
            # Check harmonic_front_4_audit for adaptive tracking
            if 'harmonic_front_4_audit' in contradiction_results:
                audit = contradiction_results['harmonic_front_4_audit']
                has_feedback_mechanism = 'total_contradictions' in audit
                learning_loop_evidence['audit_metrics'] = audit
        
        # Check prior_history for learning loop evidence
        if prior_history is not None and len(prior_history) > 0:
            prior_updates_detected = True
            
            # Track mechanism types
            mechanism_types = set()
            for entry in prior_history:
                if 'mechanism_type_priors' in entry:
                    mechanism_types.update(entry['mechanism_type_priors'].keys())
            mechanism_types_tracked = list(mechanism_types)
            
            # Calculate uncertainty reduction
            if len(prior_history) >= 2:
                # Compare first and last entries
                first_entry = prior_history[0]
                last_entry = prior_history[-1]
                
                if 'mechanism_type_priors' in first_entry and \
                   'mechanism_type_priors' in last_entry:
                    
                    # Calculate entropy as measure of uncertainty
                    import numpy as np
                    
                    def calculate_entropy(priors):
                        values = list(priors.values())
                        if not values:
                            return 0.0
                        # Normalize
                        total = sum(values)
                        if total == 0:
                            return 0.0
                        probs = [v / total for v in values]
                        # Shannon entropy
                        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
                        return entropy
                    
                    first_entropy = calculate_entropy(first_entry['mechanism_type_priors'])
                    last_entropy = calculate_entropy(last_entry['mechanism_type_priors'])
                    
                    if first_entropy > 0:
                        uncertainty_reduction = (first_entropy - last_entropy) / first_entropy
                        learning_loop_evidence['uncertainty_reduction_pct'] = \
                            uncertainty_reduction * 100
                        learning_loop_evidence['first_entropy'] = first_entropy
                        learning_loop_evidence['last_entropy'] = last_entropy
            
            learning_loop_evidence['prior_history_length'] = len(prior_history)
            learning_loop_evidence['mechanism_types_tracked'] = mechanism_types_tracked
        
        # Quality grading
        if has_correction_mechanism and has_feedback_mechanism and \
           prior_updates_detected and (uncertainty_reduction is not None and uncertainty_reduction >= 0.05):
            quality_grade = 'Excelente'
        elif has_correction_mechanism and has_feedback_mechanism:
            quality_grade = 'Bueno'
        else:
            quality_grade = 'Regular'
        
        # Evidence
        evidence = {
            'has_correction_mechanism': has_correction_mechanism,
            'has_feedback_mechanism': has_feedback_mechanism,
            'prior_updates_detected': prior_updates_detected,
            'learning_loop_evidence': learning_loop_evidence
        }
        
        # Recommendations
        recommendations = []
        if not has_correction_mechanism:
            recommendations.append(
                "Implementar mecanismo de corrección basado en detección de contradicciones"
            )
        if not has_feedback_mechanism:
            recommendations.append(
                "Establecer sistema de retroalimentación para capturar resultados de implementación"
            )
        if not prior_updates_detected:
            recommendations.append(
                "Implementar seguimiento de priors bayesianos para aprendizaje adaptativo"
            )
        if uncertainty_reduction is None or uncertainty_reduction < 0.05:
            recommendations.append(
                "Aumentar iteraciones de calibración para lograr reducción de incertidumbre ≥5%"
            )
        
        return D6Q4AdaptiveMEResult(
            has_correction_mechanism=has_correction_mechanism,
            has_feedback_mechanism=has_feedback_mechanism,
            mechanism_types_tracked=mechanism_types_tracked,
            prior_updates_detected=prior_updates_detected,
            learning_loop_evidence=learning_loop_evidence,
            uncertainty_reduction=uncertainty_reduction,
            quality_grade=quality_grade,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _audit_d1_q5_d6_q5_contextual_restrictions(
        self,
        text: str,
        contradiction_results: Optional[Dict[str, Any]] = None
    ) -> D1Q5D6Q5RestrictionsResult:
        """
        D1-Q5, D6-Q5: Contextual Restrictions
        
        Check Criteria:
        - Analyzes ≥3 restrictions (Legal/Budgetary/Temporal)
        - Adapts to groups
        
        Quality Evidence:
        - Verify TemporalLogicVerifier is_consistent=True
        
        SOTA Performance:
        - Multi-restriction coherence per process-tracing contexts (Beach 2019)
        """
        # Extract regulatory analysis from contradiction results
        legal_constraints = []
        budgetary_constraints = []
        temporal_constraints = []
        competency_constraints = []
        temporal_consistency = True
        
        if contradiction_results is not None and \
           'd1_q5_regulatory_analysis' in contradiction_results:
            
            regulatory = contradiction_results['d1_q5_regulatory_analysis']
            
            # Extract constraint types
            constraint_types_detected = regulatory.get('constraint_types_detected', {})
            
            legal_constraints = [
                ref for ref in regulatory.get('regulatory_references', [])
                if any(pattern in ref.lower() 
                       for pattern in ['ley', 'decreto', 'acuerdo', 'resolución'])
            ]
            
            # Extract budgetary constraints from text patterns
            if constraint_types_detected.get('Budgetary', 0) > 0:
                budgetary_constraints = ['Restricción presupuestal identificada']
            
            # Extract temporal constraints
            if constraint_types_detected.get('Temporal', 0) > 0:
                temporal_constraints = ['Restricción temporal identificada']
            
            # Check temporal consistency
            temporal_consistency = regulatory.get('is_consistent', True)
        
        # Also search text directly for additional constraints
        import re
        
        # Legal patterns
        legal_patterns = [
            r'ley\s+\d+\s+de\s+\d{4}',
            r'decreto\s+\d+',
            r'acuerdo\s+municipal\s+\d+',
            r'competencia\s+municipal'
        ]
        for pattern in legal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            legal_constraints.extend(matches[:3])  # Limit to 3 examples
        
        # Budgetary patterns
        budgetary_patterns = [
            r'restricción\s+presupuestal',
            r'límite\s+fiscal',
            r'capacidad\s+financiera',
            r'recursos\s+disponibles'
        ]
        for pattern in budgetary_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                budgetary_constraints.append(pattern)
        
        # Temporal patterns
        temporal_patterns = [
            r'plazo\s+(?:legal|establecido)',
            r'horizonte\s+temporal',
            r'cuatrienio',
            r'periodo\s+de\s+gobierno'
        ]
        for pattern in temporal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                temporal_constraints.append(pattern)
        
        # Competency patterns
        competency_patterns = [
            r'competencia\s+(?:administrativa|territorial)',
            r'capacidad\s+(?:técnica|institucional)',
            r'ámbito\s+municipal'
        ]
        for pattern in competency_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                competency_constraints.append(pattern)
        
        # Remove duplicates
        legal_constraints = list(set(legal_constraints))
        budgetary_constraints = list(set(budgetary_constraints))
        temporal_constraints = list(set(temporal_constraints))
        competency_constraints = list(set(competency_constraints))
        
        # Count restriction types
        restriction_types_detected = []
        if legal_constraints:
            restriction_types_detected.append('Legal')
        if budgetary_constraints:
            restriction_types_detected.append('Budgetary')
        if temporal_constraints:
            restriction_types_detected.append('Temporal')
        if competency_constraints:
            restriction_types_detected.append('Competency')
        
        restriction_count = len(restriction_types_detected)
        meets_minimum_threshold = restriction_count >= 3
        
        # Quality grading
        if meets_minimum_threshold and temporal_consistency:
            quality_grade = 'Excelente'
        elif meets_minimum_threshold or temporal_consistency:
            quality_grade = 'Bueno'
        else:
            quality_grade = 'Regular'
        
        # Evidence
        evidence = {
            'restriction_types_detected': restriction_types_detected,
            'restriction_count': restriction_count,
            'temporal_consistency': temporal_consistency,
            'legal_constraints_count': len(legal_constraints),
            'budgetary_constraints_count': len(budgetary_constraints),
            'temporal_constraints_count': len(temporal_constraints),
            'competency_constraints_count': len(competency_constraints)
        }
        
        # Recommendations
        recommendations = []
        if not meets_minimum_threshold:
            missing_types = set(['Legal', 'Budgetary', 'Temporal', 'Competency']) - \
                           set(restriction_types_detected)
            recommendations.append(
                f"Documentar al menos 3 tipos de restricciones. Faltantes: {', '.join(missing_types)}"
            )
        if not temporal_consistency:
            recommendations.append(
                "Resolver inconsistencias temporales para garantizar coherencia de restricciones"
            )
        if restriction_count < 4:
            recommendations.append(
                "Fortalecer análisis de restricciones contextuales según Beach (2019)"
            )
        
        return D1Q5D6Q5RestrictionsResult(
            restriction_types_detected=restriction_types_detected,
            restriction_count=restriction_count,
            meets_minimum_threshold=meets_minimum_threshold,
            temporal_consistency=temporal_consistency,
            legal_constraints=legal_constraints[:5],  # Limit to 5 examples
            budgetary_constraints=budgetary_constraints[:5],
            temporal_constraints=temporal_constraints[:5],
            competency_constraints=competency_constraints[:5],
            quality_grade=quality_grade,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _calculate_overall_assessment(
        self,
        d6_q1: D6Q1AxiomaticResult,
        d6_q3: D6Q3InconsistencyResult,
        d6_q4: D6Q4AdaptiveMEResult,
        d1_q5_d6_q5: D1Q5D6Q5RestrictionsResult
    ) -> tuple:
        """
        Calculate overall quality assessment based on individual audit results
        
        Returns:
            (overall_quality, meets_sota_standards, critical_issues, recommendations)
        """
        # Calculate quality score
        quality_scores = {
            'Excelente': 3,
            'Bueno': 2,
            'Regular': 1
        }
        
        scores = [
            quality_scores.get(d6_q1.quality_grade, 0),
            quality_scores.get(d6_q3.quality_grade, 0),
            quality_scores.get(d6_q4.quality_grade, 0),
            quality_scores.get(d1_q5_d6_q5.quality_grade, 0)
        ]
        
        avg_score = sum(scores) / len(scores)
        
        # Overall quality
        if avg_score >= 2.5:
            overall_quality = 'Excelente'
        elif avg_score >= 1.5:
            overall_quality = 'Bueno'
        else:
            overall_quality = 'Regular'
        
        # SOTA standards: all criteria must be at least "Bueno"
        meets_sota_standards = all(score >= 2 for score in scores)
        
        # Critical issues
        critical_issues = []
        if not d6_q1.has_five_elements:
            critical_issues.append(
                "D6-Q1: Estructura de Teoría de Cambio incompleta"
            )
        if not d6_q1.violaciones_orden_empty:
            critical_issues.append(
                f"D6-Q1: {d6_q1.violaciones_orden_count} violaciones de orden causal"
            )
        if not d6_q3.flags_below_limit:
            critical_issues.append(
                f"D6-Q3: {d6_q3.causal_incoherence_count} flags de incoherencia causal (>= 5)"
            )
        if not d6_q4.has_correction_mechanism:
            critical_issues.append(
                "D6-Q4: Falta mecanismo de corrección adaptativo"
            )
        if not d1_q5_d6_q5.meets_minimum_threshold:
            critical_issues.append(
                f"D1-Q5/D6-Q5: Solo {d1_q5_d6_q5.restriction_count} tipos de restricciones (< 3)"
            )
        
        # Consolidated recommendations
        recommendations = []
        recommendations.extend(d6_q1.recommendations[:2])
        recommendations.extend(d6_q3.recommendations[:2])
        recommendations.extend(d6_q4.recommendations[:2])
        recommendations.extend(d1_q5_d6_q5.recommendations[:2])
        
        # Add priority recommendation if not SOTA
        if not meets_sota_standards:
            recommendations.insert(0,
                "PRIORIDAD: Elevar todos los criterios a nivel 'Bueno' o superior para cumplir SOTA"
            )
        
        return overall_quality, meets_sota_standards, critical_issues, recommendations
    
    def _save_audit_report(self, report: D6AuditReport) -> None:
        """Save audit report to JSON file"""
        filename = f"d6_audit_{report.plan_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename
        
        # Convert dataclasses to dict
        report_dict = {
            'timestamp': report.timestamp,
            'plan_name': report.plan_name,
            'dimension': report.dimension,
            'd6_q1_axiomatic': {
                'has_five_elements': report.d6_q1_axiomatic.has_five_elements,
                'elements_present': report.d6_q1_axiomatic.elements_present,
                'elements_missing': report.d6_q1_axiomatic.elements_missing,
                'violaciones_orden_empty': report.d6_q1_axiomatic.violaciones_orden_empty,
                'violaciones_orden_count': report.d6_q1_axiomatic.violaciones_orden_count,
                'caminos_completos_exist': report.d6_q1_axiomatic.caminos_completos_exist,
                'caminos_completos_count': report.d6_q1_axiomatic.caminos_completos_count,
                'quality_grade': report.d6_q1_axiomatic.quality_grade,
                'evidence': report.d6_q1_axiomatic.evidence,
                'recommendations': report.d6_q1_axiomatic.recommendations
            },
            'd6_q3_inconsistency': {
                'causal_incoherence_count': report.d6_q3_inconsistency.causal_incoherence_count,
                'total_contradictions': report.d6_q3_inconsistency.total_contradictions,
                'flags_below_limit': report.d6_q3_inconsistency.flags_below_limit,
                'has_pilot_testing': report.d6_q3_inconsistency.has_pilot_testing,
                'pilot_testing_mentions_count': len(report.d6_q3_inconsistency.pilot_testing_mentions),
                'quality_grade': report.d6_q3_inconsistency.quality_grade,
                'evidence': report.d6_q3_inconsistency.evidence,
                'recommendations': report.d6_q3_inconsistency.recommendations
            },
            'd6_q4_adaptive_me': {
                'has_correction_mechanism': report.d6_q4_adaptive_me.has_correction_mechanism,
                'has_feedback_mechanism': report.d6_q4_adaptive_me.has_feedback_mechanism,
                'mechanism_types_tracked': report.d6_q4_adaptive_me.mechanism_types_tracked,
                'prior_updates_detected': report.d6_q4_adaptive_me.prior_updates_detected,
                'uncertainty_reduction': report.d6_q4_adaptive_me.uncertainty_reduction,
                'quality_grade': report.d6_q4_adaptive_me.quality_grade,
                'evidence': report.d6_q4_adaptive_me.evidence,
                'recommendations': report.d6_q4_adaptive_me.recommendations
            },
            'd1_q5_d6_q5_restrictions': {
                'restriction_types_detected': report.d1_q5_d6_q5_restrictions.restriction_types_detected,
                'restriction_count': report.d1_q5_d6_q5_restrictions.restriction_count,
                'meets_minimum_threshold': report.d1_q5_d6_q5_restrictions.meets_minimum_threshold,
                'temporal_consistency': report.d1_q5_d6_q5_restrictions.temporal_consistency,
                'quality_grade': report.d1_q5_d6_q5_restrictions.quality_grade,
                'evidence': report.d1_q5_d6_q5_restrictions.evidence,
                'recommendations': report.d1_q5_d6_q5_restrictions.recommendations
            },
            'overall_quality': report.overall_quality,
            'meets_sota_standards': report.meets_sota_standards,
            'critical_issues': report.critical_issues,
            'actionable_recommendations': report.actionable_recommendations,
            'audit_metadata': report.audit_metadata
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"D6 audit report saved to: {filepath}")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def execute_d6_audit(
    causal_graph: 'nx.DiGraph',
    text: str,
    plan_name: str = "PDM",
    dimension: str = "estratégico",
    contradiction_results: Optional[Dict[str, Any]] = None,
    prior_history: Optional[List[Dict[str, Any]]] = None,
    log_dir: Optional[Path] = None
) -> D6AuditReport:
    """
    Convenience function to execute D6 audit
    
    Args:
        causal_graph: NetworkX DiGraph with teoria_cambio structure
        text: Full PDM text for analysis
        plan_name: Name of the plan being audited
        dimension: Dimension being analyzed
        contradiction_results: Optional pre-computed contradiction analysis
        prior_history: Optional prior update history for learning loop
        log_dir: Optional directory for audit logs
        
    Returns:
        D6AuditReport with comprehensive audit results
    """
    orchestrator = D6AuditOrchestrator(log_dir=log_dir)
    return orchestrator.execute_full_audit(
        causal_graph=causal_graph,
        text=text,
        plan_name=plan_name,
        dimension=dimension,
        contradiction_results=contradiction_results,
        prior_history=prior_history
    )
