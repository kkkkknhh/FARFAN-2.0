#!/usr/bin/env python3
"""
Risk Mitigation Layer for FARFAN 2.0
Pre-execution risk assessment and mitigation for pipeline stages

Este módulo implementa:
1. Evaluación de riesgos pre-ejecución mediante predicados detectores
2. Invocación automática de estrategias de mitigación
3. Escalación basada en severidad (CRITICAL→abort, HIGH→retry 1x, MEDIUM→retry 2x, LOW→fallback)
4. Logging estructurado de eventos de riesgo, mitigación y resultados
5. Wrapper para ejecución de etapas con manejo de excepciones
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime

logger = logging.getLogger("risk_mitigation_layer")


class RiskSeverity(Enum):
    """Niveles de severidad de riesgo con escalación definida"""
    CRITICAL = 4  # Abort inmediato
    HIGH = 3      # Retry 1x antes de abort
    MEDIUM = 2    # Retry 2x con fallback
    LOW = 1       # Solo fallback como caso excepcional


class RiskCategory(Enum):
    """Categorías de riesgo específicas por etapa del pipeline"""
    # Stage 1-2: Document Extraction
    PDF_CORRUPTED = "pdf_corrupted"
    PDF_UNREADABLE = "pdf_unreadable"
    MISSING_SECTIONS = "missing_sections"
    EMPTY_DOCUMENT = "empty_document"
    
    # Stage 3: Semantic Analysis
    NLP_MODEL_UNAVAILABLE = "nlp_model_unavailable"
    TEXT_TOO_SHORT = "text_too_short"
    ENCODING_ERROR = "encoding_error"
    
    # Stage 4: Causal Extraction
    NO_CAUSAL_CHAINS = "no_causal_chains"
    GRAPH_DISCONNECTED = "graph_disconnected"
    INSUFFICIENT_NODES = "insufficient_nodes"
    
    # Stage 5: Mechanism Inference
    BAYESIAN_INFERENCE_FAILURE = "bayesian_inference_failure"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    
    # Stage 6: Financial Audit
    MISSING_BUDGET_DATA = "missing_budget_data"
    BUDGET_INCONSISTENCY = "budget_inconsistency"
    NEGATIVE_ALLOCATIONS = "negative_allocations"
    
    # Stage 7: DNP Validation
    DNP_STANDARDS_VIOLATION = "dnp_standards_violation"
    COMPETENCIA_MISMATCH = "competencia_mismatch"
    MISSING_MGA_INDICATORS = "missing_mga_indicators"
    
    # Stage 8: Question Answering
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MODULE_UNAVAILABLE = "module_unavailable"
    
    # Stage 9: Report Generation
    REPORT_GENERATION_FAILURE = "report_generation_failure"
    DATA_SERIALIZATION_ERROR = "data_serialization_error"


@dataclass
class Risk:
    """Definición de un riesgo con su detector y estrategia de mitigación"""
    category: RiskCategory
    severity: RiskSeverity
    probability: float  # 0.0 a 1.0
    impact: float  # 0.0 a 1.0
    detector_predicate: Callable[[Any], bool]  # Función que detecta el riesgo
    mitigation_strategy: Callable[[Any], Any]  # Función que mitiga el riesgo
    description: str = ""
    stage: str = ""
    
    def risk_score(self) -> float:
        """Calcula score de riesgo (probability × impact)"""
        return self.probability * self.impact


@dataclass
class MitigationResult:
    """Resultado de un intento de mitigación"""
    risk: Risk
    success: bool
    attempts: int
    error_message: Optional[str] = None
    outcome_description: str = ""
    mitigation_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CriticalRiskUnmitigatedException(Exception):
    """Excepción para riesgos CRITICAL no mitigados"""
    def __init__(self, risk: Risk, result: MitigationResult):
        self.risk = risk
        self.result = result
        super().__init__(
            f"CRITICAL risk [{risk.category.value}] could not be mitigated. "
            f"Execution aborted. Details: {result.error_message}"
        )


class HighRiskUnmitigatedException(Exception):
    """Excepción para riesgos HIGH no mitigados después de retry"""
    def __init__(self, risk: Risk, result: MitigationResult):
        self.risk = risk
        self.result = result
        super().__init__(
            f"HIGH risk [{risk.category.value}] could not be mitigated after {result.attempts} attempts. "
            f"Execution aborted. Details: {result.error_message}"
        )


class RiskRegistry:
    """
    Registro de riesgos organizados por etapa del pipeline
    
    Permite registrar riesgos con sus detectores y estrategias de mitigación,
    y recuperarlos por etapa para evaluación pre-ejecución.
    """
    
    def __init__(self):
        self.risks_by_stage: Dict[str, List[Risk]] = {}
        logger.info("RiskRegistry inicializado")
    
    def register_risk(self, stage: str, risk: Risk):
        """
        Registra un riesgo para una etapa específica
        
        Args:
            stage: Nombre de la etapa (ej: "STAGE_1_2", "STAGE_3", etc.)
            risk: Objeto Risk con detector y estrategia
        """
        if stage not in self.risks_by_stage:
            self.risks_by_stage[stage] = []
        
        risk.stage = stage
        self.risks_by_stage[stage].append(risk)
        
        logger.debug(
            f"Riesgo registrado: stage={stage}, category={risk.category.value}, "
            f"severity={risk.severity.name}, score={risk.risk_score():.2f}"
        )
    
    def get_risks_for_stage(self, stage: str) -> List[Risk]:
        """
        Recupera todos los riesgos registrados para una etapa
        
        Args:
            stage: Nombre de la etapa
        
        Returns:
            Lista de riesgos (ordenados por severidad descendente)
        """
        risks = self.risks_by_stage.get(stage, [])
        # Ordenar por severidad (CRITICAL primero) y luego por score
        return sorted(risks, key=lambda r: (r.severity.value, r.risk_score()), reverse=True)
    
    def get_all_risks(self) -> Dict[str, List[Risk]]:
        """Retorna todos los riesgos organizados por etapa"""
        return self.risks_by_stage
    
    def get_statistics(self) -> Dict[str, Any]:
        """Genera estadísticas del registro de riesgos"""
        total_risks = sum(len(risks) for risks in self.risks_by_stage.values())
        
        by_severity = {severity: 0 for severity in RiskSeverity}
        for risks in self.risks_by_stage.values():
            for risk in risks:
                by_severity[risk.severity] += 1
        
        return {
            "total_risks": total_risks,
            "stages_covered": len(self.risks_by_stage),
            "by_severity": {sev.name: count for sev, count in by_severity.items()},
            "avg_risks_per_stage": total_risks / len(self.risks_by_stage) if self.risks_by_stage else 0
        }


class RiskMitigationLayer:
    """
    Capa de mitigación de riesgos con evaluación pre-ejecución
    
    Implementa:
    - Evaluación de predicados detectores por etapa
    - Invocación automática de estrategias de mitigación
    - Escalación basada en severidad
    - Logging estructurado de todos los eventos
    """
    
    def __init__(self, registry: RiskRegistry):
        self.registry = registry
        self.mitigation_history: List[MitigationResult] = []
        logger.info("RiskMitigationLayer inicializado")
    
    def assess_stage_risks(self, stage: str, context: Any) -> List[Risk]:
        """
        Evalúa todos los riesgos de una etapa usando sus predicados detectores
        
        Args:
            stage: Nombre de la etapa
            context: Contexto/datos para evaluar (típicamente PipelineContext)
        
        Returns:
            Lista de riesgos detectados (ordenados por severidad)
        """
        logger.info(f"[RISK ASSESSMENT] Evaluando riesgos para etapa: {stage}")
        
        risks = self.registry.get_risks_for_stage(stage)
        detected_risks = []
        
        for risk in risks:
            try:
                is_detected = risk.detector_predicate(context)
                
                if is_detected:
                    detected_risks.append(risk)
                    logger.warning(
                        f"[RISK DETECTED] "
                        f"category={risk.category.value}, "
                        f"severity={risk.severity.name}, "
                        f"probability={risk.probability:.2f}, "
                        f"impact={risk.impact:.2f}, "
                        f"score={risk.risk_score():.2f}, "
                        f"description={risk.description}"
                    )
            except Exception as e:
                logger.error(
                    f"[RISK DETECTION ERROR] Error evaluando detector para {risk.category.value}: {e}"
                )
        
        logger.info(f"[RISK ASSESSMENT] {len(detected_risks)} riesgos detectados de {len(risks)} evaluados")
        
        return detected_risks
    
    def execute_mitigation(self, risk: Risk, context: Any) -> MitigationResult:
        """
        Ejecuta estrategia de mitigación con lógica de retry basada en severidad
        
        Severidad y estrategia:
        - CRITICAL: No retry, abort inmediato si falla
        - HIGH: 1 retry antes de abort
        - MEDIUM: 2 retries con fallback
        - LOW: Solo fallback documentado
        
        Args:
            risk: Riesgo a mitigar
            context: Contexto para la mitigación
        
        Returns:
            MitigationResult con outcome
        """
        logger.info(
            f"[MITIGATION START] "
            f"category={risk.category.value}, "
            f"severity={risk.severity.name}, "
            f"strategy=attempting"
        )
        
        start_time = time.time()
        max_attempts = self._get_max_attempts(risk.severity)
        
        result = MitigationResult(
            risk=risk,
            success=False,
            attempts=0
        )
        
        for attempt in range(1, max_attempts + 1):
            result.attempts = attempt
            
            try:
                logger.info(
                    f"[MITIGATION ATTEMPT] "
                    f"attempt={attempt}/{max_attempts}, "
                    f"category={risk.category.value}"
                )
                
                # Ejecutar estrategia de mitigación
                mitigation_output = risk.mitigation_strategy(context)
                
                # Verificar si la mitigación fue exitosa
                # Re-evaluar el detector
                risk_still_present = risk.detector_predicate(context)
                
                if not risk_still_present:
                    result.success = True
                    result.outcome_description = (
                        f"Mitigación exitosa en intento {attempt}. "
                        f"Output: {mitigation_output}"
                    )
                    
                    logger.info(
                        f"[MITIGATION SUCCESS] "
                        f"category={risk.category.value}, "
                        f"attempt={attempt}/{max_attempts}, "
                        f"outcome={result.outcome_description}"
                    )
                    break
                else:
                    logger.warning(
                        f"[MITIGATION PARTIAL] "
                        f"Intento {attempt} completado pero riesgo persiste"
                    )
                    
            except Exception as e:
                result.error_message = str(e)
                logger.error(
                    f"[MITIGATION FAILED] "
                    f"category={risk.category.value}, "
                    f"attempt={attempt}/{max_attempts}, "
                    f"error={result.error_message}"
                )
                
                # Para CRITICAL, no seguir intentando
                if risk.severity == RiskSeverity.CRITICAL:
                    break
        
        result.mitigation_time = time.time() - start_time
        
        # Log resultado final
        if result.success:
            logger.info(
                f"[MITIGATION COMPLETE] "
                f"category={risk.category.value}, "
                f"success=True, "
                f"attempts={result.attempts}, "
                f"time={result.mitigation_time:.2f}s"
            )
        else:
            logger.error(
                f"[MITIGATION COMPLETE] "
                f"category={risk.category.value}, "
                f"success=False, "
                f"attempts={result.attempts}, "
                f"time={result.mitigation_time:.2f}s, "
                f"final_error={result.error_message}"
            )
        
        # Guardar en historial
        self.mitigation_history.append(result)
        
        return result
    
    def _get_max_attempts(self, severity: RiskSeverity) -> int:
        """Determina número máximo de intentos según severidad"""
        if severity == RiskSeverity.CRITICAL:
            return 1  # Abort inmediato si falla
        elif severity == RiskSeverity.HIGH:
            return 2  # 1 retry (intento inicial + 1)
        elif severity == RiskSeverity.MEDIUM:
            return 3  # 2 retries (intento inicial + 2)
        elif severity == RiskSeverity.LOW:
            return 1  # Solo fallback documentado
        return 1
    
    def wrap_stage_execution(
        self, 
        stage: str, 
        stage_function: Callable[[Any], Any],
        context: Any
    ) -> Any:
        """
        Wrapper para ejecutar una etapa con evaluación de riesgos pre-ejecución
        
        Flujo:
        1. Evaluar riesgos para la etapa
        2. Para cada riesgo detectado, ejecutar mitigación
        3. Si riesgos CRITICAL/HIGH no mitigados → raise exception
        4. Si riesgos MEDIUM mitigados o LOW con fallback → continuar con degradación
        5. Ejecutar función de la etapa
        6. Log resultado final
        
        Args:
            stage: Nombre de la etapa
            stage_function: Función a ejecutar
            context: Contexto del pipeline
        
        Returns:
            Resultado de stage_function
        
        Raises:
            CriticalRiskUnmitigatedException: Si riesgo CRITICAL no mitigado
            HighRiskUnmitigatedException: Si riesgo HIGH no mitigado
        """
        logger.info(f"[STAGE WRAPPER] Iniciando wrapper para etapa: {stage}")
        
        # PASO 1: Evaluación de riesgos pre-ejecución
        detected_risks = self.assess_stage_risks(stage, context)
        
        if not detected_risks:
            logger.info(f"[STAGE WRAPPER] No se detectaron riesgos. Procediendo con ejecución normal.")
        else:
            logger.warning(
                f"[STAGE WRAPPER] {len(detected_risks)} riesgos detectados. "
                f"Iniciando proceso de mitigación."
            )
            
            # PASO 2: Mitigar cada riesgo detectado
            for risk in detected_risks:
                result = self.execute_mitigation(risk, context)
                
                # PASO 3: Decisión según severidad y resultado
                if not result.success:
                    if risk.severity == RiskSeverity.CRITICAL:
                        logger.critical(
                            f"[STAGE WRAPPER] CRITICAL risk no mitigado. ABORTANDO ejecución."
                        )
                        raise CriticalRiskUnmitigatedException(risk, result)
                    
                    elif risk.severity == RiskSeverity.HIGH:
                        logger.error(
                            f"[STAGE WRAPPER] HIGH risk no mitigado después de {result.attempts} intentos. "
                            f"ABORTANDO ejecución."
                        )
                        raise HighRiskUnmitigatedException(risk, result)
                    
                    elif risk.severity == RiskSeverity.MEDIUM:
                        logger.warning(
                            f"[STAGE WRAPPER] MEDIUM risk no completamente mitigado. "
                            f"Continuando con DEGRADACIÓN DOCUMENTADA."
                        )
                        # Documentar degradación en contexto
                        self._document_degradation(context, risk, result)
                    
                    elif risk.severity == RiskSeverity.LOW:
                        logger.info(
                            f"[STAGE WRAPPER] LOW risk. Usando estrategia de fallback documentada."
                        )
                        self._document_degradation(context, risk, result)
                else:
                    logger.info(
                        f"[STAGE WRAPPER] Riesgo {risk.category.value} mitigado exitosamente."
                    )
        
        # PASO 4: Ejecutar función de etapa
        logger.info(f"[STAGE WRAPPER] Ejecutando función de etapa: {stage}")
        
        try:
            stage_start = time.time()
            result = stage_function(context)
            stage_time = time.time() - stage_start
            
            logger.info(
                f"[STAGE WRAPPER] Etapa {stage} completada exitosamente en {stage_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"[STAGE WRAPPER] Error durante ejecución de etapa {stage}: {e}",
                exc_info=True
            )
            raise
    
    def _document_degradation(self, context: Any, risk: Risk, result: MitigationResult):
        """Documenta degradación en el contexto para trazabilidad"""
        if not hasattr(context, 'degradations'):
            context.degradations = []
        
        degradation = {
            'stage': risk.stage,
            'category': risk.category.value,
            'severity': risk.severity.name,
            'description': risk.description,
            'mitigation_attempted': True,
            'mitigation_success': result.success,
            'impact_on_results': 'Results may have reduced quality or completeness',
            'timestamp': result.timestamp
        }
        
        context.degradations.append(degradation)
        
        logger.info(f"[DEGRADATION DOCUMENTED] {risk.category.value} en etapa {risk.stage}")
    
    def get_mitigation_report(self) -> Dict[str, Any]:
        """
        Genera reporte completo de mitigaciones ejecutadas
        
        Returns:
            Dict con estadísticas y detalles de mitigaciones
        """
        if not self.mitigation_history:
            return {
                'total_mitigations': 0,
                'message': 'No se han ejecutado mitigaciones'
            }
        
        successful = sum(1 for r in self.mitigation_history if r.success)
        failed = len(self.mitigation_history) - successful
        
        by_severity = {}
        for result in self.mitigation_history:
            sev = result.risk.severity.name
            if sev not in by_severity:
                by_severity[sev] = {'total': 0, 'successful': 0, 'failed': 0}
            by_severity[sev]['total'] += 1
            if result.success:
                by_severity[sev]['successful'] += 1
            else:
                by_severity[sev]['failed'] += 1
        
        total_time = sum(r.mitigation_time for r in self.mitigation_history)
        avg_time = total_time / len(self.mitigation_history)
        
        return {
            'total_mitigations': len(self.mitigation_history),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(self.mitigation_history) if self.mitigation_history else 0,
            'total_time': total_time,
            'average_time': avg_time,
            'by_severity': by_severity,
            'details': [
                {
                    'category': r.risk.category.value,
                    'severity': r.risk.severity.name,
                    'success': r.success,
                    'attempts': r.attempts,
                    'time': r.mitigation_time,
                    'outcome': r.outcome_description,
                    'timestamp': r.timestamp
                }
                for r in self.mitigation_history
            ]
        }


# ==================== FUNCIONES DE UTILIDAD ====================

def create_default_risk_registry() -> RiskRegistry:
    """
    Crea un RiskRegistry con riesgos predefinidos comunes
    
    Returns:
        RiskRegistry con riesgos configurados para todas las etapas
    """
    registry = RiskRegistry()
    
    # STAGE 1-2: Document Extraction
    registry.register_risk(
        "STAGE_1_2",
        Risk(
            category=RiskCategory.EMPTY_DOCUMENT,
            severity=RiskSeverity.CRITICAL,
            probability=0.1,
            impact=1.0,
            detector_predicate=lambda ctx: len(getattr(ctx, 'raw_text', '')) < 100,
            mitigation_strategy=lambda ctx: "Cannot mitigate empty document - manual intervention required",
            description="Documento vacío o con menos de 100 caracteres"
        )
    )
    
    registry.register_risk(
        "STAGE_1_2",
        Risk(
            category=RiskCategory.MISSING_SECTIONS,
            severity=RiskSeverity.MEDIUM,
            probability=0.3,
            impact=0.6,
            detector_predicate=lambda ctx: len(getattr(ctx, 'sections', {})) < 3,
            mitigation_strategy=lambda ctx: "Using full text analysis as fallback",
            description="Pocas secciones identificadas en el documento"
        )
    )
    
    # STAGE 4: Causal Extraction
    registry.register_risk(
        "STAGE_4",
        Risk(
            category=RiskCategory.NO_CAUSAL_CHAINS,
            severity=RiskSeverity.HIGH,
            probability=0.2,
            impact=0.9,
            detector_predicate=lambda ctx: len(getattr(ctx, 'causal_chains', [])) == 0,
            mitigation_strategy=lambda ctx: "Attempting alternative causal extraction methods",
            description="No se encontraron cadenas causales en el texto"
        )
    )
    
    registry.register_risk(
        "STAGE_4",
        Risk(
            category=RiskCategory.INSUFFICIENT_NODES,
            severity=RiskSeverity.MEDIUM,
            probability=0.25,
            impact=0.7,
            detector_predicate=lambda ctx: len(getattr(ctx, 'nodes', {})) < 5,
            mitigation_strategy=lambda ctx: "Using text-based fallback for node extraction",
            description="Número insuficiente de nodos en el grafo causal"
        )
    )
    
    # STAGE 6: Financial Audit
    registry.register_risk(
        "STAGE_6",
        Risk(
            category=RiskCategory.MISSING_BUDGET_DATA,
            severity=RiskSeverity.MEDIUM,
            probability=0.4,
            impact=0.5,
            detector_predicate=lambda ctx: len(getattr(ctx, 'financial_allocations', {})) == 0,
            mitigation_strategy=lambda ctx: "Proceeding with qualitative analysis only",
            description="No se encontraron datos presupuestarios en el documento"
        )
    )
    
    # STAGE 8: Question Answering
    registry.register_risk(
        "STAGE_8",
        Risk(
            category=RiskCategory.INSUFFICIENT_EVIDENCE,
            severity=RiskSeverity.LOW,
            probability=0.5,
            impact=0.4,
            detector_predicate=lambda ctx: len(getattr(ctx, 'raw_text', '')) < 5000,
            mitigation_strategy=lambda ctx: "Using conservative scoring with documented uncertainty",
            description="Evidencia limitada para responder preguntas con alta confianza"
        )
    )
    
    logger.info(f"RiskRegistry predefinido creado con {len(registry.get_all_risks())} riesgos")
    
    return registry
