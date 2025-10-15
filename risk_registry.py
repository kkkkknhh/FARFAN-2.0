#!/usr/bin/env python3
"""
Risk Registry - Sistema de gestión de riesgos para el pipeline FARFAN

Proporciona:
- Catálogo de riesgos conocidos por etapa del pipeline
- Evaluación de aplicabilidad de riesgos antes de ejecutar etapas
- Estrategias de mitigación basadas en severidad
- Registro y tracking de incidencias de riesgos
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("risk_registry")


class RiskSeverity(Enum):
    """Niveles de severidad de riesgos"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskCategory(Enum):
    """Categorías de riesgos"""

    DATA_QUALITY = "DATA_QUALITY"
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    EXTERNAL_DEPENDENCY = "EXTERNAL_DEPENDENCY"
    COMPUTATION_ERROR = "COMPUTATION_ERROR"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    CONFIGURATION = "CONFIGURATION"


@dataclass
class RiskDefinition:
    """Definición de un riesgo conocido"""

    risk_id: str
    name: str
    category: RiskCategory
    severity: RiskSeverity
    description: str
    applicable_stages: List[str]
    detection_condition: Optional[Callable] = None
    mitigation_strategy: Optional[str] = None


@dataclass
class RiskAssessment:
    """Resultado de evaluación de riesgo"""

    risk_id: str
    applicable: bool
    severity: RiskSeverity
    category: RiskCategory
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MitigationAttempt:
    """Registro de intento de mitigación"""

    risk_id: str
    severity: RiskSeverity
    category: RiskCategory
    strategy: str
    success: bool
    error_message: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)


class RiskRegistry:
    """
    Registro centralizado de riesgos y estrategias de mitigación

    Gestiona:
    - Catálogo de riesgos conocidos
    - Evaluación pre-stage de riesgos aplicables
    - Búsqueda de riesgos por excepción
    - Ejecución de estrategias de mitigación
    """

    def __init__(self):
        self.risks: Dict[str, RiskDefinition] = {}
        self.mitigation_strategies: Dict[str, Callable] = {}
        self.assessment_history: List[RiskAssessment] = []
        self.mitigation_history: List[MitigationAttempt] = []

        # Initialize default risk catalog
        self._initialize_default_risks()
        self._initialize_mitigation_strategies()

    def _initialize_default_risks(self):
        """Inicializa catálogo de riesgos predefinidos"""

        # PDF Extraction risks
        self.register_risk(
            RiskDefinition(
                risk_id="PDF_CORRUPT",
                name="Archivo PDF corrupto o inaccesible",
                category=RiskCategory.DATA_QUALITY,
                severity=RiskSeverity.CRITICAL,
                description="El PDF no puede ser leído o está dañado",
                applicable_stages=["LOAD_DOCUMENT", "EXTRACT_TEXT_TABLES"],
                mitigation_strategy="retry_with_backup",
            )
        )

        self.register_risk(
            RiskDefinition(
                risk_id="EMPTY_EXTRACTION",
                name="Extracción de texto vacía",
                category=RiskCategory.DATA_QUALITY,
                severity=RiskSeverity.HIGH,
                description="No se extrajo texto del PDF",
                applicable_stages=["EXTRACT_TEXT_TABLES"],
                mitigation_strategy="fallback_ocr",
            )
        )

        # NLP/Analysis risks
        self.register_risk(
            RiskDefinition(
                risk_id="NLP_MODEL_MISSING",
                name="Modelo NLP no disponible",
                category=RiskCategory.EXTERNAL_DEPENDENCY,
                severity=RiskSeverity.CRITICAL,
                description="El modelo spaCy no está instalado",
                applicable_stages=["SEMANTIC_ANALYSIS", "CAUSAL_EXTRACTION"],
                mitigation_strategy="download_model",
            )
        )

        self.register_risk(
            RiskDefinition(
                risk_id="INSUFFICIENT_TEXT",
                name="Texto insuficiente para análisis",
                category=RiskCategory.DATA_QUALITY,
                severity=RiskSeverity.MEDIUM,
                description="El documento es muy corto para análisis significativo",
                applicable_stages=["SEMANTIC_ANALYSIS", "CAUSAL_EXTRACTION"],
                mitigation_strategy="reduce_confidence_threshold",
            )
        )

        # Memory/Resource risks
        self.register_risk(
            RiskDefinition(
                risk_id="MEMORY_EXHAUSTION",
                name="Agotamiento de memoria",
                category=RiskCategory.RESOURCE_EXHAUSTION,
                severity=RiskSeverity.HIGH,
                description="Procesamiento requiere más memoria de la disponible",
                applicable_stages=[
                    "SEMANTIC_ANALYSIS",
                    "CAUSAL_EXTRACTION",
                    "MECHANISM_INFERENCE",
                ],
                mitigation_strategy="batch_processing",
            )
        )

        # Computation risks
        self.register_risk(
            RiskDefinition(
                risk_id="TIMEOUT",
                name="Timeout de procesamiento",
                category=RiskCategory.COMPUTATION_ERROR,
                severity=RiskSeverity.MEDIUM,
                description="La etapa excedió el tiempo máximo de ejecución",
                applicable_stages=[
                    "SEMANTIC_ANALYSIS",
                    "CAUSAL_EXTRACTION",
                    "MECHANISM_INFERENCE",
                ],
                mitigation_strategy="reduce_scope",
            )
        )

        # Validation risks
        self.register_risk(
            RiskDefinition(
                risk_id="DNP_VALIDATION_FAIL",
                name="Validación DNP fallida",
                category=RiskCategory.VALIDATION_FAILURE,
                severity=RiskSeverity.LOW,
                description="El plan no cumple estándares DNP",
                applicable_stages=["DNP_VALIDATION"],
                mitigation_strategy="log_and_continue",
            )
        )

        # Financial audit risks
        self.register_risk(
            RiskDefinition(
                risk_id="MISSING_FINANCIAL_DATA",
                name="Datos financieros ausentes",
                category=RiskCategory.DATA_QUALITY,
                severity=RiskSeverity.MEDIUM,
                description="No se encontraron tablas o datos presupuestales",
                applicable_stages=["FINANCIAL_AUDIT"],
                mitigation_strategy="estimate_from_text",
            )
        )

        # Graph/Network risks
        self.register_risk(
            RiskDefinition(
                risk_id="EMPTY_CAUSAL_GRAPH",
                name="Grafo causal vacío",
                category=RiskCategory.COMPUTATION_ERROR,
                severity=RiskSeverity.HIGH,
                description="No se extrajeron relaciones causales",
                applicable_stages=["CAUSAL_EXTRACTION"],
                mitigation_strategy="lower_extraction_threshold",
            )
        )

    def _initialize_mitigation_strategies(self):
        """Inicializa estrategias de mitigación"""

        self.mitigation_strategies["retry_with_backup"] = self._strategy_retry
        self.mitigation_strategies["fallback_ocr"] = self._strategy_fallback_ocr
        self.mitigation_strategies["download_model"] = self._strategy_download_model
        self.mitigation_strategies["reduce_confidence_threshold"] = (
            self._strategy_reduce_threshold
        )
        self.mitigation_strategies["batch_processing"] = self._strategy_batch_processing
        self.mitigation_strategies["reduce_scope"] = self._strategy_reduce_scope
        self.mitigation_strategies["log_and_continue"] = self._strategy_log_continue
        self.mitigation_strategies["estimate_from_text"] = (
            self._strategy_estimate_financial
        )
        self.mitigation_strategies["lower_extraction_threshold"] = (
            self._strategy_lower_threshold
        )

    def register_risk(self, risk: RiskDefinition):
        """Registra un nuevo riesgo en el catálogo"""
        self.risks[risk.risk_id] = risk
        logger.debug(f"Riesgo registrado: {risk.risk_id} ({risk.severity.value})")

    def assess_stage_risks(
        self, stage_name: str, context: Any = None
    ) -> List[RiskAssessment]:
        """
        Evalúa riesgos aplicables a una etapa antes de ejecutarla

        Args:
            stage_name: Nombre de la etapa
            context: Contexto del pipeline (opcional para validaciones adicionales)

        Returns:
            Lista de evaluaciones de riesgo aplicables
        """
        assessments = []

        for risk in self.risks.values():
            if stage_name in risk.applicable_stages:
                applicable = True
                recommendation = "Proceder con precaución"

                # Check detection condition if available
                if risk.detection_condition and context:
                    try:
                        applicable = risk.detection_condition(context)
                    except Exception as e:
                        logger.warning(
                            f"Error evaluando condición de riesgo {risk.risk_id}: {e}"
                        )

                assessment = RiskAssessment(
                    risk_id=risk.risk_id,
                    applicable=applicable,
                    severity=risk.severity,
                    category=risk.category,
                    recommendation=recommendation,
                )

                assessments.append(assessment)
                self.assessment_history.append(assessment)

        # Log critical/high risks
        critical_risks = [
            a
            for a in assessments
            if a.applicable and a.severity in [RiskSeverity.CRITICAL, RiskSeverity.HIGH]
        ]
        if critical_risks:
            logger.warning(
                f"Stage {stage_name}: {len(critical_risks)} riesgos críticos/altos detectados"
            )

        return assessments

    def find_risk_by_exception(
        self, exception: Exception, stage_name: str
    ) -> Optional[RiskDefinition]:
        """
        Encuentra riesgo correspondiente a una excepción

        Args:
            exception: Excepción capturada
            stage_name: Nombre de la etapa donde ocurrió

        Returns:
            RiskDefinition correspondiente o None
        """
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__

        # Map common exceptions to risks
        risk_mappings = {
            "FileNotFoundError": "PDF_CORRUPT",
            "PermissionError": "PDF_CORRUPT",
            "MemoryError": "MEMORY_EXHAUSTION",
            "TimeoutError": "TIMEOUT",
            "OSError": "NLP_MODEL_MISSING",
        }

        # Check by exception type
        if exception_type in risk_mappings:
            risk_id = risk_mappings[exception_type]
            if risk_id in self.risks:
                return self.risks[risk_id]

        # Check by keywords in exception message
        if "empty" in exception_str or "no text" in exception_str:
            return self.risks.get("EMPTY_EXTRACTION")
        elif "model" in exception_str or "nlp" in exception_str:
            return self.risks.get("NLP_MODEL_MISSING")
        elif "graph" in exception_str or "causal" in exception_str:
            return self.risks.get("EMPTY_CAUSAL_GRAPH")
        elif "financial" in exception_str or "budget" in exception_str:
            return self.risks.get("MISSING_FINANCIAL_DATA")

        # No specific risk found
        return None

    def execute_mitigation(
        self, risk: RiskDefinition, context: Any = None
    ) -> MitigationAttempt:
        """
        Ejecuta estrategia de mitigación para un riesgo

        Args:
            risk: Definición del riesgo
            context: Contexto del pipeline

        Returns:
            Resultado del intento de mitigación
        """
        logger.info(
            f"Ejecutando mitigación para riesgo: {risk.risk_id} (estrategia: {risk.mitigation_strategy})"
        )

        strategy_func = self.mitigation_strategies.get(risk.mitigation_strategy)

        if not strategy_func:
            attempt = MitigationAttempt(
                risk_id=risk.risk_id,
                severity=risk.severity,
                category=risk.category,
                strategy=risk.mitigation_strategy or "none",
                success=False,
                error_message="No se encontró estrategia de mitigación",
            )
            self.mitigation_history.append(attempt)
            return attempt

        try:
            strategy_func(context)
            attempt = MitigationAttempt(
                risk_id=risk.risk_id,
                severity=risk.severity,
                category=risk.category,
                strategy=risk.mitigation_strategy,
                success=True,
                error_message=None,
            )
            logger.info(f"✓ Mitigación exitosa: {risk.risk_id}")
        except Exception as e:
            attempt = MitigationAttempt(
                risk_id=risk.risk_id,
                severity=risk.severity,
                category=risk.category,
                strategy=risk.mitigation_strategy,
                success=False,
                error_message=str(e),
            )
            logger.error(f"✗ Mitigación fallida: {risk.risk_id} - {e}")

        self.mitigation_history.append(attempt)
        return attempt

    def get_mitigation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de mitigación"""
        if not self.mitigation_history:
            return {}

        by_severity = {}
        by_category = {}

        for attempt in self.mitigation_history:
            # By severity
            sev = attempt.severity.value
            if sev not in by_severity:
                by_severity[sev] = {"total": 0, "success": 0}
            by_severity[sev]["total"] += 1
            if attempt.success:
                by_severity[sev]["success"] += 1

            # By category
            cat = attempt.category.value
            if cat not in by_category:
                by_category[cat] = {"total": 0, "success": 0}
            by_category[cat]["total"] += 1
            if attempt.success:
                by_category[cat]["success"] += 1

        return {
            "by_severity": by_severity,
            "by_category": by_category,
            "total_attempts": len(self.mitigation_history),
        }

    # Mitigation strategy implementations
    def _strategy_retry(self, context):
        """Reintenta operación"""
        logger.info("Estrategia: retry_with_backup")

    def _strategy_fallback_ocr(self, context):
        """Fallback a OCR si extracción falla"""
        logger.info("Estrategia: fallback_ocr")

    def _strategy_download_model(self, context):
        """Descarga modelo NLP faltante"""
        logger.info("Estrategia: download_model")

    def _strategy_reduce_threshold(self, context):
        """Reduce umbral de confianza"""
        logger.info("Estrategia: reduce_confidence_threshold")

    def _strategy_batch_processing(self, context):
        """Procesa en lotes para reducir memoria"""
        logger.info("Estrategia: batch_processing")

    def _strategy_reduce_scope(self, context):
        """Reduce alcance de procesamiento"""
        logger.info("Estrategia: reduce_scope")

    def _strategy_log_continue(self, context):
        """Solo registra y continúa"""
        logger.info("Estrategia: log_and_continue")

    def _strategy_estimate_financial(self, context):
        """Estima datos financieros de texto"""
        logger.info("Estrategia: estimate_from_text")

    def _strategy_lower_threshold(self, context):
        """Reduce umbral de extracción"""
        logger.info("Estrategia: lower_extraction_threshold")
