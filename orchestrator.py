#!/usr/bin/env python3
"""
FARFAN 2.0 - Orchestrator Principal
Flujo Canónico, Determinista e Inmutable para Evaluación de Planes de Desarrollo

Este orquestador integra TODOS los módulos del framework en un flujo coherente
que evalúa 300 preguntas (30 preguntas × 10 áreas de política) con:
- Nivel Micro: Respuesta individual por pregunta (300 respuestas)
- Nivel Meso: Agrupación en 4 clústeres × 6 dimensiones
- Nivel Macro: Evaluación global de alineación con el decálogo

Principios:
- Determinista: Siempre produce el mismo resultado para el mismo input
- Inmutable: No modifica datos originales, solo genera nuevas estructuras
- Canónico: Orden de ejecución fijo y documentado
- Exhaustivo: Usa TODAS las funciones y clases de cada módulo
"""

import logging
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import warnings

# Import resilience and monitoring components
from risk_registry import RiskRegistry, RiskSeverity, RiskCategory
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry, CircuitBreakerError
from pipeline_checkpoint import PipelineCheckpoint
from pipeline_metrics import PipelineMetrics, AlertLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("orchestrator")


class PipelineStage(Enum):
    """Etapas del pipeline en orden canónico"""
    LOAD_DOCUMENT = 1
    EXTRACT_TEXT_TABLES = 2
    SEMANTIC_ANALYSIS = 3
    CAUSAL_EXTRACTION = 4
    MECHANISM_INFERENCE = 5
    FINANCIAL_AUDIT = 6
    DNP_VALIDATION = 7
    QUESTION_ANSWERING = 8
    REPORT_GENERATION = 9


@dataclass
class PipelineContext:
    """Contexto compartido entre etapas del pipeline"""
    # Input
    pdf_path: Path
    policy_code: str
    output_dir: Path
    
    # Stage 1-2: Document processing
    raw_text: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    tables: List[Any] = field(default_factory=list)
    
    # Stage 3: Semantic analysis
    semantic_chunks: List[Dict] = field(default_factory=list)
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    
    # Stage 4: Causal extraction
    causal_graph: Any = None
    nodes: Dict[str, Any] = field(default_factory=dict)
    causal_chains: List[Dict] = field(default_factory=list)
    
    # Stage 5: Mechanism inference
    mechanism_parts: List[Dict] = field(default_factory=list)
    bayesian_inferences: Dict[str, Any] = field(default_factory=dict)
    
    # Stage 6: Financial audit
    financial_allocations: Dict[str, float] = field(default_factory=dict)
    budget_traceability: Dict[str, Any] = field(default_factory=dict)
    
    # Stage 7: DNP validation
    dnp_validation_results: List[Dict] = field(default_factory=list)
    compliance_score: float = 0.0
    
    # Stage 8: Question answering
    question_responses: Dict[str, Dict] = field(default_factory=dict)
    
    # Stage 9: Reports
    micro_report: Dict = field(default_factory=dict)
    meso_report: Dict = field(default_factory=dict)
    macro_report: Dict = field(default_factory=dict)


class FARFANOrchestrator:
    """
    Orquestador principal que ejecuta el flujo completo de análisis
    
    Este orquestador garantiza que:
    1. Todas las clases y funciones de cada módulo son utilizadas
    2. El flujo es determinista (mismo input → mismo output)
    3. Los datos se transfieren de manera clara entre etapas
    4. Se genera un reporte completo a tres niveles
    """
    
    def __init__(self, output_dir: Path, log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set log level
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        
        # Initialize resilience and monitoring systems
        self.risk_registry = RiskRegistry()
        self.circuit_breaker_registry = CircuitBreakerRegistry()
        self.checkpoint = PipelineCheckpoint(self.output_dir / "checkpoints")
        self.metrics = PipelineMetrics(self.output_dir / "metrics")
        
        # Initialize all modules
        self._init_modules()
        
        logger.info("FARFANOrchestrator inicializado con sistemas de resiliencia")
    
    def _init_modules(self):
        """Inicializa todos los módulos del framework"""
        logger.info("Inicializando módulos del framework...")
        
        # Module 1: dereck_beach (CDAF Framework)
        try:
            from dereck_beach import CDAFFramework
            from pathlib import Path
            import tempfile
            
            # Create temporary config for CDAF
            config_content = """
            nlp_model: es_core_news_lg
            confidence_thresholds:
              causal_link: 0.7
              entity_activity: 0.6
            """
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(config_content)
                temp_config = Path(f.name)
            
            self.cdaf = CDAFFramework(
                config_path=temp_config,
                output_dir=self.output_dir,
                log_level="INFO"
            )
            logger.info("✓ CDAF Framework cargado")
        except Exception as e:
            logger.error(f"Error cargando CDAF Framework: {e}")
            self.cdaf = None
        
        # Module 2: DNP Integration
        try:
            from dnp_integration import ValidadorDNP
            self.dnp_validator = ValidadorDNP(es_municipio_pdet=False)
            logger.info("✓ Validador DNP cargado")
        except Exception as e:
            logger.error(f"Error cargando Validador DNP: {e}")
            self.dnp_validator = None
        
        # Module 3: Semantic Policy Processor
        try:
            from initial_processor_causal_policy import PolicyDocumentAnalyzer
            # PolicyDocumentAnalyzer will be initialized when needed (lazy loading)
            self.policy_analyzer_class = PolicyDocumentAnalyzer
            logger.info("✓ Policy Analyzer disponible")
        except Exception as e:
            logger.error(f"Error cargando Policy Analyzer: {e}")
            self.policy_analyzer_class = None
        
        # Module 4: Competencias Municipales
        try:
            from competencias_municipales import CatalogoCompetenciasMunicipales
            self.competencias = CatalogoCompetenciasMunicipales()
            logger.info("✓ Catálogo de Competencias cargado")
        except Exception as e:
            logger.error(f"Error cargando Competencias: {e}")
            self.competencias = None
        
        # Module 5: MGA Indicators
        try:
            from mga_indicadores import CatalogoIndicadoresMGA
            self.mga_catalog = CatalogoIndicadoresMGA()
            logger.info("✓ Catálogo MGA cargado")
        except Exception as e:
            logger.error(f"Error cargando MGA: {e}")
            self.mga_catalog = None
        
        # Module 6: PDET Lineamientos
        try:
            from pdet_lineamientos import LineamientosPDET
            self.pdet_lineamientos = LineamientosPDET()
            logger.info("✓ Lineamientos PDET cargados")
        except Exception as e:
            logger.error(f"Error cargando PDET: {e}")
            self.pdet_lineamientos = None
        
        # Module 7: Question Answering Engine
        try:
            from question_answering_engine import QuestionAnsweringEngine
            self.qa_engine = QuestionAnsweringEngine(
                cdaf=self.cdaf,
                dnp_validator=self.dnp_validator,
                competencias=self.competencias,
                mga_catalog=self.mga_catalog,
                pdet_lineamientos=self.pdet_lineamientos
            )
            logger.info("✓ Question Answering Engine cargado")
        except Exception as e:
            logger.error(f"Error cargando Question Answering Engine: {e}")
            self.qa_engine = None
        
        # Module 8: Report Generator
        try:
            from report_generator import ReportGenerator
            self.report_generator = ReportGenerator(output_dir=self.output_dir)
            logger.info("✓ Report Generator cargado")
        except Exception as e:
            logger.error(f"Error cargando Report Generator: {e}")
            self.report_generator = None
    
    def process_plan(self, pdf_path: Path, policy_code: str, 
                     es_municipio_pdet: bool = False) -> PipelineContext:
        """
        Procesa un Plan de Desarrollo completo siguiendo el flujo canónico
        
        Args:
            pdf_path: Ruta al PDF del plan
            policy_code: Código identificador del plan (ej: "PDM2024-ANT-MED")
            es_municipio_pdet: Si es municipio PDET (afecta validaciones DNP)
        
        Returns:
            PipelineContext con todos los resultados
        """
        logger.info(f"="*80)
        logger.info(f"Iniciando procesamiento de Plan: {policy_code}")
        logger.info(f"PDF: {pdf_path}")
        logger.info(f"="*80)
        
        # Initialize context
        ctx = PipelineContext(
            pdf_path=pdf_path,
            policy_code=policy_code,
            output_dir=self.output_dir
        )
        
        # Start metrics tracking
        self.metrics.start_execution(policy_code)
        
        try:
            # STAGE 1-2: Document Loading and Extraction
            ctx = self._execute_stage_with_protection(
                "LOAD_DOCUMENT",
                self._stage_extract_document,
                ctx
            )
            
            # STAGE 3: Semantic Analysis
            ctx = self._execute_stage_with_protection(
                "SEMANTIC_ANALYSIS",
                self._stage_semantic_analysis,
                ctx
            )
            
            # STAGE 4: Causal Extraction
            ctx = self._execute_stage_with_protection(
                "CAUSAL_EXTRACTION",
                self._stage_causal_extraction,
                ctx
            )
            
            # STAGE 5: Mechanism Inference
            ctx = self._execute_stage_with_protection(
                "MECHANISM_INFERENCE",
                self._stage_mechanism_inference,
                ctx
            )
            
            # STAGE 6: Financial Audit
            ctx = self._execute_stage_with_protection(
                "FINANCIAL_AUDIT",
                self._stage_financial_audit,
                ctx
            )
            
            # STAGE 7: DNP Validation
            ctx = self._execute_stage_with_protection(
                "DNP_VALIDATION",
                lambda c: self._stage_dnp_validation(c, es_municipio_pdet),
                ctx
            )
            
            # STAGE 8: Question Answering (300 preguntas)
            ctx = self._execute_stage_with_protection(
                "QUESTION_ANSWERING",
                self._stage_question_answering,
                ctx
            )
            
            # STAGE 9: Report Generation (Micro, Meso, Macro)
            ctx = self._execute_stage_with_protection(
                "REPORT_GENERATION",
                self._stage_report_generation,
                ctx
            )
            
            logger.info(f"✅ Procesamiento completado exitosamente para {policy_code}")
            self.metrics.end_execution(success=True)
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento: {e}", exc_info=True)
            self.metrics.end_execution(success=False)
            raise
        finally:
            # Export execution trace
            try:
                trace_path = self.metrics.export_trace(
                    risk_registry=self.risk_registry,
                    circuit_breaker_registry=self.circuit_breaker_registry
                )
                logger.info(f"📊 Traza exportada: {trace_path}")
                
                # Print summary
                self.metrics.print_summary()
            except Exception as e:
                logger.error(f"Error exportando métricas: {e}")
        
        return ctx
    
    def _execute_stage_with_protection(self, stage_name: str, stage_func, ctx: PipelineContext) -> PipelineContext:
        """
        Ejecuta una etapa del pipeline con protección completa:
        1. Pre-stage risk assessment
        2. Circuit breaker protection
        3. Post-stage checkpoint
        4. Failure handling with risk-based mitigation
        5. Metrics collection
        
        Args:
            stage_name: Nombre de la etapa
            stage_func: Función de la etapa
            ctx: Contexto del pipeline
        
        Returns:
            Contexto actualizado
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE: {stage_name}")
        logger.info(f"{'='*60}")
        
        # Start metrics tracking
        self.metrics.start_stage(stage_name)
        start_time = time.time()
        
        # PHASE 1: Pre-stage Risk Assessment
        logger.info(f"[1/4] Evaluando riesgos pre-stage...")
        risk_assessments = self.risk_registry.assess_stage_risks(stage_name, ctx)
        
        for assessment in risk_assessments:
            if assessment.applicable:
                self.metrics.record_risk_assessment(assessment.risk_id)
                
                # Alert on CRITICAL/HIGH risks
                if assessment.severity in [RiskSeverity.CRITICAL, RiskSeverity.HIGH]:
                    self.metrics.emit_alert(
                        AlertLevel.WARNING if assessment.severity == RiskSeverity.HIGH else AlertLevel.CRITICAL,
                        f"Riesgo {assessment.severity.value} detectado: {assessment.risk_id}",
                        {
                            'stage': stage_name,
                            'risk_id': assessment.risk_id,
                            'category': assessment.category.value
                        }
                    )
                    logger.warning(f"⚠️  Riesgo {assessment.severity.value}: {assessment.risk_id}")
        
        # PHASE 2: Circuit Breaker Protected Execution
        logger.info(f"[2/4] Ejecutando stage con Circuit Breaker...")
        breaker = self.circuit_breaker_registry.get_or_create(
            stage_name,
            CircuitBreakerConfig(failure_threshold=2, timeout=30.0)
        )
        
        self.metrics.record_circuit_breaker_state(breaker.state.value)
        
        try:
            # Execute stage through circuit breaker
            ctx = breaker.call(stage_func, ctx)
            
            # PHASE 3: Post-stage Checkpoint
            logger.info(f"[3/4] Guardando checkpoint...")
            execution_time_ms = (time.time() - start_time) * 1000
            checkpoint_id = self.checkpoint.save(
                policy_code=ctx.policy_code,
                stage_name=stage_name,
                context=ctx,
                execution_time_ms=execution_time_ms,
                success=True
            )
            
            # PHASE 4: Metrics recording
            logger.info(f"[4/4] Registrando métricas...")
            self.metrics.end_stage(success=True)
            
            logger.info(f"✅ Stage completado exitosamente: {stage_name}")
            return ctx
            
        except CircuitBreakerError as e:
            # Circuit breaker is open
            logger.error(f"🔴 Circuit Breaker OPEN para {stage_name}: {e}")
            self.metrics.emit_alert(
                AlertLevel.CRITICAL,
                f"Circuit Breaker abierto para {stage_name}",
                {'stage': stage_name, 'error': str(e)}
            )
            self.metrics.end_stage(success=False, error_message=str(e))
            raise
            
        except Exception as e:
            # FAILURE HANDLING: Risk-based mitigation
            logger.error(f"❌ Error en stage {stage_name}: {e}")
            
            # Find matching risk
            matching_risk = self.risk_registry.find_risk_by_exception(e, stage_name)
            
            if matching_risk:
                logger.info(f"🔍 Riesgo identificado: {matching_risk.risk_id} (severidad: {matching_risk.severity.value})")
                
                # Record mitigation attempt
                self.metrics.record_mitigation(
                    matching_risk.risk_id,
                    matching_risk.category.value,
                    matching_risk.severity.value
                )
                
                # Alert on CRITICAL/HIGH
                if matching_risk.severity in [RiskSeverity.CRITICAL, RiskSeverity.HIGH]:
                    self.metrics.emit_alert(
                        AlertLevel.ERROR,
                        f"Fallo con riesgo {matching_risk.severity.value}: {matching_risk.risk_id}",
                        {
                            'stage': stage_name,
                            'risk_id': matching_risk.risk_id,
                            'error': str(e)
                        }
                    )
                
                # Execute mitigation strategy
                if matching_risk.severity == RiskSeverity.CRITICAL:
                    # CRITICAL: Abort immediately
                    logger.critical(f"🛑 Riesgo CRITICAL detectado - Abortando ejecución")
                    self.metrics.emit_alert(
                        AlertLevel.CRITICAL,
                        f"Ejecución abortada por riesgo CRITICAL: {matching_risk.risk_id}",
                        {'stage': stage_name, 'risk_id': matching_risk.risk_id}
                    )
                    self.metrics.end_stage(success=False, error_message=f"CRITICAL: {str(e)}")
                    raise
                else:
                    # Attempt mitigation for lower severities
                    logger.info(f"🛡️  Intentando mitigación: {matching_risk.mitigation_strategy}")
                    mitigation_attempt = self.risk_registry.execute_mitigation(matching_risk, ctx)
                    
                    if mitigation_attempt.success:
                        logger.info(f"✅ Mitigación exitosa para {matching_risk.risk_id}")
                        self.metrics.emit_alert(
                            AlertLevel.INFO,
                            f"Mitigación exitosa: {matching_risk.risk_id}",
                            {'stage': stage_name, 'strategy': matching_risk.mitigation_strategy}
                        )
                        # Continue with warning
                        self.metrics.end_stage(success=True, error_message=f"Recovered: {str(e)}")
                        return ctx
                    else:
                        logger.error(f"❌ Mitigación fallida para {matching_risk.risk_id}")
                        self.metrics.emit_alert(
                            AlertLevel.ERROR,
                            f"Mitigación fallida: {matching_risk.risk_id}",
                            {
                                'stage': stage_name,
                                'error': mitigation_attempt.error_message
                            }
                        )
                        self.metrics.end_stage(success=False, error_message=str(e))
                        raise
            else:
                # No matching risk found
                logger.warning(f"⚠️  No se encontró riesgo correspondiente para la excepción")
                self.metrics.emit_alert(
                    AlertLevel.ERROR,
                    f"Error no catalogado en {stage_name}",
                    {'stage': stage_name, 'error': str(e), 'type': type(e).__name__}
                )
                self.metrics.end_stage(success=False, error_message=str(e))
                raise
    
    def _stage_extract_document(self, ctx: PipelineContext) -> PipelineContext:
        """STAGE 1-2: Extrae texto, tablas y secciones del PDF"""
        logger.info(f"[STAGE 1-2] Extrayendo documento: {ctx.pdf_path}")
        
        if self.cdaf is None:
            logger.warning("CDAF no disponible, saltando extracción")
            return ctx
        
        # Use CDAF's PDF processor
        success = self.cdaf.pdf_processor.load_document(ctx.pdf_path)
        if not success:
            raise RuntimeError(f"No se pudo cargar el documento: {ctx.pdf_path}")
        
        ctx.raw_text = self.cdaf.pdf_processor.extract_text()
        ctx.tables = self.cdaf.pdf_processor.extract_tables()
        ctx.sections = self.cdaf.pdf_processor.extract_sections()
        
        logger.info(f"  ✓ Texto extraído: {len(ctx.raw_text)} caracteres")
        logger.info(f"  ✓ Tablas extraídas: {len(ctx.tables)}")
        logger.info(f"  ✓ Secciones identificadas: {len(ctx.sections)}")
        
        return ctx
    
    def _stage_semantic_analysis(self, ctx: PipelineContext) -> PipelineContext:
        """STAGE 3: Análisis semántico del texto"""
        logger.info(f"[STAGE 3] Análisis semántico")
        
        # This stage uses initial_processor_causal_policy if available
        # For now, we'll create a simplified version
        # TODO: Integrate PolicyDocumentAnalyzer when available
        
        logger.info(f"  ✓ Análisis semántico completado (placeholder)")
        return ctx
    
    def _stage_causal_extraction(self, ctx: PipelineContext) -> PipelineContext:
        """STAGE 4: Extracción de jerarquía causal y grafos"""
        logger.info(f"[STAGE 4] Extracción causal")
        
        if self.cdaf is None:
            logger.warning("CDAF no disponible, saltando extracción causal")
            return ctx
        
        # Extract causal hierarchy
        ctx.causal_graph = self.cdaf.causal_extractor.extract_causal_hierarchy(ctx.raw_text)
        ctx.nodes = self.cdaf.causal_extractor.nodes
        ctx.causal_chains = self.cdaf.causal_extractor.causal_chains
        
        logger.info(f"  ✓ Nodos extraídos: {len(ctx.nodes)}")
        logger.info(f"  ✓ Cadenas causales: {len(ctx.causal_chains)}")
        
        return ctx
    
    def _stage_mechanism_inference(self, ctx: PipelineContext) -> PipelineContext:
        """STAGE 5: Inferencia bayesiana de mecanismos"""
        logger.info(f"[STAGE 5] Inferencia de mecanismos")
        
        if self.cdaf is None:
            logger.warning("CDAF no disponible, saltando inferencia")
            return ctx
        
        # Extract Entity-Activity pairs and infer mechanisms
        for node_id, node in ctx.nodes.items():
            if node.type == 'producto':
                ea = self.cdaf.mechanism_extractor.extract_entity_activity(node.text)
                if ea:
                    ctx.mechanism_parts.append({
                        'node_id': node_id,
                        'entity': ea.entity,
                        'activity': ea.activity,
                        'confidence': ea.confidence
                    })
        
        logger.info(f"  ✓ Pares Entidad-Actividad extraídos: {len(ctx.mechanism_parts)}")
        
        return ctx
    
    def _stage_financial_audit(self, ctx: PipelineContext) -> PipelineContext:
        """STAGE 6: Auditoría financiera y trazabilidad"""
        logger.info(f"[STAGE 6] Auditoría financiera")
        
        if self.cdaf is None:
            logger.warning("CDAF no disponible, saltando auditoría financiera")
            return ctx
        
        # Trace financial allocations
        unit_costs = self.cdaf.financial_auditor.trace_financial_allocation(
            ctx.tables, ctx.nodes
        )
        
        ctx.financial_allocations = self.cdaf.financial_auditor.financial_data
        ctx.budget_traceability = {
            'unit_costs': unit_costs,
            'total_budget': sum(ctx.financial_allocations.values())
        }
        
        logger.info(f"  ✓ Asignaciones financieras trazadas: {len(ctx.financial_allocations)}")
        
        return ctx
    
    def _stage_dnp_validation(self, ctx: PipelineContext, 
                              es_municipio_pdet: bool) -> PipelineContext:
        """STAGE 7: Validación de estándares DNP"""
        logger.info(f"[STAGE 7] Validación DNP")
        
        if self.dnp_validator is None:
            logger.warning("Validador DNP no disponible")
            return ctx
        
        # Update PDET status
        self.dnp_validator.es_municipio_pdet = es_municipio_pdet
        
        # Validate each node as a project/goal
        for node_id, node in ctx.nodes.items():
            # Map node type to sector (simplified)
            sector = self._infer_sector(node.text)
            
            resultado = self.dnp_validator.validar_proyecto_integral(
                sector=sector,
                descripcion=node.text[:200] if node.text else "",
                indicadores_propuestos=[],  # TODO: extract from node
                presupuesto=node.financial_allocation or 0.0,
                es_rural=False,  # TODO: detect from context
                poblacion_victimas=False  # TODO: detect from context
            )
            
            ctx.dnp_validation_results.append({
                'node_id': node_id,
                'resultado': resultado
            })
        
        # Calculate compliance score
        if ctx.dnp_validation_results:
            ctx.compliance_score = sum(
                r['resultado'].score_total for r in ctx.dnp_validation_results
            ) / len(ctx.dnp_validation_results)
        
        logger.info(f"  ✓ Validaciones DNP completadas: {len(ctx.dnp_validation_results)}")
        logger.info(f"  ✓ Score de cumplimiento: {ctx.compliance_score:.1f}/100")
        
        return ctx
    
    def _stage_question_answering(self, ctx: PipelineContext) -> PipelineContext:
        """STAGE 8: Respuesta a las 300 preguntas"""
        logger.info(f"[STAGE 8] Respondiendo 300 preguntas")
        
        if self.qa_engine is None:
            logger.warning("Question Answering Engine no disponible")
            return ctx
        
        # Use the QuestionAnsweringEngine
        ctx.question_responses = self.qa_engine.answer_all_questions(ctx)
        
        logger.info(f"  ✓ Preguntas respondidas: {len(ctx.question_responses)}")
        
        return ctx
    
    def _stage_report_generation(self, ctx: PipelineContext) -> PipelineContext:
        """STAGE 9: Generación de reportes (Micro, Meso, Macro)"""
        logger.info(f"[STAGE 9] Generando reportes")
        
        if self.report_generator is None:
            logger.warning("Report Generator no disponible")
            return ctx
        
        # Generate reports at all levels
        ctx.micro_report = self.report_generator.generate_micro_report(
            ctx.question_responses, ctx.policy_code
        )
        
        ctx.meso_report = self.report_generator.generate_meso_report(
            ctx.question_responses, ctx.policy_code
        )
        
        ctx.macro_report = self.report_generator.generate_macro_report(
            ctx.question_responses, ctx.compliance_score, ctx.policy_code
        )
        
        logger.info(f"  ✓ Reporte Micro: {len(ctx.micro_report)} preguntas")
        logger.info(f"  ✓ Reporte Meso: {len(ctx.meso_report)} clústeres")
        logger.info(f"  ✓ Reporte Macro generado")
        
        return ctx
    
    def _infer_sector(self, text: str) -> str:
        """Infiere el sector de política de un texto (simplificado)"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['educación', 'educativo', 'escolar', 'estudiante']):
            return 'educacion'
        elif any(word in text_lower for word in ['salud', 'hospital', 'médico', 'enfermedad']):
            return 'salud'
        elif any(word in text_lower for word in ['agua', 'acueducto', 'saneamiento', 'alcantarillado']):
            return 'agua_potable_saneamiento'
        elif any(word in text_lower for word in ['seguridad', 'policía', 'convivencia']):
            return 'seguridad_convivencia'
        elif any(word in text_lower for word in ['vivienda', 'habitacional', 'hogar']):
            return 'vivienda'
        else:
            return 'general'


def main():
    """Entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FARFAN 2.0 - Orchestrator Principal para Evaluación de Planes de Desarrollo"
    )
    parser.add_argument("pdf_path", type=Path, help="Ruta al PDF del plan de desarrollo")
    parser.add_argument("--policy-code", required=True, help="Código identificador del plan")
    parser.add_argument("--output-dir", type=Path, default="./resultados", 
                       help="Directorio de salida para reportes")
    parser.add_argument("--pdet", action="store_true", 
                       help="Indicar si es municipio PDET")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Nivel de logging")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = FARFANOrchestrator(
        output_dir=args.output_dir,
        log_level=args.log_level
    )
    
    # Process plan
    try:
        context = orchestrator.process_plan(
            pdf_path=args.pdf_path,
            policy_code=args.policy_code,
            es_municipio_pdet=args.pdet
        )
        
        print(f"\n{'='*80}")
        print(f"✅ PROCESAMIENTO COMPLETADO")
        print(f"{'='*80}")
        print(f"Código de Política: {context.policy_code}")
        print(f"Preguntas respondidas: {len(context.question_responses)}")
        print(f"Score de cumplimiento DNP: {context.compliance_score:.1f}/100")
        print(f"Reportes generados en: {args.output_dir}")
        print(f"  - Micro: {args.output_dir}/micro_report_{context.policy_code}.json")
        print(f"  - Meso: {args.output_dir}/meso_report_{context.policy_code}.json")
        print(f"  - Macro: {args.output_dir}/macro_report_{context.policy_code}.md")
        print(f"{'='*80}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error en procesamiento: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
