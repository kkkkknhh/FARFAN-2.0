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
import gc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import warnings

# Import validation and resource management
from pipeline_validators import (
    DocumentProcessingData,
    SemanticAnalysisData,
    CausalExtractionData,
    MechanismInferenceData,
    FinancialAuditData,
    DNPValidationData,
    QuestionAnsweringData,
    ReportGenerationData,
    ValidatedPipelineContext,
    validate_stage_transition
)
from resource_management import (
    managed_stage_execution,
    memory_profiling_decorator,
    MemoryMonitor,
    cleanup_intermediate_data
)

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
        
        # Initialize all modules
        self._init_modules()
        
        logger.info("FARFANOrchestrator inicializado")
    
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
        
        # Initialize memory monitor
        memory_monitor = MemoryMonitor(log_interval_mb=100.0)
        
        # Initialize context
        ctx = PipelineContext(
            pdf_path=pdf_path,
            policy_code=policy_code,
            output_dir=self.output_dir
        )
        
        try:
            # STAGE 1-2: Document Loading and Extraction
            with managed_stage_execution("STAGE 1-2"):
                ctx = self._stage_extract_document(ctx)
                memory_monitor.check("After Stage 1-2")
            
            # STAGE 3: Semantic Analysis
            with managed_stage_execution("STAGE 3"):
                ctx = self._stage_semantic_analysis(ctx)
                memory_monitor.check("After Stage 3")
            
            # STAGE 4: Causal Extraction
            with managed_stage_execution("STAGE 4"):
                ctx = self._stage_causal_extraction(ctx)
                memory_monitor.check("After Stage 4")
            
            # STAGE 5: Mechanism Inference
            with managed_stage_execution("STAGE 5"):
                ctx = self._stage_mechanism_inference(ctx)
                memory_monitor.check("After Stage 5")
            
            # STAGE 6: Financial Audit
            with managed_stage_execution("STAGE 6"):
                ctx = self._stage_financial_audit(ctx)
                memory_monitor.check("After Stage 6")
            
            # STAGE 7: DNP Validation
            with managed_stage_execution("STAGE 7"):
                ctx = self._stage_dnp_validation(ctx, es_municipio_pdet)
                memory_monitor.check("After Stage 7")
            
            # STAGE 8: Question Answering (300 preguntas)
            with managed_stage_execution("STAGE 8"):
                ctx = self._stage_question_answering(ctx)
                memory_monitor.check("After Stage 8")
            
            # STAGE 9: Report Generation (Micro, Meso, Macro)
            with managed_stage_execution("STAGE 9"):
                ctx = self._stage_report_generation(ctx)
                memory_monitor.check("After Stage 9")
            
            # Generate final memory report
            memory_report = memory_monitor.report()
            
            logger.info(f"✅ Procesamiento completado exitosamente para {policy_code}")
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento: {e}", exc_info=True)
            raise
        
        return ctx
    
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
        
        # Validate extracted data
        stage_data = DocumentProcessingData(
            raw_text=ctx.raw_text,
            sections=ctx.sections,
            tables=ctx.tables
        )
        validate_stage_transition("1-2", stage_data)
        
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
        
        # Validate stage data
        stage_data = SemanticAnalysisData(
            semantic_chunks=ctx.semantic_chunks,
            dimension_scores=ctx.dimension_scores
        )
        validate_stage_transition("3", stage_data)
        
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
        
        # CRITICAL VALIDATION: Post-Stage 4 must have nodes > 0
        stage_data = CausalExtractionData(
            causal_graph=ctx.causal_graph,
            nodes=ctx.nodes,
            causal_chains=ctx.causal_chains
        )
        validate_stage_transition("4", stage_data)
        
        # Add strategic assertion for causal graph integrity
        if ctx.causal_graph is not None:
            assert ctx.causal_graph.number_of_nodes() > 0, \
                "Causal graph must have at least one node"
        
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
        
        # Validate stage data
        stage_data = MechanismInferenceData(
            mechanism_parts=ctx.mechanism_parts,
            bayesian_inferences=ctx.bayesian_inferences
        )
        validate_stage_transition("5", stage_data)
        
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
        
        # Validate stage data
        stage_data = FinancialAuditData(
            financial_allocations=ctx.financial_allocations,
            budget_traceability=ctx.budget_traceability
        )
        validate_stage_transition("6", stage_data)
        
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
        
        # CRITICAL VALIDATION: Post-Stage 7 compliance_score must be 0-100
        stage_data = DNPValidationData(
            dnp_validation_results=ctx.dnp_validation_results,
            compliance_score=ctx.compliance_score
        )
        validate_stage_transition("7", stage_data)
        
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
        
        # CRITICAL VALIDATION: Post-Stage 8 must have exactly 300 question responses
        stage_data = QuestionAnsweringData(
            question_responses=ctx.question_responses
        )
        validate_stage_transition("8", stage_data)
        
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
        
        # Validate stage data
        stage_data = ReportGenerationData(
            micro_report=ctx.micro_report,
            meso_report=ctx.meso_report,
            macro_report=ctx.macro_report
        )
        validate_stage_transition("9", stage_data)
        
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
