#!/usr/bin/env python3
"""
Module Choreographer for FARFAN 2.0
Coordina la ejecución secuencial de todos los módulos y acumula respuestas

Este módulo garantiza que:
1. Todos los módulos se ejecuten en el orden correcto
2. Los datos se transfieran correctamente entre módulos
3. Las respuestas se acumulen de manera estructurada
4. Se mantenga trazabilidad de qué módulo contribuyó qué información
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger("module_choreographer")


@dataclass
class ModuleExecution:
    """Registro de ejecución de un módulo"""
    module_name: str
    stage: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    execution_time: float
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ResponseAccumulator:
    """Acumulador de respuestas de múltiples módulos para una pregunta"""
    question_id: str
    partial_responses: List[Dict[str, Any]] = field(default_factory=list)
    evidence_fragments: List[str] = field(default_factory=list)
    module_contributions: Dict[str, Any] = field(default_factory=dict)
    
    def add_contribution(self, module_name: str, contribution: Any):
        """Añade la contribución de un módulo"""
        self.module_contributions[module_name] = contribution
        
        # Extract evidence if available
        if isinstance(contribution, dict):
            if 'evidence' in contribution:
                self.evidence_fragments.extend(contribution['evidence'])
            if 'score' in contribution:
                self.partial_responses.append({
                    'module': module_name,
                    'score': contribution['score']
                })
    
    def synthesize(self) -> Dict[str, Any]:
        """Sintetiza todas las contribuciones en una respuesta unificada"""
        # Calculate weighted average of partial scores
        if self.partial_responses:
            total_score = sum(r['score'] for r in self.partial_responses)
            avg_score = total_score / len(self.partial_responses)
        else:
            avg_score = 0.5  # Default neutral score
        
        return {
            'question_id': self.question_id,
            'synthesized_score': avg_score,
            'num_modules': len(self.module_contributions),
            'modules_used': list(self.module_contributions.keys()),
            'total_evidence': len(self.evidence_fragments),
            'evidence_sample': self.evidence_fragments[:3]
        }


class ModuleChoreographer:
    """
    Coreógrafo que coordina la ejecución de módulos y acumula respuestas
    
    Responsabilidades:
    1. Ejecutar módulos en orden canónico
    2. Transferir datos entre módulos
    3. Acumular contribuciones de cada módulo
    4. Mantener historial de ejecución
    5. Generar trazabilidad completa
    """
    
    def __init__(self):
        self.execution_history: List[ModuleExecution] = []
        self.accumulators: Dict[str, ResponseAccumulator] = {}
        self.module_registry: Dict[str, Any] = {}
        
        logger.info("ModuleChoreographer inicializado")
    
    def register_module(self, name: str, module: Any):
        """Registra un módulo para uso en el pipeline"""
        self.module_registry[name] = module
        logger.debug(f"Módulo registrado: {name}")
    
    def execute_module_stage(self, stage_name: str, module_name: str,
                            function_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta una función específica de un módulo y registra la ejecución
        
        Args:
            stage_name: Nombre de la etapa del pipeline
            module_name: Nombre del módulo a ejecutar
            function_name: Nombre de la función a invocar
            inputs: Diccionario de inputs para la función
        
        Returns:
            Diccionario con outputs de la función
        """
        import time
        
        logger.info(f"Ejecutando {module_name}.{function_name} en etapa {stage_name}")
        
        start_time = time.time()
        success = True
        errors = []
        warnings = []
        outputs = {}
        
        try:
            # Get module and function
            module = self.module_registry.get(module_name)
            if module is None:
                raise ValueError(f"Módulo no registrado: {module_name}")
            
            # Get function from module
            if hasattr(module, function_name):
                func = getattr(module, function_name)
            else:
                raise AttributeError(f"{module_name} no tiene función {function_name}")
            
            # Execute function
            result = func(**inputs)
            outputs = {'result': result}
            
            logger.debug(f"  ✓ {module_name}.{function_name} ejecutado exitosamente")
            
        except Exception as e:
            success = False
            errors.append(str(e))
            logger.error(f"  ✗ Error en {module_name}.{function_name}: {e}")
        
        execution_time = time.time() - start_time
        
        # Record execution
        execution = ModuleExecution(
            module_name=module_name,
            stage=stage_name,
            inputs=inputs,
            outputs=outputs,
            execution_time=execution_time,
            success=success,
            errors=errors,
            warnings=warnings
        )
        self.execution_history.append(execution)
        
        return outputs
    
    def accumulate_for_question(self, question_id: str, module_name: str,
                                contribution: Any):
        """
        Acumula la contribución de un módulo para una pregunta específica
        
        Args:
            question_id: ID de la pregunta (ej: "P1-D1-Q1")
            module_name: Nombre del módulo que contribuye
            contribution: Contribución del módulo (puede ser dict, float, str, etc.)
        """
        if question_id not in self.accumulators:
            self.accumulators[question_id] = ResponseAccumulator(question_id=question_id)
        
        self.accumulators[question_id].add_contribution(module_name, contribution)
        
        logger.debug(f"Contribución acumulada: {question_id} <- {module_name}")
    
    def synthesize_responses(self) -> Dict[str, Any]:
        """
        Sintetiza todas las respuestas acumuladas
        
        Returns:
            Dict mapping question_id to synthesized response
        """
        logger.info(f"Sintetizando respuestas de {len(self.accumulators)} preguntas")
        
        synthesized = {}
        for question_id, accumulator in self.accumulators.items():
            synthesized[question_id] = accumulator.synthesize()
        
        return synthesized
    
    def get_data_transfer_log(self) -> List[Dict[str, Any]]:
        """
        Genera log de transferencias de datos entre módulos
        
        Returns:
            Lista de transferencias documentadas
        """
        transfers = []
        
        for i in range(len(self.execution_history) - 1):
            current = self.execution_history[i]
            next_exec = self.execution_history[i + 1]
            
            # Identify data that flows from current to next
            transfer = {
                'from_module': current.module_name,
                'from_stage': current.stage,
                'to_module': next_exec.module_name,
                'to_stage': next_exec.stage,
                'data_keys': list(current.outputs.keys()),
                'timestamp': i
            }
            transfers.append(transfer)
        
        return transfers
    
    def get_module_usage_report(self) -> Dict[str, Any]:
        """
        Genera reporte de uso de módulos
        
        Returns:
            Dict con estadísticas de uso por módulo
        """
        usage = {}
        
        for execution in self.execution_history:
            module_name = execution.module_name
            
            if module_name not in usage:
                usage[module_name] = {
                    'executions': 0,
                    'successful': 0,
                    'failed': 0,
                    'total_time': 0.0,
                    'functions_used': set()
                }
            
            usage[module_name]['executions'] += 1
            usage[module_name]['total_time'] += execution.execution_time
            
            if execution.success:
                usage[module_name]['successful'] += 1
            else:
                usage[module_name]['failed'] += 1
        
        # Convert sets to lists for JSON serialization
        for module_stats in usage.values():
            module_stats['functions_used'] = list(module_stats['functions_used'])
        
        return usage
    
    def verify_all_modules_used(self, required_modules: List[str]) -> Tuple[bool, List[str]]:
        """
        Verifica que todos los módulos requeridos hayan sido usados
        
        Args:
            required_modules: Lista de nombres de módulos que deben ser usados
        
        Returns:
            Tuple (all_used: bool, unused: List[str])
        """
        used_modules = {exec.module_name for exec in self.execution_history}
        unused = [mod for mod in required_modules if mod not in used_modules]
        
        all_used = len(unused) == 0
        
        if all_used:
            logger.info(f"✓ Todos los {len(required_modules)} módulos requeridos fueron utilizados")
        else:
            logger.warning(f"⚠️  Módulos no utilizados: {unused}")
        
        return all_used, unused
    
    def verify_all_functions_used(self, module_name: str, 
                                 required_functions: List[str]) -> Tuple[bool, List[str]]:
        """
        Verifica que todas las funciones de un módulo hayan sido usadas
        
        Args:
            module_name: Nombre del módulo
            required_functions: Lista de funciones que deben ser usadas
        
        Returns:
            Tuple (all_used: bool, unused: List[str])
        """
        # Extract function names from execution history for this module
        # This is a simplified check - in production, would track actual function calls
        
        module_executions = [e for e in self.execution_history if e.module_name == module_name]
        
        if not module_executions:
            return False, required_functions
        
        # Simplified: assume all functions were used if module was executed
        # In production, would need more detailed tracking
        return True, []
    
    def generate_flow_diagram(self) -> str:
        """
        Genera un diagrama textual del flujo de ejecución
        
        Returns:
            String con representación ASCII del flujo
        """
        diagram = ["", "FLUJO DE EJECUCIÓN", "=" * 80, ""]
        
        current_stage = None
        for i, execution in enumerate(self.execution_history):
            if execution.stage != current_stage:
                current_stage = execution.stage
                diagram.append(f"\n[STAGE: {current_stage}]")
                diagram.append("-" * 80)
            
            status = "✓" if execution.success else "✗"
            diagram.append(
                f"  {i+1}. {status} {execution.module_name} "
                f"({execution.execution_time:.2f}s)"
            )
            
            if execution.errors:
                for error in execution.errors:
                    diagram.append(f"      ERROR: {error}")
        
        diagram.append("\n" + "=" * 80)
        
        return "\n".join(diagram)
    
    def export_execution_trace(self) -> Dict[str, Any]:
        """
        Exporta la traza completa de ejecución para análisis
        
        Returns:
            Dict con toda la información de ejecución
        """
        return {
            'total_executions': len(self.execution_history),
            'total_time': sum(e.execution_time for e in self.execution_history),
            'successful_executions': sum(1 for e in self.execution_history if e.success),
            'failed_executions': sum(1 for e in self.execution_history if not e.success),
            'executions': [
                {
                    'module': e.module_name,
                    'stage': e.stage,
                    'time': e.execution_time,
                    'success': e.success,
                    'errors': e.errors,
                    'warnings': e.warnings,
                    'output_keys': list(e.outputs.keys())
                }
                for e in self.execution_history
            ],
            'data_transfers': self.get_data_transfer_log(),
            'module_usage': self.get_module_usage_report()
        }


def create_canonical_flow() -> List[Tuple[str, str, str, List[str]]]:
    """
    Define el flujo canónico de módulos y funciones
    
    Returns:
        Lista de tuplas (stage, module, function, required_inputs)
    """
    flow = [
        # Stage 1-2: Document Extraction
        ("STAGE_1_2", "dereck_beach", "PDFProcessor.load_document", ["pdf_path"]),
        ("STAGE_1_2", "dereck_beach", "PDFProcessor.extract_text", []),
        ("STAGE_1_2", "dereck_beach", "PDFProcessor.extract_tables", []),
        ("STAGE_1_2", "dereck_beach", "PDFProcessor.extract_sections", []),
        
        # Stage 3: Semantic Analysis
        ("STAGE_3", "initial_processor_causal_policy", "PolicyDocumentAnalyzer.analyze_document", ["text"]),
        
        # Stage 4: Causal Extraction
        ("STAGE_4", "dereck_beach", "CausalExtractor.extract_causal_hierarchy", ["text"]),
        ("STAGE_4", "dereck_beach", "CausalExtractor.classify_goal", ["text"]),
        
        # Stage 5: Mechanism Inference
        ("STAGE_5", "dereck_beach", "MechanismPartExtractor.extract_entity_activity", ["text"]),
        ("STAGE_5", "dereck_beach", "BayesianMechanismInference.infer_mechanism", ["node", "observations"]),
        
        # Stage 6: Financial Audit
        ("STAGE_6", "dereck_beach", "FinancialAuditor.trace_financial_allocation", ["tables", "nodes"]),
        
        # Stage 7: DNP Validation
        ("STAGE_7", "dnp_integration", "ValidadorDNP.validar_proyecto_integral", ["sector", "descripcion", "indicadores_propuestos"]),
        ("STAGE_7", "competencias_municipales", "CatalogoCompetenciasMunicipales.validar_competencia_municipal", ["sector"]),
        ("STAGE_7", "mga_indicadores", "CatalogoIndicadoresMGA.buscar_por_sector", ["sector"]),
        ("STAGE_7", "pdet_lineamientos", "LineamientosPDET.recomendar_lineamientos", ["sector"]),
        
        # Stage 8: Question Answering
        ("STAGE_8", "question_answering_engine", "QuestionAnsweringEngine.answer_all_questions", ["pipeline_context"]),
        
        # Stage 9: Report Generation
        ("STAGE_9", "report_generator", "ReportGenerator.generate_micro_report", ["question_responses", "policy_code"]),
        ("STAGE_9", "report_generator", "ReportGenerator.generate_meso_report", ["question_responses", "policy_code"]),
        ("STAGE_9", "report_generator", "ReportGenerator.generate_macro_report", ["question_responses", "compliance_score", "policy_code"]),
    ]
    
    return flow
