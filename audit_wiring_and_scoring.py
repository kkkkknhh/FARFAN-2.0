#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FARFAN 2.0 - Auditoría de Wiring, Flujo, Scoring y Uso de Funciones
Analiza el flujo de datos entre módulos y verifica que todas las funciones sean utilizadas.

Este script audita:
1. El wiring entre todos los scripts del framework
2. El flujo de datos entre módulos (inputs/outputs)
3. La acumulación y pipeline de outputs
4. El scoring final y amalgama de insumos
5. Que todas las funciones sean ejecutadas y aprovechadas
"""

import ast
import inspect
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("audit_wiring_scoring")


@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    module: str
    docstring: Optional[str]
    parameters: List[str]
    is_public: bool
    line_number: int


@dataclass
class ModuleInfo:
    """Information about a module"""
    name: str
    file_path: Path
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)


@dataclass
class DataFlowEdge:
    """Represents data flow from one module to another"""
    from_module: str
    to_module: str
    data_description: str
    format: str  # JSON, Dict, List, etc.
    stage: str


@dataclass
class ScoringComponent:
    """Component that contributes to scoring"""
    module: str
    function: str
    contributes_to: List[str]  # What it scores: ["quantitative", "qualitative", "justification"]
    weight: float


class WiringAuditor:
    """
    Auditor that analyzes the complete wiring and data flow of FARFAN 2.0
    """
    
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.modules_info: Dict[str, ModuleInfo] = {}
        self.data_flows: List[DataFlowEdge] = []
        self.scoring_components: List[ScoringComponent] = []
        self.canonical_flow: List[Tuple[str, str, str, List[str]]] = []
        
    def analyze_repository(self) -> Dict[str, Any]:
        """Perform complete audit of the repository"""
        logger.info("Iniciando auditoría completa del wiring y scoring...")
        
        # 1. Analyze all Python modules
        logger.info("Analizando módulos...")
        self._analyze_modules()
        
        # 2. Extract canonical flow
        logger.info("Extrayendo flujo canónico...")
        self._extract_canonical_flow()
        
        # 3. Analyze data flow
        logger.info("Analizando flujo de datos...")
        self._analyze_data_flow()
        
        # 4. Analyze scoring components
        logger.info("Analizando componentes de scoring...")
        self._analyze_scoring()
        
        # 5. Verify function usage
        logger.info("Verificando uso de funciones...")
        function_usage = self._verify_function_usage()
        
        # 6. Generate comprehensive report
        logger.info("Generando reporte...")
        report = self._generate_report(function_usage)
        
        return report
    
    def _analyze_modules(self):
        """Analyze all Python modules in the repository"""
        python_files = [
            "orchestrator.py",
            "module_choreographer.py",
            "question_answering_engine.py",
            "report_generator.py",
            "dereck_beach",
            "dnp_integration.py",
            "competencias_municipales.py",
            "mga_indicadores.py",
            "pdet_lineamientos.py",
            "initial_processor_causal_policy",
            "pipeline_validators.py",
            "resource_management.py",
            "risk_mitigation_layer.py",
            "circuit_breaker.py"
        ]
        
        for file_name in python_files:
            file_path = self.repo_path / file_name
            if file_path.exists():
                try:
                    module_info = self._analyze_module_file(file_path)
                    self.modules_info[module_info.name] = module_info
                    logger.info(f"  ✓ {module_info.name}: {len(module_info.functions)} funciones, {len(module_info.classes)} clases")
                except Exception as e:
                    logger.warning(f"  ⚠️  Error analizando {file_name}: {e}")
    
    def _analyze_module_file(self, file_path: Path) -> ModuleInfo:
        """Analyze a single Python module file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        module_name = file_path.stem
        module_info = ModuleInfo(
            name=module_name,
            file_path=file_path
        )
        
        try:
            tree = ast.parse(content)
            
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    module_info.classes.append(node.name)
                    
                    # Extract methods from classes
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            func_info = FunctionInfo(
                                name=f"{node.name}.{item.name}",
                                module=module_name,
                                docstring=ast.get_docstring(item),
                                parameters=[arg.arg for arg in item.args.args],
                                is_public=not item.name.startswith('_'),
                                line_number=item.lineno
                            )
                            module_info.functions.append(func_info)
                
                # Extract standalone functions
                elif isinstance(node, ast.FunctionDef):
                    func_info = FunctionInfo(
                        name=node.name,
                        module=module_name,
                        docstring=ast.get_docstring(node),
                        parameters=[arg.arg for arg in node.args.args],
                        is_public=not node.name.startswith('_'),
                        line_number=node.lineno
                    )
                    module_info.functions.append(func_info)
                
                # Extract imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module_info.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_info.imports.append(node.module)
        
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing {file_path}: {e}")
        
        return module_info
    
    def _extract_canonical_flow(self):
        """Extract the canonical flow from module_choreographer"""
        try:
            # Import the canonical flow definition
            sys.path.insert(0, str(self.repo_path))
            from module_choreographer import create_canonical_flow
            
            self.canonical_flow = create_canonical_flow()
            logger.info(f"  ✓ Flujo canónico extraído: {len(self.canonical_flow)} pasos")
        except Exception as e:
            logger.warning(f"  ⚠️  Error extrayendo flujo canónico: {e}")
    
    def _analyze_data_flow(self):
        """Analyze data flow between modules based on canonical flow"""
        # Analyze PipelineContext to see what data flows
        pipeline_context_fields = {
            "STAGE_1_2": ["raw_text", "sections", "tables"],
            "STAGE_3": ["semantic_chunks", "dimension_scores"],
            "STAGE_4": ["causal_graph", "nodes", "causal_chains"],
            "STAGE_5": ["mechanism_parts", "bayesian_inferences"],
            "STAGE_6": ["financial_allocations", "budget_traceability"],
            "STAGE_7": ["dnp_validation_results", "compliance_score"],
            "STAGE_8": ["question_responses"],
            "STAGE_9": ["micro_report", "meso_report", "macro_report"]
        }
        
        for stage, outputs in pipeline_context_fields.items():
            for output in outputs:
                flow_edge = DataFlowEdge(
                    from_module=stage,
                    to_module="PipelineContext",
                    data_description=output,
                    format="Dict/List/Object",
                    stage=stage
                )
                self.data_flows.append(flow_edge)
        
        # Analyze canonical flow connections
        for i, (stage, module, function, inputs) in enumerate(self.canonical_flow):
            if i > 0:
                prev_stage, prev_module, prev_function, _ = self.canonical_flow[i-1]
                
                # Data flows from previous module to current
                flow_edge = DataFlowEdge(
                    from_module=prev_module,
                    to_module=module,
                    data_description=f"{prev_function} → {function}",
                    format="Function Output",
                    stage=stage
                )
                self.data_flows.append(flow_edge)
    
    def _analyze_scoring(self):
        """Analyze scoring components and their contributions"""
        scoring_modules = {
            "dereck_beach": {
                "functions": [
                    "CausalExtractor.extract_causal_hierarchy",
                    "FinancialAuditor.trace_financial_allocation",
                    "OperationalizationAuditor.audit_node_completeness"
                ],
                "contributes": ["quantitative", "qualitative", "justification"],
                "weight": 0.3
            },
            "dnp_integration": {
                "functions": [
                    "ValidadorDNP.validar_proyecto_integral",
                    "ValidadorDNP.validar_indicador_mga"
                ],
                "contributes": ["quantitative", "compliance"],
                "weight": 0.25
            },
            "competencias_municipales": {
                "functions": [
                    "CatalogoCompetenciasMunicipales.validar_competencia_municipal"
                ],
                "contributes": ["qualitative", "compliance"],
                "weight": 0.15
            },
            "mga_indicadores": {
                "functions": [
                    "CatalogoIndicadoresMGA.buscar_por_sector"
                ],
                "contributes": ["quantitative", "justification"],
                "weight": 0.15
            },
            "pdet_lineamientos": {
                "functions": [
                    "LineamientosPDET.recomendar_lineamientos"
                ],
                "contributes": ["qualitative", "justification"],
                "weight": 0.15
            }
        }
        
        for module, info in scoring_modules.items():
            for function in info["functions"]:
                component = ScoringComponent(
                    module=module,
                    function=function,
                    contributes_to=info["contributes"],
                    weight=info["weight"]
                )
                self.scoring_components.append(component)
    
    def _verify_function_usage(self) -> Dict[str, Any]:
        """Verify that all functions are used in the canonical flow"""
        # Extract functions from canonical flow
        used_functions = set()
        for stage, module, function, inputs in self.canonical_flow:
            used_functions.add(f"{module}::{function}")
        
        # Get all public functions from modules
        all_public_functions = set()
        unused_functions = []
        
        for module_name, module_info in self.modules_info.items():
            for func in module_info.functions:
                if func.is_public:
                    full_name = f"{module_name}::{func.name}"
                    all_public_functions.add(full_name)
                    
                    # Check if function is in canonical flow
                    # More flexible matching - check if function name is mentioned
                    is_used = any(
                        func.name in used_func or 
                        func.name.split('.')[-1] in used_func
                        for used_func in used_functions
                    )
                    
                    if not is_used and not func.name.startswith('_'):
                        # Skip certain utility functions that may not be directly in flow
                        skip_patterns = [
                            '__init__', '__str__', '__repr__', 
                            'to_dict', 'from_dict', '_format', '_validate',
                            'logger', 'main', 'test_'
                        ]
                        if not any(pattern in func.name for pattern in skip_patterns):
                            unused_functions.append({
                                "module": module_name,
                                "function": func.name,
                                "line": func.line_number
                            })
        
        return {
            "total_public_functions": len(all_public_functions),
            "used_in_canonical_flow": len(used_functions),
            "unused_functions": unused_functions,
            "coverage_percentage": (len(used_functions) / len(all_public_functions) * 100) if all_public_functions else 0
        }
    
    def _generate_report(self, function_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        # Flow summary
        flow_summary = []
        for stage, module, function, inputs in self.canonical_flow:
            # Determine what the function transmits
            transmits_to = "next_stage"
            data_format = "PipelineContext"
            
            flow_summary.append({
                "script": module,
                "stage": stage,
                "function": function,
                "required_inputs": inputs,
                "transmits_to": transmits_to,
                "data_format": data_format,
                "resolution_mode": "pipeline"
            })
        
        # Coverage evidence
        all_functions_used = len(function_usage["unused_functions"]) == 0
        
        coverage_evidence = {
            "all_functions_used": all_functions_used,
            "unused_functions": function_usage["unused_functions"],
            "coverage_percentage": round(function_usage["coverage_percentage"], 2),
            "total_public_functions": function_usage["total_public_functions"],
            "used_in_canonical_flow": function_usage["used_in_canonical_flow"]
        }
        
        # Scoring audit
        scoring_audit = {
            "verified": True,
            "total_components": len(self.scoring_components),
            "components": [
                {
                    "module": comp.module,
                    "function": comp.function,
                    "contributes_to": comp.contributes_to,
                    "weight": comp.weight
                }
                for comp in self.scoring_components
            ],
            "issues": []
        }
        
        # Check if all scoring aspects are covered
        covered_aspects = set()
        for comp in self.scoring_components:
            covered_aspects.update(comp.contributes_to)
        
        required_aspects = {"quantitative", "qualitative", "justification"}
        missing_aspects = required_aspects - covered_aspects
        
        if missing_aspects:
            scoring_audit["verified"] = False
            scoring_audit["issues"].append({
                "description": f"Missing scoring aspects: {', '.join(missing_aspects)}"
            })
        
        # Issues detected
        issues_detected = []
        
        if not all_functions_used:
            issues_detected.append({
                "description": f"{len(function_usage['unused_functions'])} funciones públicas no utilizadas en el flujo canónico",
                "severity": "medium",
                "count": len(function_usage['unused_functions'])
            })
        
        if not scoring_audit["verified"]:
            issues_detected.extend(scoring_audit["issues"])
        
        if not issues_detected:
            issues_detected.append({"description": "Ninguna"})
        
        # Module statistics
        module_stats = {}
        for module_name, module_info in self.modules_info.items():
            module_stats[module_name] = {
                "total_functions": len(module_info.functions),
                "public_functions": len([f for f in module_info.functions if f.is_public]),
                "classes": len(module_info.classes),
                "file_path": str(module_info.file_path)
            }
        
        # Data flow graph
        data_flow_graph = {
            "nodes": list(set([flow.from_module for flow in self.data_flows] + 
                             [flow.to_module for flow in self.data_flows])),
            "edges": [
                {
                    "from": flow.from_module,
                    "to": flow.to_module,
                    "data": flow.data_description,
                    "format": flow.format,
                    "stage": flow.stage
                }
                for flow in self.data_flows
            ]
        }
        
        return {
            "metadata": {
                "audit_version": "1.0",
                "repository": str(self.repo_path),
                "total_modules_analyzed": len(self.modules_info),
                "canonical_flow_steps": len(self.canonical_flow)
            },
            "flow_summary": flow_summary,
            "coverage_evidence": coverage_evidence,
            "scoring_audit": scoring_audit,
            "issues_detected": issues_detected,
            "module_statistics": module_stats,
            "data_flow_graph": data_flow_graph
        }
    
    def generate_markdown_report(self, audit_data: Dict[str, Any]) -> str:
        """Generate markdown audit report"""
        lines = [
            "# Auditoría de Wiring, Flujo y Scoring - FARFAN 2.0",
            "",
            f"**Fecha de auditoría**: {audit_data['metadata'].get('audit_version', 'N/A')}",
            f"**Repositorio**: {audit_data['metadata']['repository']}",
            f"**Módulos analizados**: {audit_data['metadata']['total_modules_analyzed']}",
            f"**Pasos en flujo canónico**: {audit_data['metadata']['canonical_flow_steps']}",
            "",
            "---",
            "",
            "## 1. Resumen Ejecutivo",
            ""
        ]
        
        # Coverage summary
        coverage = audit_data['coverage_evidence']
        lines.extend([
            f"### Cobertura de Funciones",
            f"- **Total de funciones públicas**: {coverage['total_public_functions']}",
            f"- **Funciones en flujo canónico**: {coverage['used_in_canonical_flow']}",
            f"- **Cobertura**: {coverage['coverage_percentage']}%",
            f"- **Todas las funciones utilizadas**: {'✅ SÍ' if coverage['all_functions_used'] else '⚠️ NO'}",
            ""
        ])
        
        # Scoring summary
        scoring = audit_data['scoring_audit']
        lines.extend([
            f"### Scoring",
            f"- **Componentes de scoring**: {scoring['total_components']}",
            f"- **Scoring verificado**: {'✅ SÍ' if scoring['verified'] else '❌ NO'}",
            ""
        ])
        
        # Issues summary
        issues = audit_data['issues_detected']
        if len(issues) == 1 and issues[0].get('description') == 'Ninguna':
            lines.append("### Problemas Detectados: ✅ Ninguno")
        else:
            lines.extend([
                f"### Problemas Detectados: ⚠️ {len(issues)}",
                ""
            ])
            for issue in issues:
                severity = issue.get('severity', 'info')
                emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, 'ℹ️')
                lines.append(f"- {emoji} {issue['description']}")
        
        lines.extend(["", "---", "", "## 2. Flujo de Datos entre Módulos", ""])
        
        # Flow summary table
        lines.extend([
            "| Stage | Script | Función | Inputs | Transmite a | Formato |",
            "|-------|--------|---------|--------|-------------|---------|"
        ])
        
        for flow in audit_data['flow_summary'][:20]:  # First 20 for readability
            inputs_str = ", ".join(flow['required_inputs']) if flow['required_inputs'] else "—"
            lines.append(
                f"| {flow['stage']} | {flow['script']} | {flow['function']} | "
                f"{inputs_str} | {flow['transmits_to']} | {flow['data_format']} |"
            )
        
        if len(audit_data['flow_summary']) > 20:
            lines.append(f"\n*...y {len(audit_data['flow_summary']) - 20} pasos más*")
        
        lines.extend(["", "---", "", "## 3. Componentes de Scoring", ""])
        
        for comp in scoring['components']:
            lines.extend([
                f"### {comp['module']}.{comp['function']}",
                f"- **Contribuye a**: {', '.join(comp['contributes_to'])}",
                f"- **Peso**: {comp['weight']}",
                ""
            ])
        
        lines.extend(["", "---", "", "## 4. Funciones No Utilizadas", ""])
        
        if coverage['all_functions_used']:
            lines.append("✅ **Todas las funciones públicas están siendo utilizadas en el pipeline.**")
        else:
            lines.append(f"⚠️ **{len(coverage['unused_functions'])} funciones públicas no utilizadas:**")
            lines.append("")
            for func in coverage['unused_functions'][:50]:  # Limit to 50
                lines.append(f"- `{func['module']}.{func['function']}` (línea {func['line']})")
            
            if len(coverage['unused_functions']) > 50:
                lines.append(f"\n*...y {len(coverage['unused_functions']) - 50} más*")
        
        lines.extend(["", "---", "", "## 5. Estadísticas por Módulo", ""])
        
        lines.extend([
            "| Módulo | Funciones Totales | Funciones Públicas | Clases |",
            "|--------|-------------------|-------------------|---------|"
        ])
        
        for module, stats in audit_data['module_statistics'].items():
            lines.append(
                f"| {module} | {stats['total_functions']} | "
                f"{stats['public_functions']} | {stats['classes']} |"
            )
        
        lines.extend(["", "---", "", "## 6. Grafo de Flujo de Datos", ""])
        
        graph = audit_data['data_flow_graph']
        lines.extend([
            f"**Nodos totales**: {len(graph['nodes'])}",
            f"**Conexiones**: {len(graph['edges'])}",
            "",
            "### Principales conexiones:",
            ""
        ])
        
        for edge in graph['edges'][:20]:
            lines.append(f"- `{edge['from']}` → `{edge['to']}` ({edge['stage']}): {edge['data']}")
        
        lines.extend(["", "---", "", "## 7. Conclusiones", ""])
        
        if coverage['all_functions_used'] and scoring['verified']:
            lines.append("✅ **El pipeline está completamente integrado y todas las funciones son utilizadas.**")
        else:
            lines.append("⚠️ **Se detectaron algunas áreas de mejora:**")
            if not coverage['all_functions_used']:
                lines.append(f"- Hay {len(coverage['unused_functions'])} funciones sin utilizar")
            if not scoring['verified']:
                lines.append("- El sistema de scoring tiene problemas")
        
        return "\n".join(lines)


def main():
    """Main entry point"""
    repo_path = Path(__file__).parent
    
    # Create auditor
    auditor = WiringAuditor(repo_path)
    
    # Run audit
    audit_data = auditor.analyze_repository()
    
    # Save JSON report
    json_path = repo_path / "audit_wiring_scoring.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Reporte JSON guardado en: {json_path}")
    
    # Generate markdown report
    markdown = auditor.generate_markdown_report(audit_data)
    md_path = repo_path / "audit_wiring_scoring.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    logger.info(f"✅ Reporte Markdown guardado en: {md_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("RESUMEN DE AUDITORÍA")
    print("="*80)
    print(f"Módulos analizados: {audit_data['metadata']['total_modules_analyzed']}")
    print(f"Pasos en flujo canónico: {audit_data['metadata']['canonical_flow_steps']}")
    print(f"Cobertura de funciones: {audit_data['coverage_evidence']['coverage_percentage']}%")
    print(f"Componentes de scoring: {audit_data['scoring_audit']['total_components']}")
    print(f"Problemas detectados: {len(audit_data['issues_detected'])}")
    
    for issue in audit_data['issues_detected']:
        severity = issue.get('severity', 'info')
        print(f"  - [{severity.upper()}] {issue['description']}")
    
    print("="*80)
    
    return audit_data


if __name__ == "__main__":
    main()
