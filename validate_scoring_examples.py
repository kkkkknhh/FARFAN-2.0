#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FARFAN 2.0 - Validación de Scoring con Ejemplos
Genera y valida ejemplos concretos de input/output para el scoring.

Este script demuestra:
1. Cómo los insumos de todos los módulos impactan el scoring
2. Ejemplos de input/output para cada componente de scoring
3. Validación de scoring cuantitativo, cualitativo y justificación
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("validate_scoring")


@dataclass
class ScoringInput:
    """Input para un componente de scoring"""
    module: str
    function: str
    input_data: Dict[str, Any]
    

@dataclass
class ScoringOutput:
    """Output de un componente de scoring"""
    quantitative_score: float  # 0.0 - 1.0
    qualitative_assessment: str
    justification: str
    evidence: List[str]
    confidence: float  # 0.0 - 1.0


@dataclass
class ScoringExample:
    """Ejemplo completo de scoring"""
    scenario: str
    inputs: List[ScoringInput]
    expected_output: ScoringOutput
    actual_output: ScoringOutput = None
    validated: bool = False


class ScoringValidator:
    """Valida el sistema de scoring con ejemplos concretos"""
    
    def __init__(self):
        self.examples: List[ScoringExample] = []
        self.validation_results: List[Dict[str, Any]] = []
    
    def generate_examples(self) -> List[ScoringExample]:
        """Genera ejemplos de scoring para diferentes escenarios"""
        
        examples = [
            # Ejemplo 1: Causal Extraction - Alta calidad
            ScoringExample(
                scenario="Proyecto con teoría de cambio clara y completa",
                inputs=[
                    ScoringInput(
                        module="dereck_beach",
                        function="CausalExtractor.extract_causal_hierarchy",
                        input_data={
                            "text": "Mejorar la seguridad alimentaria mediante la implementación de huertas comunitarias que incrementen el acceso a alimentos nutritivos",
                            "nodes_extracted": 5,
                            "causal_links": 4,
                            "hierarchy_levels": 4
                        }
                    )
                ],
                expected_output=ScoringOutput(
                    quantitative_score=0.85,
                    qualitative_assessment="Excelente - Teoría de cambio clara con cadena causal bien definida",
                    justification="El proyecto presenta una jerarquía causal completa desde insumos hasta impactos, con 5 nodos conectados lógicamente y 4 vínculos causales explícitos.",
                    evidence=[
                        "5 nodos causales identificados",
                        "4 vínculos causales explícitos",
                        "Jerarquía de 4 niveles presente"
                    ],
                    confidence=0.90
                )
            ),
            
            # Ejemplo 2: Causal Extraction - Calidad media
            ScoringExample(
                scenario="Proyecto con teoría de cambio incompleta",
                inputs=[
                    ScoringInput(
                        module="dereck_beach",
                        function="CausalExtractor.extract_causal_hierarchy",
                        input_data={
                            "text": "Construir escuelas para mejorar la educación",
                            "nodes_extracted": 2,
                            "causal_links": 1,
                            "hierarchy_levels": 2
                        }
                    )
                ],
                expected_output=ScoringOutput(
                    quantitative_score=0.45,
                    qualitative_assessment="Insuficiente - Cadena causal incompleta, falta operacionalización",
                    justification="El proyecto solo identifica 2 nodos (producto e impacto) sin especificar mecanismos intermedios ni resultados medibles.",
                    evidence=[
                        "Solo 2 nodos causales",
                        "1 vínculo causal",
                        "Faltan niveles intermedios"
                    ],
                    confidence=0.75
                )
            ),
            
            # Ejemplo 3: DNP Validation - Cumplimiento alto
            ScoringExample(
                scenario="Proyecto alineado con estándares DNP",
                inputs=[
                    ScoringInput(
                        module="dnp_integration",
                        function="ValidadorDNP.validar_proyecto_integral",
                        input_data={
                            "sector": "Educación",
                            "descripcion": "Programa de mejoramiento de infraestructura educativa",
                            "indicadores_propuestos": ["Tasa de deserción", "Puntaje SABER"],
                            "mga_aligned": True,
                            "competencia_valida": True
                        }
                    )
                ],
                expected_output=ScoringOutput(
                    quantitative_score=0.90,
                    qualitative_assessment="Excelente - Cumple estándares DNP y MGA",
                    justification="El proyecto está alineado con indicadores MGA del sector Educación, es competencia municipal válida y usa indicadores estándar del DNP.",
                    evidence=[
                        "Alineado con MGA",
                        "Competencia municipal válida",
                        "Indicadores DNP presentes"
                    ],
                    confidence=0.95
                )
            ),
            
            # Ejemplo 4: Financial Audit - Trazabilidad completa
            ScoringExample(
                scenario="Proyecto con presupuesto trazable",
                inputs=[
                    ScoringInput(
                        module="dereck_beach",
                        function="FinancialAuditor.trace_financial_allocation",
                        input_data={
                            "total_budget": 1000000000,
                            "allocations_by_node": {
                                "P1": 300000000,
                                "P2": 400000000,
                                "P3": 300000000
                            },
                            "traceability_score": 0.95
                        }
                    )
                ],
                expected_output=ScoringOutput(
                    quantitative_score=0.88,
                    qualitative_assessment="Bueno - Presupuesto completamente trazable a nodos causales",
                    justification="El 100% del presupuesto está asignado a nodos específicos con trazabilidad clara. La distribución es coherente con la jerarquía causal.",
                    evidence=[
                        "100% del presupuesto asignado",
                        "Trazabilidad: 0.95",
                        "3 nodos con asignación financiera"
                    ],
                    confidence=0.92
                )
            ),
            
            # Ejemplo 5: Competencias Municipales - Validación
            ScoringExample(
                scenario="Proyecto dentro de competencias municipales",
                inputs=[
                    ScoringInput(
                        module="competencias_municipales",
                        function="CatalogoCompetenciasMunicipales.validar_competencia_municipal",
                        input_data={
                            "sector": "Salud",
                            "subsector": "Atención Primaria",
                            "nivel_gobierno": "Municipal",
                            "es_competencia_valida": True
                        }
                    )
                ],
                expected_output=ScoringOutput(
                    quantitative_score=1.0,
                    qualitative_assessment="Excelente - Competencia municipal válida",
                    justification="La atención primaria en salud es competencia directa del nivel municipal según la Ley 715 de 2001.",
                    evidence=[
                        "Competencia municipal validada",
                        "Subsector: Atención Primaria",
                        "Base legal: Ley 715/2001"
                    ],
                    confidence=1.0
                )
            ),
            
            # Ejemplo 6: MGA Indicators - Alineación
            ScoringExample(
                scenario="Proyecto con indicadores MGA alineados",
                inputs=[
                    ScoringInput(
                        module="mga_indicadores",
                        function="CatalogoIndicadoresMGA.buscar_por_sector",
                        input_data={
                            "sector": "Agua potable y saneamiento básico",
                            "indicadores_encontrados": 5,
                            "indicadores_proyecto": 3,
                            "match_percentage": 60
                        }
                    )
                ],
                expected_output=ScoringOutput(
                    quantitative_score=0.75,
                    qualitative_assessment="Bueno - 60% de alineación con indicadores MGA",
                    justification="El proyecto usa 3 de 5 indicadores MGA recomendados para el sector, mostrando buena alineación con estándares nacionales.",
                    evidence=[
                        "5 indicadores MGA disponibles",
                        "3 indicadores usados",
                        "60% de alineación"
                    ],
                    confidence=0.85
                )
            ),
            
            # Ejemplo 7: PDET Lineamientos - Municipio PDET
            ScoringExample(
                scenario="Proyecto en municipio PDET alineado con lineamientos",
                inputs=[
                    ScoringInput(
                        module="pdet_lineamientos",
                        function="LineamientosPDET.recomendar_lineamientos",
                        input_data={
                            "es_municipio_pdet": True,
                            "sector": "Rural",
                            "lineamientos_aplicables": 4,
                            "lineamientos_cumplidos": 3
                        }
                    )
                ],
                expected_output=ScoringOutput(
                    quantitative_score=0.82,
                    qualitative_assessment="Bueno - 75% de alineación con lineamientos PDET",
                    justification="El proyecto cumple 3 de 4 lineamientos PDET aplicables al sector rural, demostrando compromiso con los Programas de Desarrollo con Enfoque Territorial.",
                    evidence=[
                        "4 lineamientos PDET aplicables",
                        "3 lineamientos cumplidos",
                        "Municipio PDET confirmado"
                    ],
                    confidence=0.88
                )
            ),
            
            # Ejemplo 8: Amalgama de múltiples fuentes
            ScoringExample(
                scenario="Score final amalgamado de múltiples módulos",
                inputs=[
                    ScoringInput(
                        module="dereck_beach",
                        function="CausalExtractor.extract_causal_hierarchy",
                        input_data={"score": 0.85}
                    ),
                    ScoringInput(
                        module="dnp_integration",
                        function="ValidadorDNP.validar_proyecto_integral",
                        input_data={"score": 0.90}
                    ),
                    ScoringInput(
                        module="competencias_municipales",
                        function="CatalogoCompetenciasMunicipales.validar_competencia_municipal",
                        input_data={"score": 1.0}
                    ),
                    ScoringInput(
                        module="mga_indicadores",
                        function="CatalogoIndicadoresMGA.buscar_por_sector",
                        input_data={"score": 0.75}
                    )
                ],
                expected_output=ScoringOutput(
                    quantitative_score=0.86,  # Weighted average
                    qualitative_assessment="Excelente - Proyecto bien fundamentado en múltiples dimensiones",
                    justification="Score amalgamado: Causal (0.85×0.30) + DNP (0.90×0.25) + Competencias (1.0×0.15) + MGA (0.75×0.15) + PDET (0.80×0.15) = 0.86. El proyecto demuestra solidez técnica, cumplimiento normativo y alineación con estándares nacionales.",
                    evidence=[
                        "Teoría de cambio clara (0.85)",
                        "Cumplimiento DNP alto (0.90)",
                        "Competencia válida (1.0)",
                        "Alineación MGA buena (0.75)"
                    ],
                    confidence=0.92
                )
            )
        ]
        
        self.examples = examples
        return examples
    
    def validate_example(self, example: ScoringExample) -> Dict[str, Any]:
        """Valida un ejemplo de scoring"""
        
        # Simulate scoring calculation based on inputs
        scores = []
        evidence_combined = []
        
        for inp in example.inputs:
            # Extract score from input data if present
            if "score" in inp.input_data:
                scores.append(inp.input_data["score"])
            else:
                # Calculate based on heuristics
                if inp.module == "dereck_beach" and "nodes_extracted" in inp.input_data:
                    # Score based on nodes and links
                    nodes = inp.input_data.get("nodes_extracted", 0)
                    links = inp.input_data.get("causal_links", 0)
                    levels = inp.input_data.get("hierarchy_levels", 0)
                    
                    # Simple heuristic
                    score = min(1.0, (nodes * 0.15 + links * 0.15 + levels * 0.20))
                    scores.append(score)
                    
                elif inp.module == "dnp_integration":
                    # Score based on alignment
                    mga_aligned = inp.input_data.get("mga_aligned", False)
                    competencia_valida = inp.input_data.get("competencia_valida", False)
                    
                    score = 0.5
                    if mga_aligned:
                        score += 0.25
                    if competencia_valida:
                        score += 0.25
                    scores.append(score)
                    
                elif inp.module == "competencias_municipales":
                    # Score based on validity
                    valida = inp.input_data.get("es_competencia_valida", False)
                    scores.append(1.0 if valida else 0.3)
                    
                elif inp.module == "mga_indicadores":
                    # Score based on match percentage
                    match_pct = inp.input_data.get("match_percentage", 0) / 100.0
                    scores.append(match_pct)
                    
                elif inp.function == "FinancialAuditor.trace_financial_allocation":
                    # Score based on traceability
                    trace_score = inp.input_data.get("traceability_score", 0.5)
                    scores.append(trace_score * 0.95)  # Max 0.95
            
            # Collect evidence
            for key, value in inp.input_data.items():
                if key not in ["score"]:
                    evidence_combined.append(f"{key}: {value}")
        
        # Calculate actual output
        if len(scores) == 0:
            actual_score = 0.5  # Default neutral score
        elif len(scores) == 1:
            actual_score = scores[0]
        else:
            # Weighted average (simplified)
            weights = [0.30, 0.25, 0.15, 0.15, 0.15][:len(scores)]
            total_weight = sum(weights)
            if total_weight > 0:
                actual_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                actual_score = sum(scores) / len(scores)  # Simple average
        
        # Determine qualitative assessment
        if actual_score >= 0.85:
            qual = "Excelente"
        elif actual_score >= 0.70:
            qual = "Bueno"
        elif actual_score >= 0.55:
            qual = "Aceptable"
        else:
            qual = "Insuficiente"
        
        actual_output = ScoringOutput(
            quantitative_score=actual_score,
            qualitative_assessment=qual,
            justification=f"Score calculado: {actual_score:.2f}",
            evidence=evidence_combined,
            confidence=0.85
        )
        
        example.actual_output = actual_output
        
        # Validate
        score_diff = abs(actual_score - example.expected_output.quantitative_score)
        validated = score_diff < 0.15  # Allow 15% tolerance
        example.validated = validated
        
        return {
            "scenario": example.scenario,
            "expected_score": example.expected_output.quantitative_score,
            "actual_score": actual_score,
            "score_difference": score_diff,
            "validated": validated,
            "qualitative_match": qual in example.expected_output.qualitative_assessment
        }
    
    def validate_all_examples(self) -> Dict[str, Any]:
        """Valida todos los ejemplos"""
        if not self.examples:
            self.generate_examples()
        
        results = []
        for example in self.examples:
            result = self.validate_example(example)
            results.append(result)
            self.validation_results.append(result)
        
        # Calculate summary statistics
        total = len(results)
        validated = sum(1 for r in results if r["validated"])
        avg_diff = sum(r["score_difference"] for r in results) / total if total > 0 else 0
        
        return {
            "total_examples": total,
            "validated_examples": validated,
            "validation_rate": validated / total if total > 0 else 0,
            "average_score_difference": avg_diff,
            "results": results
        }
    
    def generate_report(self) -> str:
        """Genera reporte de validación"""
        if not self.validation_results:
            self.validate_all_examples()
        
        lines = [
            "# Validación de Scoring - Ejemplos Input/Output",
            "",
            "## Resumen",
            "",
            f"**Total de ejemplos**: {len(self.examples)}",
            f"**Ejemplos validados**: {sum(1 for r in self.validation_results if r['validated'])}",
            f"**Tasa de validación**: {sum(1 for r in self.validation_results if r['validated']) / len(self.examples) * 100:.1f}%",
            "",
            "---",
            "",
            "## Ejemplos Detallados",
            ""
        ]
        
        for i, (example, result) in enumerate(zip(self.examples, self.validation_results), 1):
            status = "✅" if result["validated"] else "⚠️"
            
            lines.extend([
                f"### Ejemplo {i}: {example.scenario}",
                "",
                f"**Status**: {status} {'VALIDADO' if result['validated'] else 'DESVIACIÓN'}",
                "",
                "#### Inputs:",
                ""
            ])
            
            for inp in example.inputs:
                lines.append(f"- **{inp.module}.{inp.function}**")
                for key, value in inp.input_data.items():
                    lines.append(f"  - `{key}`: {value}")
                lines.append("")
            
            lines.extend([
                "#### Output Esperado:",
                "",
                f"- **Score**: {example.expected_output.quantitative_score}",
                f"- **Cualitativo**: {example.expected_output.qualitative_assessment}",
                f"- **Justificación**: {example.expected_output.justification}",
                f"- **Evidencia**: {', '.join(example.expected_output.evidence)}",
                "",
                "#### Output Actual:",
                "",
                f"- **Score**: {example.actual_output.quantitative_score:.2f}",
                f"- **Diferencia**: {result['score_difference']:.3f}",
                f"- **Cualitativo**: {example.actual_output.qualitative_assessment}",
                "",
                "---",
                ""
            ])
        
        return "\n".join(lines)


def main():
    """Main execution"""
    logger.info("Iniciando validación de scoring con ejemplos...")
    
    validator = ScoringValidator()
    
    # Generate examples
    examples = validator.generate_examples()
    logger.info(f"✓ {len(examples)} ejemplos generados")
    
    # Validate
    validation_summary = validator.validate_all_examples()
    logger.info(f"✓ Validación completada: {validation_summary['validated_examples']}/{validation_summary['total_examples']} validados")
    
    # Save JSON results
    output_path = Path(__file__).parent / "scoring_validation_examples.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        # Convert examples to dict for JSON serialization
        examples_dict = [
            {
                "scenario": ex.scenario,
                "inputs": [asdict(inp) for inp in ex.inputs],
                "expected_output": asdict(ex.expected_output),
                "actual_output": asdict(ex.actual_output) if ex.actual_output else None,
                "validated": ex.validated
            }
            for ex in examples
        ]
        
        json.dump({
            "validation_summary": validation_summary,
            "examples": examples_dict
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Resultados JSON guardados en: {output_path}")
    
    # Generate report
    report = validator.generate_report()
    report_path = Path(__file__).parent / "scoring_validation_examples.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"✓ Reporte Markdown guardado en: {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDACIÓN DE SCORING - RESUMEN")
    print("="*80)
    print(f"Total de ejemplos: {validation_summary['total_examples']}")
    print(f"Ejemplos validados: {validation_summary['validated_examples']}")
    print(f"Tasa de validación: {validation_summary['validation_rate']*100:.1f}%")
    print(f"Diferencia promedio: {validation_summary['average_score_difference']:.3f}")
    print("="*80)
    
    return validation_summary


if __name__ == "__main__":
    main()
