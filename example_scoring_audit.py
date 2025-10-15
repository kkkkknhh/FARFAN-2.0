#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage of Scoring Audit Module
Demonstrates comprehensive audit of scoring system
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from scoring_audit import ScoringSystemAuditor, EXPECTED_TOTAL_QUESTIONS

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger("example_audit")


@dataclass
class MockResponse:
    """Mock response for demonstration"""
    pregunta_id: str
    nota_cuantitativa: float
    respuesta_texto: str = "Example response"
    argumento: str = "Example argument"
    evidencia: list = None
    modulos_utilizados: list = None
    nivel_confianza: float = 0.85


def create_sample_responses():
    """Create sample 300-question response set"""
    logger.info("Creating sample responses for 300 questions...")
    responses = {}
    
    for p in range(1, 11):  # P1-P10
        for d in range(1, 7):  # D1-D6
            for q in range(1, 6):  # Q1-Q5
                question_id = f"P{p}-D{d}-Q{q}"
                
                # Vary scores by dimension
                if d == 6:  # D6 Theory of Change - some below threshold
                    score = 0.50 if p % 3 == 0 else 0.75
                elif d == 1:  # D1 Diagnóstico
                    score = 0.85
                elif d == 5:  # D5 Impactos
                    score = 0.65
                else:
                    score = 0.72
                
                responses[question_id] = MockResponse(
                    pregunta_id=question_id,
                    nota_cuantitativa=score
                )
    
    logger.info(f"Created {len(responses)} responses")
    return responses


def create_sample_dimension_weights():
    """Create sample dimension weights per policy"""
    logger.info("Creating sample dimension weights...")
    
    weights = {}
    for p in range(1, 11):
        policy_id = f"P{p}"
        weights[policy_id] = {
            "D1": 0.15,
            "D2": 0.15,
            "D3": 0.20,
            "D4": 0.20,
            "D5": 0.15,
            "D6": 0.15
        }
    
    return weights


def create_sample_meso_report():
    """Create sample MESO report structure"""
    logger.info("Creating sample MESO report...")
    
    return {
        'metadata': {
            'report_level': 'MESO',
            'clusters': 4,
            'dimensions': 6
        },
        'clusters': {
            'C1': {
                'nombre': 'Seguridad y Paz',
                'puntos_incluidos': ['P1', 'P2', 'P8'],
                'dimensiones': {
                    'D1': {'score': 0.82, 'num_preguntas': 15},
                    'D2': {'score': 0.75, 'num_preguntas': 15},
                    'D3': {'score': 0.70, 'num_preguntas': 15},
                    'D4': {'score': 0.68, 'num_preguntas': 15},
                    'D5': {'score': 0.65, 'num_preguntas': 15},
                    'D6': {'score': 0.62, 'num_preguntas': 15}
                }
            },
            'C2': {
                'nombre': 'Derechos Sociales',
                'puntos_incluidos': ['P4', 'P5', 'P6'],
                'dimensiones': {
                    'D1': {'score': 0.85, 'num_preguntas': 15},
                    'D2': {'score': 0.72, 'num_preguntas': 15},
                    'D3': {'score': 0.73, 'num_preguntas': 15},
                    'D4': {'score': 0.71, 'num_preguntas': 15},
                    'D5': {'score': 0.66, 'num_preguntas': 15},
                    'D6': {'score': 0.58, 'num_preguntas': 15}
                }
            },
            'C3': {
                'nombre': 'Territorio y Ambiente',
                'puntos_incluidos': ['P3', 'P7'],
                'dimensiones': {
                    'D1': {'score': 0.88, 'num_preguntas': 10},
                    'D2': {'score': 0.74, 'num_preguntas': 10},
                    'D3': {'score': 0.76, 'num_preguntas': 10},
                    'D4': {'score': 0.70, 'num_preguntas': 10},
                    'D5': {'score': 0.67, 'num_preguntas': 10},
                    'D6': {'score': 0.68, 'num_preguntas': 10}
                }
            },
            'C4': {
                'nombre': 'Poblaciones Especiales',
                'puntos_incluidos': ['P9', 'P10'],
                'dimensiones': {
                    'D1': {'score': 0.80, 'num_preguntas': 10},
                    'D2': {'score': 0.69, 'num_preguntas': 10},
                    'D3': {'score': 0.68, 'num_preguntas': 10},
                    'D4': {'score': 0.65, 'num_preguntas': 10},
                    'D5': {'score': 0.62, 'num_preguntas': 10},
                    'D6': {'score': 0.60, 'num_preguntas': 10}
                }
            }
        }
    }


def create_sample_macro_report(global_score: float):
    """Create sample MACRO report structure"""
    logger.info("Creating sample MACRO report...")
    
    return {
        'metadata': {
            'report_level': 'MACRO'
        },
        'evaluacion_global': {
            'score_global': global_score,
            'score_dnp_compliance': 72.5,
            'nivel_alineacion': 'BUENO',
            'total_preguntas': 300
        },
        'analisis_retrospectivo': {
            'fortalezas': ['Diagnóstico robusto', 'Indicadores claros'],
            'debilidades': ['Teoría de cambio incompleta', 'Métricas de impacto débiles']
        },
        'analisis_prospectivo': {
            'recomendaciones': ['Fortalecer D6', 'Mejorar articulación causal']
        },
        'recomendaciones_prioritarias': [
            'Desarrollar teoría de cambio explícita',
            'Especificar cadenas causales',
            'Definir supuestos críticos'
        ]
    }


def create_sample_dnp_results():
    """Create sample DNP validation results"""
    logger.info("Creating sample DNP results...")
    
    @dataclass
    class MockDNPResults:
        cumple_competencias: bool = True
        cumple_mga: bool = True
        cumple_pdet: bool = False
        nivel_cumplimiento: str = "BUENO"
        score_total: float = 72.5
        competencias_validadas: list = None
        indicadores_mga_usados: list = None
        recomendaciones: list = None
    
    return MockDNPResults(
        competencias_validadas=['Educación', 'Salud', 'Agua potable'],
        indicadores_mga_usados=['IND-EDU-001', 'IND-SAL-002'],
        recomendaciones=['Ampliar cobertura MGA', 'Incluir indicadores PDET']
    )


def main():
    """Run comprehensive audit example"""
    print("="*70)
    print("FARFAN 2.0 - Scoring System Audit Example")
    print("="*70)
    print()
    
    # Initialize auditor
    auditor = ScoringSystemAuditor(output_dir=Path("example_audit_output"))
    
    # Create sample data
    responses = create_sample_responses()
    dimension_weights = create_sample_dimension_weights()
    meso_report = create_sample_meso_report()
    
    # Calculate global score for MACRO
    all_scores = [r.nota_cuantitativa for r in responses.values()]
    global_score = sum(all_scores) / len(all_scores)
    
    macro_report = create_sample_macro_report(global_score)
    dnp_results = create_sample_dnp_results()
    
    print(f"\nRunning comprehensive audit on {len(responses)} questions...")
    print()
    
    # Run audit
    report = auditor.audit_complete_system(
        question_responses=responses,
        dimension_weights=dimension_weights,
        meso_report=meso_report,
        macro_report=macro_report,
        dnp_results=dnp_results
    )
    
    # Export report
    print("\nExporting audit report...")
    output_path = auditor.export_report(filename="example_audit_report.json")
    print(f"Report saved to: {output_path}")
    
    # Display key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    print(f"\n1. Matrix Structure:")
    print(f"   - Expected: {EXPECTED_TOTAL_QUESTIONS} questions")
    print(f"   - Found: {report.total_questions_found} questions")
    print(f"   - Policies: {len(report.policies_found)}/10")
    print(f"   - Dimensions: {len(report.dimensions_found)}/6")
    print(f"   - Status: {'✓ VALID' if report.matrix_valid else '✗ INVALID'}")
    
    print(f"\n2. MICRO Scores:")
    print(f"   - Status: {'✓ VALID' if report.micro_scores_valid else '✗ INVALID'}")
    print(f"   - Issues: {len([i for i in report.micro_issues if i.category == 'micro_scoring'])}")
    
    print(f"\n3. MESO Aggregation:")
    print(f"   - Status: {'✓ VALID' if report.meso_aggregation_valid else '✗ INVALID'}")
    print(f"   - Weight issues: {len(report.meso_weight_issues)}")
    print(f"   - Convergence gaps: {len(report.meso_convergence_gaps)}")
    
    print(f"\n4. MACRO Alignment:")
    print(f"   - Status: {'✓ VALID' if report.macro_alignment_valid else '✗ INVALID'}")
    print(f"   - Issues: {len(report.macro_issues)}")
    
    print(f"\n5. D6 Theory of Change (Critical Threshold = 0.55):")
    print(f"   - Scores below threshold: {len(report.d6_scores_below_threshold)}")
    if report.d6_scores_below_threshold:
        print(f"   - ⚠ WARNING: {len(report.d6_scores_below_threshold)} D6 questions need improvement")
        for item in report.d6_scores_below_threshold[:3]:
            print(f"     • {item['question_id']}: {item['score']:.3f} (gap: {item['gap']:.3f})")
        if len(report.d6_scores_below_threshold) > 3:
            print(f"     ... and {len(report.d6_scores_below_threshold) - 3} more")
    
    print(f"\n6. DNP Integration:")
    print(f"   - Status: {'✓ VALID' if report.dnp_integration_valid else '✗ INVALID'}")
    print(f"   - Issues: {len(report.dnp_issues)}")
    
    print(f"\n7. Overall Summary:")
    print(f"   - Overall Status: {'✓ VALID' if report.overall_valid else '✗ HAS ISSUES'}")
    print(f"   - Total Issues: {report.total_issues}")
    print(f"   - Critical Issues: {report.critical_issues}")
    
    if report.critical_issues > 0:
        print(f"\n   CRITICAL ISSUES FOUND:")
        for issue in [i for i in (report.micro_issues + report.meso_weight_issues + 
                                  report.meso_convergence_gaps + report.macro_issues + 
                                  report.rubric_issues + report.dnp_issues) 
                     if i.severity == "CRITICAL"]:
            print(f"   • [{issue.category}] {issue.description}")
            print(f"     Location: {issue.location}")
            print(f"     Recommendation: {issue.recommendation}")
    
    print("\n" + "="*70)
    print(f"Audit complete. See {output_path} for full details.")
    print("="*70)


if __name__ == "__main__":
    main()
