#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contradiction Detection Demo
=============================

Demonstrates state-of-the-art contradiction detection in Colombian Municipal
Development Plans using transformer models and Bayesian inference.

Run with:
    python -m examples.demo_contradiction_detection
"""

import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from contradiction_deteccion import create_detector, PolicyDimension


def main():
    """Demonstrate contradiction detection on sample PDM."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    detector = create_detector()

    sample_pdm = """
PLAN DE DESARROLLO MUNICIPAL 2024-2027
"MUNICIPIO PRÓSPERO Y SOSTENIBLE"

COMPONENTE ESTRATÉGICO

Eje 1: Educación de Calidad
Objetivo: Aumentar la cobertura educativa al 95% para el año 2027.
Meta de resultado: Incrementar en 15 puntos porcentuales la cobertura en educación media.

Programa 1.1: Infraestructura Educativa
El municipio construirá 5 nuevas instituciones educativas en el primer semestre de 2025.
Recursos SGP Educación: $1,500 millones anuales.

Sin embargo, el presupuesto total del programa es de $800 millones para el cuatrienio.

Eje 2: Desarrollo Económico
La estrategia busca reducir la cobertura educativa para priorizar formación técnica.
Se ejecutará el 40% del presupuesto en el primer trimestre de 2025.

Para el segundo trimestre de 2025 se proyecta ejecutar el 70% del presupuesto total anual.

Programa 2.1: Apoyo Empresarial
Meta: Beneficiar a 10,000 familias con programas de emprendimiento.
Recursos propios asignados: $2,500 millones.
Recursos propios disponibles según plan financiero: $1,200 millones.

El programa tiene capacidad operativa para atender máximo 5,000 beneficiarios según 
análisis de capacidad institucional realizado en diagnóstico.

COMPONENTE PROGRAMÁTICO

Los proyectos de infraestructura educativa se ejecutarán después de la formación docente,
pero la formación docente requiere que primero existan las nuevas instituciones según
Acuerdo Municipal 045 de 2023.

El plan se rige por la Ley 152 de 1994 y el Decreto 1082 de 2015, estableciendo que
todos los programas deben tener indicadores de resultado. Sin embargo, el Programa 2.1
no cuenta con indicadores definidos.
"""

    result = detector.detect(
        text=sample_pdm,
        plan_name="PDM Municipio Próspero 2024-2027",
        dimension=PolicyDimension.ESTRATEGICO
    )

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"\nPlan: {result['plan_name']}")
    print(f"Dimensión analizada: {result['dimension']}")
    print(f"Total declaraciones: {result['total_statements']}")
    print(f"\nContradicciones detectadas: {result['total_contradictions']}")
    print(f"  - Críticas: {result['critical_severity_count']}")
    print(f"  - Altas: {result['high_severity_count']}")
    print(f"  - Medias: {result['medium_severity_count']}")

    print("\nMÉTRICAS DE COHERENCIA:")
    print(f"  - Score global: {result['coherence_metrics']['coherence_score']:.3f}")
    print(f"  - Calificación: {result['coherence_metrics']['quality_grade']}")
    print(f"  - Coherencia semántica: {result['coherence_metrics']['semantic_coherence']:.3f}")
    print(f"  - Consistencia temporal: {result['coherence_metrics']['temporal_consistency']:.3f}")
    print(f"  - Coherencia causal: {result['coherence_metrics']['causal_coherence']:.3f}")

    print("\nRECOMENDACIONES PRIORITARIAS:")
    for i, rec in enumerate(result['recommendations'][:3], 1):
        print(f"\n{i}. {rec['contradiction_type']} (Prioridad: {rec['priority'].upper()})")
        print(f"   Cantidad: {rec['count']} | Severidad promedio: {rec['avg_severity']:.2f}")
        print(f"   Esfuerzo estimado: {rec['estimated_effort']}")
        print(f"   Descripción: {rec['description']}")

    print("\n" + "="*80)

    output_file = Path("contradiction_analysis_result.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Resultados completos guardados en: {output_file}")


if __name__ == "__main__":
    main()
