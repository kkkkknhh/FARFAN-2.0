#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Policy Analysis Example
==============================

Demonstrates the complete FARFAN 2.0 pipeline using the orchestrator.

This example shows how to:
1. Create production dependencies
2. Initialize the pipeline
3. Run a complete policy analysis
4. Access results through contracts

Run with:
    python -m examples.basic_analysis
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.factory import create_production_dependencies
from orchestrator.pipeline import create_pipeline
from core_contracts import PipelineInput, CURRENT_VERSIONS


def main():
    """Run a basic policy analysis using the orchestrator."""
    
    # Sample PDM text
    sample_pdm = """
PLAN DE DESARROLLO MUNICIPAL 2024-2027
MUNICIPIO DE EJEMPLO, COLOMBIA

COMPONENTE ESTRATÉGICO

1. DIAGNÓSTICO TERRITORIAL
El municipio cuenta con 45,000 habitantes, de los cuales 60% reside en zona rural.
La tasa de pobreza multidimensional es 42.3%, superior al promedio departamental.
Se identifican brechas significativas en acceso a servicios básicos y educación.

2. VISIÓN ESTRATÉGICA
Para 2027, el municipio será reconocido por su desarrollo sostenible e inclusivo,
garantizando derechos fundamentales y reduciendo desigualdades estructurales.

3. EJES ESTRATÉGICOS

Eje 1: Educación de Calidad
Objetivo: Aumentar la cobertura educativa al 95% para el año 2027.
Meta de resultado: Incrementar en 15 puntos porcentuales la cobertura en educación media.
Recursos: $12,500 millones para construcción de 3 instituciones educativas.

Eje 2: Desarrollo Económico Local
Objetivo: Reducir desempleo al 8% mediante fortalecimiento empresarial.
Meta: Beneficiar a 10,000 familias con programas de emprendimiento.
Recursos: $8,000 millones para apoyo a MIPYMES y formación técnica.

4. PLAN PLURIANUAL DE INVERSIONES
Se destinarán $12,500 millones al sector educación, con meta de construir
3 instituciones educativas y capacitar 250 docentes en pedagogías innovadoras.

Inversión total cuatrienio: $85,000 millones distribuidos en:
- Educación: 30%
- Salud: 25%
- Infraestructura: 20%
- Desarrollo económico: 15%
- Otros sectores: 10%

5. SEGUIMIENTO Y EVALUACIÓN
Se implementará sistema de indicadores alineado con ODS, con mediciones semestrales
y ajustes anuales basados en evidencia.
"""
    
    # Create production dependencies
    print("=" * 80)
    print("FARFAN 2.0 - POLICY ANALYSIS PIPELINE")
    print("=" * 80)
    print("\n1. Initializing production dependencies...")
    
    deps = create_production_dependencies()
    
    # Create pipeline with dependencies
    print("2. Creating analysis pipeline...")
    pipeline = create_pipeline(
        log_port=deps["log_port"],
        file_port=deps["file_port"],
        clock_port=deps["clock_port"],
    )
    
    # Prepare pipeline input
    print("3. Preparing analysis input...")
    pipeline_input: PipelineInput = {
        "text": sample_pdm,
        "plan_name": "PDM Ejemplo 2024-2027",
        "dimension": "estratégico",
        "config": {},
        "schema_version": CURRENT_VERSIONS["pipeline"],
    }
    
    # Execute pipeline
    print("4. Executing analysis pipeline...")
    print("-" * 80)
    
    result = pipeline.orchestrate(pipeline_input)
    
    # Display results
    print("-" * 80)
    print("\n5. ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\nStatements Extracted: {result['statements']['total_count']}")
    print(f"Schema Version: {result['statements']['schema_version']}")
    
    print(f"\nContradictions Detected: {result['contradictions']['total_count']}")
    print(f"Quality Grade: {result['contradictions']['quality_grade']}")
    
    print(f"\nCoherence Score: {result['coherence_metrics']['coherence_score']:.3f}")
    print(f"Quality Status: {result['coherence_metrics']['quality_status']}")
    print(f"Causal Incoherence Count: {result['coherence_metrics']['causal_incoherence_count']}")
    
    print(f"\nRegulatory Compliance Score: {result['regulatory_analysis']['compliance_score']:.3f}")
    print(f"Critical Violations: {result['regulatory_analysis']['critical_violations']}")
    
    print(f"\nOverall Audit Grade: {result['audit_summary']['overall_grade']}")
    print(f"\nExecutive Summary:")
    print(f"  {result['audit_summary']['executive_summary']}")
    
    if result['audit_summary']['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['audit_summary']['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
