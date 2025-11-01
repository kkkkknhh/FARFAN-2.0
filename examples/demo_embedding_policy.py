#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Policy Demo
=====================

Demonstrates advanced embedding and P-D-Q canonical notation analysis for
Colombian Municipal Development Plans.

Run with:
    python -m examples.demo_embedding_policy
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emebedding_policy import (
    create_policy_embedder,
    PDQIdentifier,
    PolicyDomain,
    AnalyticalDimension,
)


def main():
    """
    Complete example: analyzing Colombian Municipal Development Plan.
    """
    logging.basicConfig(level=logging.INFO)

    # Sample PDM excerpt (simplified)
    pdm_document = """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    MUNICIPIO DE EJEMPLO, COLOMBIA
    
    EJE ESTRATÉGICO 1: DERECHOS DE LAS MUJERES E IGUALDAD DE GÉNERO
    
    DIAGNÓSTICO
    El municipio presenta una brecha de género del 18.5% en participación laboral.
    Se identificaron 2,340 mujeres en situación de vulnerabilidad económica.
    El presupuesto asignado asciende a $450 millones para el cuatrienio.
    
    DISEÑO DE INTERVENCIÓN
    Se implementarán 3 programas de empoderamiento económico:
    - Programa de formación técnica: 500 beneficiarias
    - Microcréditos productivos: $280 millones
    - Fortalecimiento empresarial: 150 emprendimientos
    
    PRODUCTOS Y OUTPUTS
    Meta cuatrienio: reducir brecha de género al 12% (reducción del 35.1%)
    Indicador: Tasa de participación laboral femenina
    Línea base: 42.3% | Meta: 55.8%
    
    RESULTADOS ESPERADOS
    Incremento del 25% en ingresos promedio de beneficiarias
    Creación de 320 nuevos empleos formales para mujeres
    Sostenibilidad: 78% de emprendimientos activos a 2 años
    """

    metadata = {
        "doc_id": "PDM_EJEMPLO_2024_2027",
        "municipality": "Ejemplo",
        "department": "Ejemplo",
        "year": 2024,
    }

    # Create embedder
    print("=" * 80)
    print("POLICY ANALYSIS EMBEDDER - PRODUCTION EXAMPLE")
    print("=" * 80)

    embedder = create_policy_embedder(model_tier="balanced")

    # Process document
    print("\n1. PROCESSING DOCUMENT")
    chunks = embedder.process_document(pdm_document, metadata)
    print(f"   Generated {len(chunks)} semantic chunks")

    # Define P-D-Q query
    pdq_query = PDQIdentifier(
        question_unique_id="P1-D1-Q3",
        policy="P1",
        dimension="D1",
        question=3,
        rubric_key="D1-Q3",
    )

    print(f"\n2. ANALYZING P-D-Q: {pdq_query['question_unique_id']}")
    print(f"   Policy: {PolicyDomain.P1.value}")
    print(f"   Dimension: {AnalyticalDimension.D1.value}")

    # Generate comprehensive report
    report = embedder.generate_pdq_report(chunks, pdq_query)

    print("\n3. ANALYSIS RESULTS")
    print(f"   Evidence chunks found: {report['evidence_count']}")
    print(f"   Overall confidence: {report['confidence']:.3f}")
    print("\n   Numerical Evaluation:")
    print(
        f"   - Point estimate: {report['numerical_evaluation']['point_estimate']:.3f}"
    )
    print(
        f"   - 95% CI: [{report['numerical_evaluation']['credible_interval_95'][0]:.3f}, "
        f"{report['numerical_evaluation']['credible_interval_95'][1]:.3f}]"
    )
    print(
        f"   - Evidence strength: {report['numerical_evaluation']['evidence_strength']}"
    )
    print(
        f"   - Numerical coherence: {report['numerical_evaluation']['numerical_coherence']:.3f}"
    )

    print("\n4. TOP EVIDENCE PASSAGES:")
    for i, passage in enumerate(report["evidence_passages"], 1):
        print(f"\n   [{i}] Relevance: {passage['relevance_score']:.3f}")
        print(f"       {passage['content'][:200]}...")

    # System diagnostics
    print("\n5. SYSTEM DIAGNOSTICS")
    diag = embedder.get_diagnostics()
    print(f"   Model: {diag['model']}")
    print(f"   Cache efficiency: {diag['embedding_cache_size']} embeddings cached")
    print(f"   Total chunks processed: {diag['total_chunks_processed']}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
