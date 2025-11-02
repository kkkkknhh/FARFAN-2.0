#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Chunking Demo
======================

Demonstrates semantic analysis of Colombian Municipal Development Plans using
BGE-M3 embeddings and Bayesian evidence accumulation.

Run with:
    python -m examples.demo_semantic_chunking
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_chunking_policy import SemanticConfig, PolicyDocumentAnalyzer


def main():
    """Example usage of semantic chunking for policy analysis."""
    
    sample_pdm = """
PLAN DE DESARROLLO MUNICIPAL 2024-2027
MUNICIPIO DE EJEMPLO, COLOMBIA

1. DIAGNÓSTICO TERRITORIAL
El municipio cuenta con 45,000 habitantes, de los cuales 60% reside en zona rural.
La tasa de pobreza multidimensional es 42.3%, superior al promedio departamental.

2. VISIÓN ESTRATÉGICA
Para 2027, el municipio será reconocido por su desarrollo sostenible e inclusivo.

3. PLAN PLURIANUAL DE INVERSIONES
Se destinarán $12,500 millones al sector educación, con meta de construir
3 instituciones educativas y capacitar 250 docentes en pedagogías innovadoras.

4. SEGUIMIENTO Y EVALUACIÓN
Se implementará sistema de indicadores alineado con ODS, con mediciones semestrales.
"""
    
    config = SemanticConfig(
        chunk_size=512,
        chunk_overlap=100,
        similarity_threshold=0.80
    )
    
    analyzer = PolicyDocumentAnalyzer(config)
    results = analyzer.analyze(sample_pdm)
    
    print(json.dumps({
        "summary": results["summary"],
        "dimensions": {
            k: {
                "evidence_strength": v["evidence_strength"],
                "confidence": v["confidence"],
                "information_gain": v["information_gain"]
            }
            for k, v in results["causal_dimensions"].items()
        }
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
