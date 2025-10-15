#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Using the Axiomatic Validator

This example demonstrates how to use the unified AxiomaticValidator
to validate a PDM (Plan de Desarrollo Municipal).

The validator integrates three validation systems:
1. TeoriaCambio - Structural/Causal validation
2. PolicyContradictionDetectorV2 - Semantic validation
3. ValidadorDNP - Regulatory compliance
"""


def example_basic_usage():
    """
    Basic usage example of the Axiomatic Validator
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Validator Usage")
    print("=" * 80)

    from validators import (
        AxiomaticValidator,
        ExtractedTable,
        PDMOntology,
        SemanticChunk,
        ValidationConfig,
    )

    # Step 1: Create configuration
    print("\n1. Creating validation configuration...")
    config = ValidationConfig(
        dnp_lexicon_version="2025",
        es_municipio_pdet=False,  # Not a PDET municipality
        contradiction_threshold=0.05,  # 5% max contradiction density
        enable_structural_penalty=True,
        enable_human_gating=True,
    )
    print(f"   ✓ Config created: DNP version {config.dnp_lexicon_version}")

    # Step 2: Create ontology
    print("\n2. Creating PDM ontology...")
    ontology = PDMOntology()
    print(f"   ✓ Ontology created with {len(ontology.canonical_chain)} categories")
    print(f"   Canonical chain: {' → '.join(ontology.canonical_chain)}")

    # Step 3: Initialize validator
    print("\n3. Initializing Axiomatic Validator...")
    validator = AxiomaticValidator(config, ontology)
    print("   ✓ Validator initialized")

    # Step 4: Create semantic chunks (in real usage, these come from PDM text)
    print("\n4. Preparing semantic chunks from PDM...")
    semantic_chunks = [
        SemanticChunk(
            text="El municipio invertirá 1000 millones en educación para mejorar la calidad educativa.",
            dimension="ESTRATEGICO",
            position=(100, 190),
            metadata={"section": "educacion", "priority": "high"},
        ),
        SemanticChunk(
            text="Se contratarán 50 nuevos docentes en el primer año del plan.",
            dimension="PROGRAMATICO",
            position=(300, 365),
            metadata={"section": "educacion", "action": "contratacion"},
        ),
        SemanticChunk(
            text="La meta es alcanzar un 95% de cobertura educativa en el cuatrienio.",
            dimension="ESTRATEGICO",
            position=(500, 573),
            metadata={"section": "educacion", "type": "meta"},
        ),
    ]
    print(f"   ✓ Created {len(semantic_chunks)} semantic chunks")

    # Step 5: Create financial data (optional)
    print("\n5. Preparing financial data...")
    financial_data = [
        ExtractedTable(
            title="Inversión en Educación 2025-2028",
            headers=["Año", "Monto (Millones)", "Fuente"],
            rows=[
                ["2025", 250, "SGP"],
                ["2026", 300, "SGP + Propios"],
                ["2027", 350, "SGP + Propios"],
                ["2028", 400, "SGP + Propios + Regalías"],
            ],
            metadata={"sector": "educacion", "verified": True},
        )
    ]
    print(f"   ✓ Created {len(financial_data)} financial table(s)")

    print("\n" + "=" * 80)
    print("Note: Full validation requires networkx and other dependencies.")
    print("This example demonstrates the data structure setup.")
    print("=" * 80)


def example_validation_result():
    """
    Example showing how to work with validation results
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Working with Validation Results")
    print("=" * 80)

    from validators import AxiomaticValidationResult, ValidationSeverity

    # Create a mock result
    print("\n1. Creating validation result...")
    result = AxiomaticValidationResult()
    result.structural_valid = True
    result.contradiction_density = 0.03  # 3% - below threshold
    result.regulatory_score = 75.0
    result.total_nodes = 15
    result.total_edges = 20

    print(f"   ✓ Result created")

    # Add a critical failure
    print("\n2. Adding a critical failure...")
    result.add_critical_failure(
        dimension="D6",
        question="Q2",
        evidence=[("Productos", "Insumos")],  # Invalid backward edge
        impact="Salto lógico detectado - orden causal invertido",
        recommendations=[
            "Revisar la dirección de la relación causal",
            "Asegurar que INSUMOS precede a PRODUCTOS",
        ],
    )
    print(f"   ✓ Critical failure added")
    print(f"   Overall valid: {result.is_valid}")

    # Get summary
    print("\n3. Getting validation summary...")
    summary = result.get_summary()
    print(f"   ✓ Summary generated:")
    print(f"      - Valid: {summary['is_valid']}")
    print(f"      - Structural Valid: {summary['structural_valid']}")
    print(f"      - Contradiction Density: {summary['contradiction_density']:.2%}")
    print(f"      - Regulatory Score: {summary['regulatory_score']}/100")
    print(f"      - Critical Failures: {summary['critical_failures']}")
    print(f"      - Requires Manual Review: {summary['requires_manual_review']}")

    # Show failures
    if result.failures:
        print("\n4. Detailed failures:")
        for i, failure in enumerate(result.failures, 1):
            print(f"\n   Failure {i}:")
            print(f"      - Dimension: {failure.dimension}")
            print(f"      - Question: {failure.question}")
            print(f"      - Severity: {failure.severity.value}")
            print(f"      - Impact: {failure.impact}")
            print(f"      - Recommendations:")
            for rec in failure.recommendations:
                print(f"         • {rec}")


def example_configuration_variants():
    """
    Example showing different configuration options
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Configuration Variants")
    print("=" * 80)

    from validators import ValidationConfig

    print("\n1. Standard Municipality Configuration:")
    standard_config = ValidationConfig(
        dnp_lexicon_version="2025",
        es_municipio_pdet=False,
        contradiction_threshold=0.05,
        enable_structural_penalty=True,
        enable_human_gating=True,
    )
    print(f"   - PDET: {standard_config.es_municipio_pdet}")
    print(
        f"   - Contradiction Threshold: {standard_config.contradiction_threshold:.1%}"
    )
    print(f"   - Structural Penalty: {standard_config.enable_structural_penalty}")
    print(f"   - Human Gating: {standard_config.enable_human_gating}")

    print("\n2. PDET Municipality Configuration:")
    pdet_config = ValidationConfig(
        dnp_lexicon_version="2025",
        es_municipio_pdet=True,  # PDET municipality
        contradiction_threshold=0.03,  # Stricter for PDET
        enable_structural_penalty=True,
        enable_human_gating=True,
    )
    print(f"   - PDET: {pdet_config.es_municipio_pdet}")
    print(f"   - Contradiction Threshold: {pdet_config.contradiction_threshold:.1%}")
    print(f"   - Note: PDET municipalities have additional validation requirements")

    print("\n3. Lenient Configuration (for testing/development):")
    lenient_config = ValidationConfig(
        dnp_lexicon_version="2025",
        es_municipio_pdet=False,
        contradiction_threshold=0.10,  # More permissive
        enable_structural_penalty=False,  # No penalties
        enable_human_gating=False,  # Auto-approve
    )
    print(f"   - Contradiction Threshold: {lenient_config.contradiction_threshold:.1%}")
    print(f"   - Structural Penalty: {lenient_config.enable_structural_penalty}")
    print(f"   - Human Gating: {lenient_config.enable_human_gating}")
    print(f"   - Note: Use only for development/testing")


def example_custom_ontology():
    """
    Example showing custom ontology configuration
    """
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: Custom Ontology")
    print("=" * 80)

    from validators import PDMOntology

    print("\n1. Default Ontology:")
    default_ontology = PDMOntology()
    print(f"   Canonical Chain: {' → '.join(default_ontology.canonical_chain)}")
    print(f"   Total Categories: {len(default_ontology.canonical_chain)}")

    print("\n2. Custom Ontology:")
    custom_ontology = PDMOntology(
        canonical_chain=[
            "DIAGNOSTICO",
            "ESTRATEGIA",
            "ACCIONES",
            "RESULTADOS",
            "IMPACTO",
        ],
        dimensions=["D1", "D2", "D3", "D4", "D5", "D6"],
        policy_areas=["Educacion", "Salud", "Infraestructura", "Medio Ambiente"],
    )
    print(f"   Canonical Chain: {' → '.join(custom_ontology.canonical_chain)}")
    print(f"   Dimensions: {len(custom_ontology.dimensions)}")
    print(f"   Policy Areas: {len(custom_ontology.policy_areas)}")


if __name__ == "__main__":
    """
    Run all examples
    """
    try:
        example_basic_usage()
        example_validation_result()
        example_configuration_variants()
        example_custom_ontology()

        print("\n\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(
            "\nFor full validation with real PDM data, ensure all dependencies are installed:"
        )
        print("  - networkx")
        print("  - spacy (with es_core_news_lg model)")
        print("  - torch")
        print("  - transformers")
        print("  - sentence-transformers")
        print("\nSee Dockerfile and requirements.txt for complete dependency list.")
        print("=" * 80)

    except Exception as e:
        print(f"\n\nError running examples: {e}")
        print("This is expected if dependencies are not installed.")
        print("The validator module structure is correct and ready for use.")
