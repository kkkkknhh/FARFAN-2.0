#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Demo for Evidence Quality Auditors
=================================================

This script demonstrates all four auditors with realistic PDM examples.
"""

from evidence_quality_auditors import (
    FinancialTraceabilityAuditor,
    IndicatorMetadata,
    OperationalizationAuditor,
    QuantifiedGapAuditor,
    SystemicRiskAuditor,
    run_all_audits,
)


def print_separator():
    print("\n" + "=" * 80 + "\n")


def demo_operationalization():
    """Demo: Operationalization Auditor (D3-Q1)"""
    print("DEMO 1: OPERATIONALIZATION AUDITOR (D3-Q1)")
    print("Purpose: Verify indicator ficha técnica completeness")
    print_separator()

    # Complete indicators (should get EXCELLENT)
    indicators_complete = [
        IndicatorMetadata(
            codigo="EDU-001",
            nombre="Tasa de cobertura neta educación preescolar",
            linea_base=75.0,
            meta=90.0,
            fuente="SIMAT - Secretaría de Educación",
            formula="(Matrícula preescolar 5 años / Población 5 años) * 100",
            unidad_medida="Porcentaje",
            periodicidad="Anual",
        ),
        IndicatorMetadata(
            codigo="EDU-002",
            nombre="Tasa de deserción escolar",
            linea_base=15.0,
            meta=5.0,
            fuente="DANE - Sistema educativo",
            formula="(Desertores año / Matrícula total) * 100",
        ),
    ]

    auditor = OperationalizationAuditor(metadata_threshold=0.80)
    result = auditor.audit_indicators(indicators_complete)

    print(f"✓ Test Case: 2 complete indicators")
    print(f"  Severity: {result.severity.value.upper()}")
    print(f"  SOTA Compliance: {'YES ✓' if result.sota_compliance else 'NO ✗'}")
    print(f"  Completeness Ratio: {result.metrics['completeness_ratio']:.1%}")
    print(f"  Complete: {result.metrics['complete_indicators']}/2")
    print(f"\n  Recommendations ({len(result.recommendations)}):")
    for rec in result.recommendations:
        print(f"    • {rec}")

    # Incomplete indicators (should get REQUIRES_REVIEW)
    indicators_incomplete = [
        IndicatorMetadata(codigo="SAL-001", nombre="Cobertura vacunación"),
        IndicatorMetadata(codigo="SAL-002", nombre="Mortalidad infantil"),
    ]

    result_incomplete = auditor.audit_indicators(indicators_incomplete)

    print(f"\n✓ Test Case: 2 incomplete indicators")
    print(f"  Severity: {result_incomplete.severity.value.upper()}")
    print(
        f"  SOTA Compliance: {'YES ✓' if result_incomplete.sota_compliance else 'NO ✗'}"
    )
    print(
        f"  Completeness Ratio: {result_incomplete.metrics['completeness_ratio']:.1%}"
    )
    print(f"  Incomplete: {result_incomplete.metrics['incomplete_indicators']}/2")


def demo_financial_traceability():
    """Demo: Financial Traceability Auditor (D1-Q3, D3-Q3)"""
    print_separator()
    print("DEMO 2: FINANCIAL TRACEABILITY AUDITOR (D1-Q3, D3-Q3)")
    print("Purpose: Verify BPIN/PPI code traceability")
    print_separator()

    # PDM with good traceability
    text_good = """
    COMPONENTE FINANCIERO - PLAN PLURIANUAL DE INVERSIONES
    
    EDUCACIÓN:
    Proyecto BPIN 2024001234567 - Mejoramiento infraestructura educativa
    Presupuesto: $5,000 millones
    Código PPI-2024000123 para construcción aulas
    
    SALUD:
    Inversión proyecto BPIN 2024009876543 - Ampliación hospital municipal
    Plan Plurianual PPI 2024000456 servicios de salud
    """

    auditor = FinancialTraceabilityAuditor(confidence_threshold=0.95)
    result = auditor.audit_financial_codes(text_good)

    print(f"✓ Test Case: PDM with BPIN and PPI codes")
    print(f"  Severity: {result.severity.value.upper()}")
    print(f"  SOTA Compliance: {'YES ✓' if result.sota_compliance else 'NO ✗'}")
    print(f"  Total codes found: {result.metrics['total_codes']}")
    print(f"    - BPIN codes: {result.metrics['bpin_codes']}")
    print(f"    - PPI codes: {result.metrics['ppi_codes']}")
    print(f"  High confidence: {result.metrics['high_confidence_ratio']:.1%}")

    # PDM without codes
    text_bad = "Proyecto de mejoramiento sin códigos de inversión definidos."
    result_bad = auditor.audit_financial_codes(text_bad)

    print(f"\n✓ Test Case: PDM without codes")
    print(f"  Severity: {result_bad.severity.value.upper()}")
    print(f"  Total codes: {result_bad.metrics['total_codes']}")
    print(f"  Recommendations ({len(result_bad.recommendations)}):")
    for rec in result_bad.recommendations[:2]:
        print(f"    • {rec}")


def demo_quantified_gaps():
    """Demo: Quantified Gap Auditor (D1-Q2)"""
    print_separator()
    print("DEMO 3: QUANTIFIED GAP AUDITOR (D1-Q2)")
    print("Purpose: Detect and quantify data gaps and deficits")
    print_separator()

    text = """
    DIAGNÓSTICO TERRITORIAL
    
    El municipio presenta múltiples desafíos:
    
    1. EDUCACIÓN: Déficit de 35% en cobertura de educación preescolar.
       Brecha de 1,200 cupos escolares para atender demanda.
       Se identifican vacíos de información sobre deserción en zonas rurales.
    
    2. SALUD: Brecha de 40% en acceso a servicios de salud especializados.
       Déficit de 25 camas hospitalarias según estándares OMS.
    
    3. DATOS: Problema crítico de subregistro en censos de población rural.
       Vacío de datos sobre población víctima del conflicto.
       Sub-registro afecta estadísticas de necesidades básicas insatisfechas.
    """

    auditor = QuantifiedGapAuditor()
    result = auditor.audit_quantified_gaps(text)

    print(f"✓ Analysis of comprehensive diagnostic text")
    print(f"  Severity: {result.severity.value.upper()}")
    print(f"  SOTA Compliance: {'YES ✓' if result.sota_compliance else 'NO ✗'}")
    print(f"  Total gaps detected: {result.metrics['total_gaps']}")
    print(
        f"  Quantified: {result.metrics['quantified_gaps']} ({result.metrics['quantification_ratio']:.1%})"
    )
    print(f"  Subregistro cases: {result.metrics['subregistro_count']}")
    print(f"\n  Gap distribution:")
    for gap_type, count in result.metrics["gap_type_distribution"].items():
        if count > 0:
            print(f"    • {gap_type}: {count}")

    print(f"\n  Sample findings:")
    for finding in result.findings[:3]:
        print(f"    • {finding['gap_type']}: quantified={finding['quantified']}")


def demo_systemic_risk():
    """Demo: Systemic Risk Auditor (D4-Q5, D5-Q4)"""
    print_separator()
    print("DEMO 4: SYSTEMIC RISK AUDITOR (D4-Q5, D5-Q4)")
    print("Purpose: Verify PND/ODS alignment and calculate systemic risk")
    print_separator()

    # Well-aligned PDM
    text_aligned = """
    MARCO ESTRATÉGICO
    
    El Plan de Desarrollo Municipal 2024-2027 se alinea estratégicamente con:
    
    • Plan Nacional de Desarrollo "Colombia Potencia Mundial de la Vida"
    • Objetivos de Desarrollo Sostenible (ODS):
      - ODS-4: Educación de Calidad
      - ODS-10: Reducción de las Desigualdades  
      - ODS-11: Ciudades y Comunidades Sostenibles
      - ODS-16: Paz, Justicia e Instituciones Sólidas
    
    Esta alineación garantiza coherencia con marcos macro-causales nacionales
    y reduce riesgos sistémicos en la implementación.
    """

    auditor = SystemicRiskAuditor(excellent_threshold=0.10)
    result = auditor.audit_systemic_risk(text_aligned)

    print(f"✓ Test Case: Well-aligned PDM")
    print(f"  Severity: {result.severity.value.upper()}")
    print(f"  SOTA Compliance: {'YES ✓' if result.sota_compliance else 'NO ✗'}")
    print(f"  PND Alignment: {'YES ✓' if result.metrics['pnd_alignment'] else 'NO ✗'}")
    print(f"  ODS Count: {result.metrics['ods_count']}")
    print(f"  ODS Numbers: {result.metrics['ods_numbers']}")
    print(f"  Risk Score: {result.metrics['risk_score']:.3f} (target: <0.10)")

    # Poorly aligned PDM
    text_unaligned = """
    ESTRATEGIA MUNICIPAL
    
    El municipio implementará programas de desarrollo local
    enfocados en necesidades identificadas por la comunidad.
    """

    result_bad = auditor.audit_systemic_risk(text_unaligned)

    print(f"\n✓ Test Case: Poorly-aligned PDM")
    print(f"  Severity: {result_bad.severity.value.upper()}")
    print(f"  Risk Score: {result_bad.metrics['risk_score']:.3f}")
    print(f"  Misalignments: {result_bad.metrics['misalignment_count']}")
    print(f"  Recommendations ({len(result_bad.recommendations)}):")
    for rec in result_bad.recommendations[:2]:
        print(f"    • {rec}")


def demo_integrated():
    """Demo: Run all auditors together"""
    print_separator()
    print("DEMO 5: INTEGRATED ANALYSIS - ALL AUDITORS")
    print("Purpose: Comprehensive PDM quality assessment")
    print_separator()

    comprehensive_pdm = """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    "HACIA UN MUNICIPIO EQUITATIVO Y SOSTENIBLE"
    
    1. DIAGNÓSTICO
    El municipio presenta un déficit de 40% en cobertura educativa y 
    brecha de 35% en acceso a servicios de salud. Se identifican vacíos 
    de información en población rural dispersa y subregistro en censos.
    
    2. COMPONENTE ESTRATÉGICO
    Alineado con Plan Nacional de Desarrollo y contribuye a:
    - ODS-4 (Educación de Calidad)
    - ODS-10 (Reducción de Desigualdades)
    - ODS-3 (Salud y Bienestar)
    
    3. COMPONENTE FINANCIERO
    Proyecto BPIN 2024000123456 - Inversión educativa: $8,000 millones
    Plan Plurianual PPI-2024000789 - Salud: $5,000 millones
    Código BPIN 2024000999888 - Infraestructura vial
    
    4. INDICADORES DE PRODUCTO
    - Tasa de cobertura neta educación preescolar
      LB: 60%, Meta: 90%, Fuente: SED
      Fórmula: (Matriculados 5 años / Población 5 años) * 100
    
    - Porcentaje de vías terciarias mejoradas
      LB: 30%, Meta: 70%, Fuente: Secretaría Obras Públicas
      Fórmula: (Km vías mejoradas / Total km vías) * 100
    """

    # Prepare indicators
    indicators = [
        IndicatorMetadata(
            codigo="EDU-001",
            nombre="Tasa de cobertura neta educación preescolar",
            linea_base=60.0,
            meta=90.0,
            fuente="SED",
            formula="(Matriculados 5 años / Población 5 años) * 100",
        ),
        IndicatorMetadata(
            codigo="INF-001",
            nombre="Porcentaje de vías terciarias mejoradas",
            linea_base=30.0,
            meta=70.0,
            fuente="Secretaría Obras Públicas",
            formula="(Km vías mejoradas / Total km vías) * 100",
        ),
    ]

    # Run all audits
    results = run_all_audits(text=comprehensive_pdm, indicators=indicators)

    print("✓ Comprehensive PDM Analysis Results:\n")

    # Summary table
    print("  Audit Type                      | Severity         | SOTA  | Findings")
    print("  " + "-" * 70)

    for audit_type, result in results.items():
        severity_str = result.severity.value.upper()[:16].ljust(16)
        sota_str = "✓" if result.sota_compliance else "✗"
        findings_count = len(result.findings)

        print(
            f"  {audit_type[:30].ljust(30)} | {severity_str} | {sota_str}     | {findings_count}"
        )

    print("\n  Overall Assessment:")
    all_compliant = all(r.sota_compliance for r in results.values())
    if all_compliant:
        print("  ✓ ALL AUDITS PASS SOTA COMPLIANCE")
        print("  Quality Grade: EXCELENTE")
    else:
        compliant_count = sum(1 for r in results.values() if r.sota_compliance)
        print(f"  ⚠ {compliant_count}/{len(results)} audits pass SOTA compliance")
        print(f"  Quality Grade: {'BUENO' if compliant_count >= 2 else 'INSUFICIENTE'}")

    print("\n  Key Metrics:")
    if "operationalization" in results:
        op = results["operationalization"]
        print(f"    • Indicator Completeness: {op.metrics['completeness_ratio']:.1%}")

    if "financial_traceability" in results:
        ft = results["financial_traceability"]
        print(f"    • Financial Codes Found: {ft.metrics['total_codes']}")

    if "quantified_gaps" in results:
        qg = results["quantified_gaps"]
        print(f"    • Gaps Quantified: {qg.metrics['quantification_ratio']:.1%}")

    if "systemic_risk" in results:
        sr = results["systemic_risk"]
        print(f"    • Systemic Risk Score: {sr.metrics['risk_score']:.3f}")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("EVIDENCE QUALITY AUDITORS - COMPREHENSIVE DEMONSTRATION")
    print("Part 3: Evidence Quality and Compliance (D1, D3, D4, D5 Audit)")
    print("=" * 80)

    demo_operationalization()
    demo_financial_traceability()
    demo_quantified_gaps()
    demo_systemic_risk()
    demo_integrated()

    print_separator()
    print("DEMONSTRATION COMPLETE")
    print("\nFor more information:")
    print("  • Full documentation: EVIDENCE_QUALITY_AUDITORS_README.md")
    print("  • Quick reference: EVIDENCE_QUALITY_QUICKREF.md")
    print("  • Source code: evidence_quality_auditors.py")
    print("  • Tests: test_evidence_quality_auditors.py (29 tests, all passing)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
