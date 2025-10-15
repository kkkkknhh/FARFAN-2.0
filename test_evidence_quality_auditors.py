#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Evidence Quality Auditors (Part 3)
==================================================

Comprehensive tests for all four auditors:
- OperationalizationAuditor (D3-Q1)
- FinancialTraceabilityAuditor (D1-Q3, D3-Q3)
- QuantifiedGapAuditor (D1-Q2)
- SystemicRiskAuditor (D4-Q5, D5-Q4)
"""

import unittest
from evidence_quality_auditors import (
    OperationalizationAuditor,
    FinancialTraceabilityAuditor,
    QuantifiedGapAuditor,
    SystemicRiskAuditor,
    IndicatorMetadata,
    AuditSeverity,
    run_all_audits
)


class TestOperationalizationAuditor(unittest.TestCase):
    """Test cases for OperationalizationAuditor (D3-Q1)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.auditor = OperationalizationAuditor(metadata_threshold=0.80)
    
    def test_complete_indicators_excellent(self):
        """Test that complete indicators receive excellent rating"""
        indicators = [
            IndicatorMetadata(
                codigo="IND-001",
                nombre="Tasa de cobertura educativa",
                linea_base=75.0,
                meta=90.0,
                fuente="DANE",
                formula="(Matriculados/Población)*100"
            ),
            IndicatorMetadata(
                codigo="IND-002",
                nombre="Tasa de deserción",
                linea_base=15.0,
                meta=5.0,
                fuente="SED",
                formula="(Desertores/Matriculados)*100"
            )
        ]
        
        result = self.auditor.audit_indicators(indicators)
        
        self.assertEqual(result.severity, AuditSeverity.EXCELLENT)
        self.assertTrue(result.sota_compliance)
        self.assertEqual(result.metrics['completeness_ratio'], 1.0)
        self.assertEqual(result.metrics['complete_indicators'], 2)
    
    def test_incomplete_indicators_requires_review(self):
        """Test that incomplete indicators receive lower rating"""
        indicators = [
            IndicatorMetadata(
                codigo="IND-001",
                nombre="Tasa de cobertura educativa",
                linea_base=75.0,
                # Missing meta, fuente, formula
            ),
            IndicatorMetadata(
                codigo="IND-002",
                nombre="Tasa de deserción",
                # All missing
            )
        ]
        
        result = self.auditor.audit_indicators(indicators)
        
        self.assertEqual(result.severity, AuditSeverity.REQUIRES_REVIEW)
        self.assertFalse(result.sota_compliance)
        self.assertEqual(result.metrics['completeness_ratio'], 0.0)
        self.assertEqual(result.metrics['incomplete_indicators'], 2)
        self.assertGreater(len(result.recommendations), 0)
    
    def test_threshold_boundary_good(self):
        """Test threshold boundary at 80%"""
        indicators = [
            # 4 complete
            IndicatorMetadata("I1", "N1", 1.0, 2.0, "F1", "form1"),
            IndicatorMetadata("I2", "N2", 1.0, 2.0, "F2", "form2"),
            IndicatorMetadata("I3", "N3", 1.0, 2.0, "F3", "form3"),
            IndicatorMetadata("I4", "N4", 1.0, 2.0, "F4", "form4"),
            # 1 incomplete (80% complete = exactly threshold)
            IndicatorMetadata("I5", "N5"),
        ]
        
        result = self.auditor.audit_indicators(indicators)
        
        self.assertIn(result.severity, [AuditSeverity.EXCELLENT, AuditSeverity.GOOD])
        self.assertAlmostEqual(result.metrics['completeness_ratio'], 0.80, places=2)
    
    def test_empty_indicators_list(self):
        """Test handling of empty indicators list"""
        result = self.auditor.audit_indicators([])
        
        self.assertIsNotNone(result)
        self.assertEqual(result.metrics['total_indicators'], 0)


class TestFinancialTraceabilityAuditor(unittest.TestCase):
    """Test cases for FinancialTraceabilityAuditor (D1-Q3, D3-Q3)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.auditor = FinancialTraceabilityAuditor(confidence_threshold=0.95)
    
    def test_bpin_code_detection(self):
        """Test detection of BPIN codes"""
        text = """
        Proyecto BPIN 2023000123456 busca mejorar la infraestructura educativa.
        Código de inversión 2024000987654 alineado con presupuesto.
        """
        
        result = self.auditor.audit_financial_codes(text)
        
        self.assertGreater(result.metrics['total_codes'], 0)
        self.assertGreater(result.metrics['bpin_codes'], 0)
        self.assertTrue(any('BPIN' in str(f) for f in result.findings))
    
    def test_ppi_code_detection(self):
        """Test detection of PPI codes"""
        text = """
        Plan Plurianual de Inversiones PPI-2023001234 establece prioridades.
        Código PPI 2024005678 para sector educativo.
        """
        
        result = self.auditor.audit_financial_codes(text)
        
        self.assertGreater(result.metrics['total_codes'], 0)
        self.assertGreater(result.metrics['ppi_codes'], 0)
    
    def test_no_codes_critical(self):
        """Test that absence of codes is critical"""
        text = "Proyecto de mejoramiento educativo sin códigos."
        
        result = self.auditor.audit_financial_codes(text)
        
        self.assertEqual(result.severity, AuditSeverity.CRITICAL)
        self.assertFalse(result.sota_compliance)
        self.assertEqual(result.metrics['total_codes'], 0)
        self.assertIn('CRÍTICO', result.recommendations[0])
    
    def test_mixed_codes(self):
        """Test detection of both BPIN and PPI codes"""
        text = """
        Proyecto financiado con código BPIN 2023000111111 y 
        Plan Plurianual PPI-2023222222.
        """
        
        result = self.auditor.audit_financial_codes(text)
        
        self.assertGreater(result.metrics['bpin_codes'], 0)
        self.assertGreater(result.metrics['ppi_codes'], 0)
    
    def test_confidence_calculation(self):
        """Test that confidence is calculated based on context"""
        text = """
        Código de proyecto BPIN 2023000123456 para inversión en educación.
        """
        
        result = self.auditor.audit_financial_codes(text)
        
        # At least one finding should exist
        self.assertGreater(len(result.findings), 0)
        
        # Check that confidence was calculated
        for finding in result.findings:
            self.assertIn('confidence', finding)
            self.assertGreaterEqual(finding['confidence'], 0.0)
            self.assertLessEqual(finding['confidence'], 1.0)


class TestQuantifiedGapAuditor(unittest.TestCase):
    """Test cases for QuantifiedGapAuditor (D1-Q2)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.auditor = QuantifiedGapAuditor()
    
    def test_deficit_detection(self):
        """Test detection of quantified deficits"""
        text = """
        El municipio presenta un déficit de 35% en cobertura educativa.
        Déficit de 1200 cupos en educación preescolar.
        """
        
        result = self.auditor.audit_quantified_gaps(text)
        
        self.assertGreater(result.metrics['total_gaps'], 0)
        deficit_gaps = result.metrics['gap_type_distribution'].get('déficit', 0)
        self.assertGreater(deficit_gaps, 0)
    
    def test_brecha_detection(self):
        """Test detection of quantified brechas"""
        text = """
        Brecha de 45% en acceso a servicios de salud.
        La brecha digital afecta a 3000 familias.
        """
        
        result = self.auditor.audit_quantified_gaps(text)
        
        brecha_gaps = result.metrics['gap_type_distribution'].get('brecha', 0)
        self.assertGreater(brecha_gaps, 0)
    
    def test_vacio_detection(self):
        """Test detection of data voids"""
        text = """
        Se identifican vacíos de información en zonas rurales.
        Vacío de datos sobre población vulnerable.
        """
        
        result = self.auditor.audit_quantified_gaps(text)
        
        vacio_gaps = result.metrics['gap_type_distribution'].get('vacío', 0)
        self.assertGreater(vacio_gaps, 0)
    
    def test_subregistro_detection(self):
        """Test detection of subregistro"""
        text = """
        Problema de subregistro en censos rurales.
        Sub-registro afecta estadísticas oficiales.
        """
        
        result = self.auditor.audit_quantified_gaps(text)
        
        self.assertGreater(result.metrics['subregistro_count'], 0)
        self.assertTrue(any('subregistro' in rec.lower() for rec in result.recommendations))
    
    def test_quantification_ratio(self):
        """Test quantification ratio calculation"""
        text = """
        Déficit de 35% en cobertura.
        Brecha de 1200 cupos.
        Vacío de información sin cuantificar.
        """
        
        result = self.auditor.audit_quantified_gaps(text)
        
        total = result.metrics['total_gaps']
        quantified = result.metrics['quantified_gaps']
        
        self.assertGreater(total, 0)
        self.assertGreater(quantified, 0)
        self.assertLess(quantified, total)  # Not all are quantified
    
    def test_high_quantification_excellent(self):
        """Test that high quantification leads to excellent rating"""
        text = """
        Déficit de 35% en educación.
        Brecha de 40% en salud.
        Déficit de 25% en vivienda.
        """
        
        result = self.auditor.audit_quantified_gaps(text)
        
        # All gaps are quantified
        self.assertEqual(result.metrics['total_gaps'], result.metrics['quantified_gaps'])
        self.assertEqual(result.severity, AuditSeverity.EXCELLENT)
        self.assertTrue(result.sota_compliance)
    
    def test_no_gaps_requires_review(self):
        """Test that no gaps detected requires review"""
        text = "Todo funciona perfectamente sin problemas."
        
        result = self.auditor.audit_quantified_gaps(text)
        
        self.assertEqual(result.metrics['total_gaps'], 0)
        self.assertEqual(result.severity, AuditSeverity.REQUIRES_REVIEW)


class TestSystemicRiskAuditor(unittest.TestCase):
    """Test cases for SystemicRiskAuditor (D4-Q5, D5-Q4)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.auditor = SystemicRiskAuditor(excellent_threshold=0.10)
    
    def test_pnd_detection(self):
        """Test detection of PND alignment"""
        text = """
        Proyecto alineado con Plan Nacional de Desarrollo (PND).
        Contribuye a las metas del PND 2023-2026.
        """
        
        result = self.auditor.audit_systemic_risk(text)
        
        self.assertTrue(result.metrics['pnd_alignment'])
        pnd_finding = next(f for f in result.findings if f['aspect'] == 'PND_alignment')
        self.assertTrue(pnd_finding['aligned'])
    
    def test_ods_detection(self):
        """Test detection of ODS alignment"""
        text = """
        Alineado con ODS-4 (Educación de Calidad).
        Contribuye a ODS-10 (Reducción de Desigualdades) y ODS-16.
        """
        
        result = self.auditor.audit_systemic_risk(text)
        
        ods_numbers = result.metrics['ods_numbers']
        self.assertIn(4, ods_numbers)
        self.assertIn(10, ods_numbers)
        self.assertIn(16, ods_numbers)
        self.assertEqual(result.metrics['ods_count'], 3)
    
    def test_sdg_alternative_format(self):
        """Test detection of SDG format"""
        text = "Aligned with SDG-17 (Partnerships for the Goals)."
        
        result = self.auditor.audit_systemic_risk(text)
        
        self.assertIn(17, result.metrics['ods_numbers'])
    
    def test_no_alignment_high_risk(self):
        """Test that no alignment increases risk score"""
        text = "Proyecto municipal sin referencias a marcos nacionales."
        
        result = self.auditor.audit_systemic_risk(text)
        
        self.assertFalse(result.metrics['pnd_alignment'])
        self.assertEqual(result.metrics['ods_count'], 0)
        self.assertGreater(result.metrics['risk_score'], 0.30)
        self.assertEqual(result.severity, AuditSeverity.REQUIRES_REVIEW)
    
    def test_full_alignment_excellent(self):
        """Test that full alignment achieves excellent rating"""
        text = """
        Proyecto totalmente alineado con Plan Nacional de Desarrollo.
        Contribuye a ODS-1, ODS-4 y ODS-11.
        """
        
        result = self.auditor.audit_systemic_risk(text)
        
        self.assertTrue(result.metrics['pnd_alignment'])
        self.assertGreater(result.metrics['ods_count'], 0)
        self.assertLess(result.metrics['risk_score'], 0.15)
        self.assertIn(result.severity, [AuditSeverity.EXCELLENT, AuditSeverity.GOOD])
    
    def test_risk_score_calculation(self):
        """Test risk score calculation logic"""
        # No alignment
        text1 = "Proyecto sin alineación."
        result1 = self.auditor.audit_systemic_risk(text1)
        
        # Partial alignment
        text2 = "Proyecto alineado con ODS-4."
        result2 = self.auditor.audit_systemic_risk(text2)
        
        # Full alignment
        text3 = "Proyecto alineado con PND y ODS-4, ODS-10, ODS-11."
        result3 = self.auditor.audit_systemic_risk(text3)
        
        # Risk should decrease with more alignment
        self.assertGreater(result1.metrics['risk_score'], result2.metrics['risk_score'])
        self.assertGreater(result2.metrics['risk_score'], result3.metrics['risk_score'])
    
    def test_misalignment_reasons(self):
        """Test that misalignment reasons are captured"""
        text = "Proyecto sin alineación."
        
        result = self.auditor.audit_systemic_risk(text)
        
        reasons = result.evidence['misalignment_reasons']
        self.assertGreater(len(reasons), 0)
        self.assertTrue(any('PND' in reason for reason in reasons))
        self.assertTrue(any('ODS' in reason for reason in reasons))


class TestIntegration(unittest.TestCase):
    """Integration tests for all auditors"""
    
    def test_run_all_audits(self):
        """Test running all audits together"""
        text = """
        Proyecto BPIN 2023000123456 busca reducir el déficit de 35% en educación.
        Alineado con PND y ODS-4.
        Se identifican vacíos de información en zonas rurales.
        """
        
        results = run_all_audits(text=text)
        
        # Should have all auditor results
        self.assertIn('financial_traceability', results)
        self.assertIn('quantified_gaps', results)
        self.assertIn('systemic_risk', results)
        
        # All should have required structure
        for audit_type, result in results.items():
            self.assertIsNotNone(result.audit_type)
            self.assertIsNotNone(result.timestamp)
            self.assertIsInstance(result.findings, list)
            self.assertIsInstance(result.metrics, dict)
            self.assertIsInstance(result.recommendations, list)
    
    def test_comprehensive_pdm_analysis(self):
        """Test comprehensive PDM analysis"""
        comprehensive_text = """
        PLAN DE DESARROLLO MUNICIPAL 2024-2027
        
        DIAGNÓSTICO:
        El municipio presenta un déficit de 40% en cobertura educativa y 
        brecha de 35% en acceso a salud. Se identifican vacíos de información
        en población rural y subregistro en censos.
        
        ESTRATEGIA:
        Proyecto BPIN 2024000111111 y PPI-2024222222 para inversión educativa.
        Alineado con Plan Nacional de Desarrollo y ODS-4 (Educación de Calidad),
        ODS-10 (Reducción de Desigualdades).
        
        INDICADORES:
        - Tasa de cobertura educativa: LB 60%, Meta 90%, Fuente: SED
        - Fórmula: (Matriculados/Población en edad escolar) * 100
        """
        
        # Create sample indicators
        indicators = [
            IndicatorMetadata(
                codigo="IND-EDU-001",
                nombre="Tasa de cobertura educativa",
                linea_base=60.0,
                meta=90.0,
                fuente="SED",
                formula="(Matriculados/Población en edad escolar) * 100"
            )
        ]
        
        results = run_all_audits(
            text=comprehensive_text,
            indicators=indicators
        )
        
        # Verify all audits ran successfully
        self.assertEqual(len(results), 4)  # All 4 auditors
        
        # Check financial traceability found codes
        ft_result = results['financial_traceability']
        self.assertGreater(ft_result.metrics['total_codes'], 0)
        
        # Check quantified gaps found deficits
        qg_result = results['quantified_gaps']
        self.assertGreater(qg_result.metrics['total_gaps'], 0)
        
        # Check systemic risk found alignment
        sr_result = results['systemic_risk']
        self.assertTrue(sr_result.metrics['pnd_alignment'])
        self.assertGreater(sr_result.metrics['ods_count'], 0)
        
        # Check operationalization found complete indicators
        op_result = results['operationalization']
        self.assertEqual(op_result.metrics['complete_indicators'], 1)


class TestSOTACompliance(unittest.TestCase):
    """Test SOTA (State of the Art) compliance references"""
    
    def test_operationalization_sota_references(self):
        """Test that operationalization auditor includes SOTA references"""
        auditor = OperationalizationAuditor()
        indicators = [
            IndicatorMetadata("I1", "N1", 1.0, 2.0, "F1", "form1")
        ]
        
        result = auditor.audit_indicators(indicators)
        
        self.assertIn('ods_alignment_benchmark', result.evidence)
        self.assertIn('bayesian_updating_reference', result.evidence)
        self.assertEqual(result.evidence['ods_alignment_benchmark'], 'UN 2020')
        self.assertEqual(result.evidence['bayesian_updating_reference'], 'Gelman 2013')
    
    def test_financial_traceability_sota_references(self):
        """Test that financial traceability auditor includes SOTA references"""
        auditor = FinancialTraceabilityAuditor()
        text = "Proyecto BPIN 2023000123456"
        
        result = auditor.audit_financial_codes(text)
        
        self.assertIn('dnp_standard', result.evidence)
        self.assertIn('fiscal_illusions_reference', result.evidence)
        self.assertEqual(result.evidence['dnp_standard'], 'Colombian DNP 2023')
        self.assertEqual(result.evidence['fiscal_illusions_reference'], 'Waldner 2015')
    
    def test_quantified_gaps_sota_references(self):
        """Test that quantified gaps auditor includes SOTA references"""
        auditor = QuantifiedGapAuditor()
        text = "Déficit de 35%"
        
        result = auditor.audit_quantified_gaps(text)
        
        self.assertIn('qca_calibration_reference', result.evidence)
        self.assertEqual(result.evidence['qca_calibration_reference'], 'Ragin 2008')
    
    def test_systemic_risk_sota_references(self):
        """Test that systemic risk auditor includes SOTA references"""
        auditor = SystemicRiskAuditor()
        text = "Alineado con PND y ODS-4"
        
        result = auditor.audit_systemic_risk(text)
        
        self.assertIn('counterfactual_rigor_reference', result.evidence)
        self.assertEqual(result.evidence['counterfactual_rigor_reference'], 'Pearl 2018')


if __name__ == '__main__':
    # Run all tests with verbose output
    unittest.main(verbosity=2)
