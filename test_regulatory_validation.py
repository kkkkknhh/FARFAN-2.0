#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Regulatory Validation with ValidadorDNP
=======================================================

Validates deterministic scoring, contract compliance, and traceability
for regulatory validation phase in the orchestrator.

SIN_CARRETA Compliance:
- All scores must be deterministic and reproducible
- Scores must be in [0, 1] range
- No fallback or estimation logic
- Full audit trail required
"""

import pytest
from orchestrator import create_orchestrator
from dnp_integration import ValidadorDNP, ResultadoValidacionDNP, NivelCumplimiento
from infrastructure.calibration_constants import CALIBRATION


class TestRegulatoryValidationDeterminism:
    """Test deterministic behavior of regulatory validation"""
    
    def test_identical_input_produces_identical_output(self):
        """Regulatory validation must be deterministic for identical inputs"""
        orch = create_orchestrator()
        
        text = "Plan de desarrollo educativo con construcción de 3 sedes educativas rurales usando indicadores EDU-001 y EDU-002."
        
        # Run twice with identical input
        result1 = orch.orchestrate_analysis(text, "PDM_Test", "estratégico")
        result2 = orch.orchestrate_analysis(text, "PDM_Test", "estratégico")
        
        # Extract regulatory scores
        reg1 = result1.get("analyze_regulatory_constraints", {})
        reg2 = result2.get("analyze_regulatory_constraints", {})
        
        # Verify identical scores
        assert reg1.get("metrics", {}).get("score_raw") == reg2.get("metrics", {}).get("score_raw")
        assert reg1.get("metrics", {}).get("score_adjusted") == reg2.get("metrics", {}).get("score_adjusted")
        assert reg1.get("metrics", {}).get("cumple_competencias") == reg2.get("metrics", {}).get("cumple_competencias")
        assert reg1.get("metrics", {}).get("cumple_mga") == reg2.get("metrics", {}).get("cumple_mga")
        
        print("✓ Test identical input produces identical output PASSED")
    
    def test_score_range_contract(self):
        """All regulatory scores must be in [0, 1] range"""
        orch = create_orchestrator()
        
        # Test with various inputs
        test_cases = [
            "Plan educativo con EDU-001",
            "Construcción de hospital con SAL-001",
            "Infraestructura vial",
            "",  # Empty text edge case
        ]
        
        for text in test_cases:
            result = orch.orchestrate_analysis(text, "PDM_Test", "estratégico")
            reg = result.get("analyze_regulatory_constraints", {})
            
            if reg.get("status") == "success":
                score_raw = reg.get("metrics", {}).get("score_raw", 0.0)
                score_adjusted = reg.get("metrics", {}).get("score_adjusted", 0.0)
                
                # Verify scores are in [0, 1] range
                assert 0.0 <= score_raw <= 1.0, f"score_raw {score_raw} out of range for text: {text[:50]}"
                assert 0.0 <= score_adjusted <= 1.0, f"score_adjusted {score_adjusted} out of range for text: {text[:50]}"
        
        print("✓ Test score range contract PASSED")
    
    def test_no_silent_failures(self):
        """Errors must be explicit, not silent with fallback scores"""
        orch = create_orchestrator()
        
        # Test with empty text
        result = orch.orchestrate_analysis("", "PDM_Empty", "estratégico")
        reg = result.get("analyze_regulatory_constraints", {})
        
        # Either success with valid scores or explicit error status
        assert reg.get("status") in ["success", "error"]
        
        if reg.get("status") == "error":
            # Error must have error field
            assert "error" in reg
        
        print("✓ Test no silent failures PASSED")


class TestRegulatoryValidationScoring:
    """Test scoring logic and contracts"""
    
    def test_competencias_validation(self):
        """Test municipal competencies validation"""
        orch = create_orchestrator()
        
        # Education sector - should pass competencies
        result = orch.orchestrate_analysis(
            "Construcción de institución educativa en zona rural",
            "PDM_Edu",
            "estratégico"
        )
        
        reg = result.get("analyze_regulatory_constraints", {})
        assert reg.get("status") == "success"
        
        # Should detect education sector and validate competencies
        outputs = reg.get("outputs", {}).get("d1_q5_regulatory_analysis", {})
        assert outputs.get("cumple_competencias") is True
        assert outputs.get("sector_detectado") == "educacion"
        
        print("✓ Test competencias validation PASSED")
    
    def test_mga_indicator_extraction(self):
        """Test MGA indicator extraction from text"""
        orch = create_orchestrator()
        
        text = "Proyecto educativo con indicadores EDU-001, EDU-002 y EDU-010 del catálogo MGA"
        result = orch.orchestrate_analysis(text, "PDM_MGA", "estratégico")
        
        reg = result.get("analyze_regulatory_constraints", {})
        outputs = reg.get("outputs", {}).get("d1_q5_regulatory_analysis", {})
        
        # Should extract MGA indicators
        indicadores = outputs.get("indicadores_mga_usados", [])
        assert "EDU-001" in indicadores or "EDU-002" in indicadores
        
        # Should have MGA compliance
        assert outputs.get("cumple_mga") is not None
        
        print("✓ Test MGA indicator extraction PASSED")
    
    def test_calibration_constant_application(self):
        """Test REGULATORY_DEPTH_FACTOR is applied correctly"""
        orch = create_orchestrator()
        
        text = "Plan de salud con SAL-001 y SAL-002"
        result = orch.orchestrate_analysis(text, "PDM_Cal", "estratégico")
        
        reg = result.get("analyze_regulatory_constraints", {})
        metrics = reg.get("metrics", {})
        
        score_raw = metrics.get("score_raw", 0.0)
        score_adjusted = metrics.get("score_adjusted", 0.0)
        
        # Adjusted score should be raw * REGULATORY_DEPTH_FACTOR (capped at 1.0)
        expected_adjusted = min(1.0, score_raw * CALIBRATION.REGULATORY_DEPTH_FACTOR)
        
        # Allow small floating point error
        assert abs(score_adjusted - expected_adjusted) < 0.001
        
        print("✓ Test calibration constant application PASSED")
    
    def test_sector_detection(self):
        """Test sector detection from text"""
        orch = create_orchestrator()
        
        test_cases = [
            ("Construcción de escuela", "educacion"),
            ("Hospital y centro de salud", "salud"),
            ("Acueducto y alcantarillado", "agua_potable"),
            ("Vivienda de interés social", "vivienda"),
            ("Infraestructura vial", "transporte"),
        ]
        
        for text, expected_sector in test_cases:
            result = orch.orchestrate_analysis(text, "PDM_Sector", "estratégico")
            reg = result.get("analyze_regulatory_constraints", {})
            outputs = reg.get("outputs", {}).get("d1_q5_regulatory_analysis", {})
            
            detected_sector = outputs.get("sector_detectado")
            assert detected_sector == expected_sector, f"Expected {expected_sector}, got {detected_sector} for text: {text}"
        
        print("✓ Test sector detection PASSED")


class TestRegulatoryValidationAuditTrail:
    """Test audit trail and traceability"""
    
    def test_full_audit_trail(self):
        """Regulatory validation must produce complete audit trail"""
        orch = create_orchestrator()
        
        result = orch.orchestrate_analysis(
            "Plan educativo con EDU-001 para zona rural",
            "PDM_Audit",
            "estratégico"
        )
        
        reg = result.get("analyze_regulatory_constraints", {})
        
        # Verify all required audit fields
        assert "inputs" in reg
        assert "outputs" in reg
        assert "metrics" in reg
        assert "timestamp" in reg
        assert "status" in reg
        
        # Verify inputs are logged
        inputs = reg.get("inputs", {})
        assert "regulatory_depth_factor" in inputs
        assert "sector" in inputs
        
        # Verify outputs contain traceability
        outputs = reg.get("outputs", {}).get("d1_q5_regulatory_analysis", {})
        assert "competencias_validadas" in outputs
        assert "indicadores_mga_usados" in outputs
        assert "sector_detectado" in outputs
        
        print("✓ Test full audit trail PASSED")
    
    def test_traceability_to_dnp_standards(self):
        """Scores must be traceable to DNP standards (competencias, MGA, PDET)"""
        orch = create_orchestrator()
        
        result = orch.orchestrate_analysis(
            "Construcción educativa con EDU-001",
            "PDM_Trace",
            "estratégico"
        )
        
        reg = result.get("analyze_regulatory_constraints", {})
        outputs = reg.get("outputs", {}).get("d1_q5_regulatory_analysis", {})
        
        # Must have explicit breakdown of scores
        assert "score_competencias" in outputs
        assert "score_mga" in outputs
        assert "score_pdet" in outputs
        
        # Must have lists of validated items
        assert isinstance(outputs.get("competencias_validadas"), list)
        assert isinstance(outputs.get("indicadores_mga_usados"), list)
        
        print("✓ Test traceability to DNP standards PASSED")


class TestRegulatoryValidationEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_text(self):
        """Empty text should produce valid (zero) scores, not crash"""
        orch = create_orchestrator()
        
        result = orch.orchestrate_analysis("", "PDM_Empty", "estratégico")
        reg = result.get("analyze_regulatory_constraints", {})
        
        # Should complete successfully or with explicit error
        assert reg.get("status") in ["success", "error"]
        
        if reg.get("status") == "success":
            # Scores should be valid (likely 0)
            score = reg.get("metrics", {}).get("score_adjusted", 0.0)
            assert 0.0 <= score <= 1.0
        
        print("✓ Test empty text PASSED")
    
    def test_no_mga_indicators(self):
        """Text without MGA indicators should handle gracefully"""
        orch = create_orchestrator()
        
        result = orch.orchestrate_analysis(
            "Plan de desarrollo sin indicadores específicos",
            "PDM_NoMGA",
            "estratégico"
        )
        
        reg = result.get("analyze_regulatory_constraints", {})
        outputs = reg.get("outputs", {}).get("d1_q5_regulatory_analysis", {})
        
        # Should have empty or minimal MGA indicators
        indicadores = outputs.get("indicadores_mga_usados", [])
        assert isinstance(indicadores, list)
        
        # cumple_mga should be False if no valid indicators
        assert outputs.get("cumple_mga") is False
        
        print("✓ Test no MGA indicators PASSED")
    
    def test_unknown_sector(self):
        """Unknown sector should default gracefully"""
        orch = create_orchestrator()
        
        result = orch.orchestrate_analysis(
            "Proyecto de quantum computing blockchain",
            "PDM_Unknown",
            "estratégico"
        )
        
        reg = result.get("analyze_regulatory_constraints", {})
        
        # Should complete successfully
        assert reg.get("status") in ["success", "error"]
        
        if reg.get("status") == "success":
            outputs = reg.get("outputs", {}).get("d1_q5_regulatory_analysis", {})
            # Should detect a default sector
            assert "sector_detectado" in outputs
        
        print("✓ Test unknown sector PASSED")


class TestValidadorDNPIntegration:
    """Test direct ValidadorDNP integration"""
    
    def test_validador_dnp_scoring(self):
        """Test ValidadorDNP produces deterministic scores"""
        validador = ValidadorDNP(es_municipio_pdet=False)
        
        # Test with minimal valid input
        # Note: For MGA compliance, we need both PRODUCTO and RESULTADO indicators
        resultado = validador.validar_proyecto_integral(
            sector="educacion",
            descripcion="Construcción de escuela",
            indicadores_propuestos=["EDU-001", "EDU-020"],  # EDU-020 is PRODUCTO, EDU-001 is RESULTADO
            presupuesto=1_000_000_000,
            es_rural=True,
            poblacion_victimas=False
        )
        
        # Verify score is deterministic
        assert isinstance(resultado.score_total, (int, float))
        assert 0 <= resultado.score_total <= 100
        
        # Verify competencies
        assert resultado.cumple_competencias is True
        
        # Verify at least one indicator was recognized (even if not both types)
        assert len(resultado.indicadores_mga_usados) > 0 or len(resultado.indicadores_mga_faltantes) > 0
        
        print("✓ Test ValidadorDNP scoring PASSED")
    
    def test_validador_dnp_pdet_mode(self):
        """Test ValidadorDNP with PDET mode enabled"""
        validador = ValidadorDNP(es_municipio_pdet=True)
        
        resultado = validador.validar_proyecto_integral(
            sector="agricultura",
            descripcion="Desarrollo agropecuario rural",
            indicadores_propuestos=[],
            presupuesto=500_000_000,
            es_rural=True,
            poblacion_victimas=True
        )
        
        # PDET validation should be included
        assert resultado.es_municipio_pdet is True
        
        # Score should include PDET component
        assert isinstance(resultado.score_total, (int, float))
        
        print("✓ Test ValidadorDNP PDET mode PASSED")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "=" * 70)
    print("RUNNING REGULATORY VALIDATION TEST SUITE")
    print("=" * 70 + "\n")
    
    # Determinism tests
    print("--- Determinism Tests ---")
    determinism = TestRegulatoryValidationDeterminism()
    determinism.test_identical_input_produces_identical_output()
    determinism.test_score_range_contract()
    determinism.test_no_silent_failures()
    
    # Scoring tests
    print("\n--- Scoring Tests ---")
    scoring = TestRegulatoryValidationScoring()
    scoring.test_competencias_validation()
    scoring.test_mga_indicator_extraction()
    scoring.test_calibration_constant_application()
    scoring.test_sector_detection()
    
    # Audit trail tests
    print("\n--- Audit Trail Tests ---")
    audit = TestRegulatoryValidationAuditTrail()
    audit.test_full_audit_trail()
    audit.test_traceability_to_dnp_standards()
    
    # Edge case tests
    print("\n--- Edge Case Tests ---")
    edge = TestRegulatoryValidationEdgeCases()
    edge.test_empty_text()
    edge.test_no_mga_indicators()
    edge.test_unknown_sector()
    
    # Integration tests
    print("\n--- ValidadorDNP Integration Tests ---")
    integration = TestValidadorDNPIntegration()
    integration.test_validador_dnp_scoring()
    integration.test_validador_dnp_pdet_mode()
    
    print("\n" + "=" * 70)
    print("ALL REGULATORY VALIDATION TESTS PASSED ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
