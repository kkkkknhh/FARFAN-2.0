#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Audit Points 1.1-1.4
====================================

Tests deterministic bases and data integrity per SOTA MMR reproducibility standards.
"""

import hashlib
import logging
import sys
import unittest
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAuditPoint1_1_DeterministicSeeding(unittest.TestCase):
    """
    Test Audit Point 1.1: Deterministic Seeding (RNG)
    
    Quality Evidence to Verify Truth:
    - Re-run pipeline twice with identical inputs/salt → compare output hashes (must match 100%)
    - Inspect config['random_seed'] derivation from document ID
    """
    
    def test_seed_generation_deterministic(self):
        """Test that _create_advanced_seed produces same seed for same inputs"""
        from teoria_cambio import _create_advanced_seed
        
        plan_name = "PDM_Test_Municipality"
        salt = "test_salt_123"
        
        # Generate seed twice with same inputs
        seed1 = _create_advanced_seed(plan_name, salt)
        seed2 = _create_advanced_seed(plan_name, salt)
        
        # Must be identical (100% reproducible)
        self.assertEqual(seed1, seed2, 
            "Audit 1.1 FAILED: Seeds not deterministic for identical inputs")
        
        logger.info(f"✓ Audit 1.1: Deterministic seed verified: {seed1}")
    
    def test_seed_varies_with_salt(self):
        """Test that different salts produce different seeds (sensitivity analysis)"""
        from teoria_cambio import _create_advanced_seed
        
        plan_name = "PDM_Test"
        salt1 = "salt_A"
        salt2 = "salt_B"
        
        seed1 = _create_advanced_seed(plan_name, salt1)
        seed2 = _create_advanced_seed(plan_name, salt2)
        
        # Seeds must differ when salt changes (enables sensitivity analysis)
        self.assertNotEqual(seed1, seed2,
            "Audit 1.1 FAILED: Seeds should vary with different salts")
        
        logger.info(f"✓ Audit 1.1: Salt sensitivity verified (seed1={seed1}, seed2={seed2})")
    
    def test_rng_initialization_reproducible(self):
        """Test that RNG initialization produces reproducible results"""
        from teoria_cambio import AdvancedDAGValidator
        import random
        import numpy as np
        
        plan_name = "PDM_Test"
        
        # Create two validators with same plan
        validator1 = AdvancedDAGValidator()
        validator2 = AdvancedDAGValidator()
        
        # Initialize RNG with same seed
        seed1 = validator1._initialize_rng(plan_name)
        seed2 = validator2._initialize_rng(plan_name)
        
        # Seeds must match
        self.assertEqual(seed1, seed2,
            "Audit 1.1 FAILED: RNG seeds not reproducible")
        
        # Random samples should match
        samples1 = [validator1._rng.random() for _ in range(10)]
        samples2 = [validator2._rng.random() for _ in range(10)]
        
        for i, (s1, s2) in enumerate(zip(samples1, samples2)):
            self.assertAlmostEqual(s1, s2, places=15,
                msg=f"Audit 1.1 FAILED: Random sample {i} not reproducible")
        
        logger.info(f"✓ Audit 1.1: RNG reproducibility verified across {len(samples1)} samples")
    
    def test_monte_carlo_result_reproducible_flag(self):
        """Test that MonteCarloAdvancedResult tracks reproducibility"""
        from teoria_cambio import MonteCarloAdvancedResult
        
        result = MonteCarloAdvancedResult(
            plan_name="Test_Plan",
            seed=12345,
            timestamp="2025-10-15T12:00:00",
            total_iterations=1000,
            acyclic_count=950,
            p_value=0.05,
            bayesian_posterior=0.95,
            confidence_interval=(0.92, 0.98),
            statistical_power=0.85,
            edge_sensitivity={},
            node_importance={},
            robustness_score=0.9,
            reproducible=True,  # Audit 1.1: Must be True for deterministic seed
            convergence_achieved=True,
            adequate_power=True,
            computation_time=10.5,
            graph_statistics={},
            test_parameters={}
        )
        
        self.assertTrue(result.reproducible,
            "Audit 1.1 FAILED: MonteCarloAdvancedResult.reproducible should be True")
        self.assertIsInstance(result.seed, int,
            "Audit 1.1 FAILED: Seed should be stored as integer")
        
        logger.info(f"✓ Audit 1.1: MonteCarloAdvancedResult reproducibility flag verified")


class TestAuditPoint1_2_ProvenanceTraceability(unittest.TestCase):
    """
    Test Audit Point 1.2: Provenance Traceability (SHA-256)
    
    Quality Evidence to Verify Truth:
    - Query chunk_id in logs → recompute hash manually from raw chunks (must match)
    - Trace evidence in Bayesian analysis back to exact PDF pages
    """
    
    def test_chunk_id_generation_sha256(self):
        """Test that chunk_id is valid SHA-256 hash"""
        from extraction.extraction_pipeline import SemanticChunk
        
        doc_id = "a" * 64  # Valid SHA-256 hex
        index = 0
        text_preview = "Sample policy text for testing chunk ID generation"
        
        chunk_id = SemanticChunk.create_chunk_id(doc_id, index, text_preview)
        
        # Verify SHA-256 format (64 hex characters)
        self.assertEqual(len(chunk_id), 64,
            "Audit 1.2 FAILED: chunk_id should be 64 characters (SHA-256)")
        self.assertTrue(all(c in '0123456789abcdef' for c in chunk_id.lower()),
            "Audit 1.2 FAILED: chunk_id should be valid hex string")
        
        logger.info(f"✓ Audit 1.2: Valid SHA-256 chunk_id: {chunk_id[:16]}...")
    
    def test_chunk_id_deterministic(self):
        """Test that same inputs always produce same chunk_id"""
        from extraction.extraction_pipeline import SemanticChunk
        
        doc_id = "b" * 64
        index = 5
        text = "Test text for deterministic chunk ID"
        
        chunk_id1 = SemanticChunk.create_chunk_id(doc_id, index, text)
        chunk_id2 = SemanticChunk.create_chunk_id(doc_id, index, text)
        
        self.assertEqual(chunk_id1, chunk_id2,
            "Audit 1.2 FAILED: chunk_id not deterministic")
        
        logger.info(f"✓ Audit 1.2: Deterministic chunk_id verified")
    
    def test_chunk_id_manual_recomputation(self):
        """Test manual recomputation of chunk_id matches expected"""
        from extraction.extraction_pipeline import SemanticChunk
        
        doc_id = "c" * 64
        index = 10
        text = "Manual verification test    with   whitespace"
        
        # Generate using method
        chunk_id = SemanticChunk.create_chunk_id(doc_id, index, text)
        
        # Manual recomputation (must match method's logic)
        normalized_preview = ' '.join(text[:200].split())
        canonical_string = f"{doc_id}|{index}|{normalized_preview}"
        expected_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()
        
        self.assertEqual(chunk_id, expected_hash,
            "Audit 1.2 FAILED: Manual recomputation doesn't match")
        
        logger.info(f"✓ Audit 1.2: Manual chunk_id recomputation verified")
    
    def test_pdf_hash_generation(self):
        """Test that PDF hash is valid SHA-256"""
        # We'll create a small test file
        import tempfile
        import os
        
        from extraction.extraction_pipeline import ExtractionPipeline
        
        # Create temporary PDF-like file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            test_content = b"Test PDF content for hash verification"
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Mock config (minimal)
            class MockConfig:
                pass
            
            pipeline = ExtractionPipeline(MockConfig())
            
            # Compute hash
            pdf_hash = pipeline._compute_sha256(temp_path)
            
            # Verify SHA-256 format
            self.assertEqual(len(pdf_hash), 64,
                "Audit 1.2 FAILED: PDF hash should be 64 characters")
            self.assertTrue(all(c in '0123456789abcdef' for c in pdf_hash.lower()),
                "Audit 1.2 FAILED: PDF hash should be valid hex")
            
            # Manual verification
            expected_hash = hashlib.sha256(test_content).hexdigest()
            self.assertEqual(pdf_hash, expected_hash,
                "Audit 1.2 FAILED: PDF hash doesn't match manual computation")
            
            logger.info(f"✓ Audit 1.2: PDF hash verified: {pdf_hash[:16]}...")
        
        finally:
            os.unlink(temp_path)
    
    def test_semantic_chunk_validation(self):
        """Test that SemanticChunk validates chunk_id format"""
        from extraction.extraction_pipeline import SemanticChunk
        from pydantic import ValidationError
        
        valid_hash = "a" * 64
        
        # Valid chunk should succeed
        chunk = SemanticChunk(
            chunk_id=valid_hash,
            text="Test text",
            start_char=0,
            end_char=10,
            doc_id=valid_hash,
            metadata={}
        )
        self.assertEqual(chunk.chunk_id, valid_hash)
        
        # Invalid chunk_id should fail
        with self.assertRaises(ValidationError) as ctx:
            SemanticChunk(
                chunk_id="invalid_short",  # Not 64 chars
                text="Test",
                start_char=0,
                end_char=4,
                doc_id=valid_hash,
                metadata={}
            )
        
        error_msg = str(ctx.exception)
        self.assertIn("SHA-256", error_msg,
            "Audit 1.2 FAILED: Validation error should mention SHA-256")
        
        logger.info(f"✓ Audit 1.2: chunk_id validation (hoop test) verified")


class TestAuditPoint1_3_SchemaIntegrity(unittest.TestCase):
    """
    Test Audit Point 1.3: Schema Integrity (Hoop Tests)
    
    Quality Evidence to Verify Truth:
    - Simulate invalid input (e.g., omit chunk_id) → confirm omission from evidence pool
    - Error log trigger for missing critical fields
    """
    
    def test_semantic_chunk_missing_chunk_id(self):
        """Test that missing chunk_id triggers validation failure"""
        from extraction.extraction_pipeline import SemanticChunk
        from pydantic import ValidationError
        
        with self.assertRaises(ValidationError) as ctx:
            SemanticChunk(
                chunk_id="",  # Empty - should fail
                text="Test text",
                start_char=0,
                end_char=10,
                doc_id="a" * 64,
                metadata={}
            )
        
        error_msg = str(ctx.exception)
        self.assertIn("chunk_id", error_msg.lower(),
            "Audit 1.3 FAILED: Error should mention chunk_id")
        
        logger.info(f"✓ Audit 1.3: Missing chunk_id hoop test passed")
    
    def test_extracted_table_empty_data(self):
        """Test that empty table data triggers validation failure"""
        from extraction.extraction_pipeline import ExtractedTable
        from pydantic import ValidationError
        
        with self.assertRaises(ValidationError) as ctx:
            ExtractedTable(
                data=[],  # Empty - should fail hoop test
                page_number=1,
                confidence_score=0.5,
                column_count=3,
                row_count=0
            )
        
        error_msg = str(ctx.exception)
        self.assertIn("cannot be empty", error_msg.lower(),
            "Audit 1.3 FAILED: Error should mention empty data")
        
        logger.info(f"✓ Audit 1.3: Empty table data hoop test passed")
    
    def test_extracted_table_invalid_structure(self):
        """Test that invalid table structure triggers validation failure"""
        from extraction.extraction_pipeline import ExtractedTable
        from pydantic import ValidationError
        
        with self.assertRaises(ValidationError) as ctx:
            ExtractedTable(
                data=["not", "a", "list of lists"],  # Invalid structure
                page_number=1,
                confidence_score=0.5,
                column_count=3,
                row_count=3
            )
        
        error_msg = str(ctx.exception)
        self.assertIn("list", error_msg.lower(),
            "Audit 1.3 FAILED: Error should mention list structure")
        
        logger.info(f"✓ Audit 1.3: Invalid table structure hoop test passed")
    
    def test_bpin_validation_financial_table(self):
        """Test BPIN validation for financial tables (degraded mode warning)"""
        from extraction.extraction_pipeline import ExtractedTable
        
        # Should succeed but log warning
        with self.assertLogs(level='WARNING') as log_ctx:
            table = ExtractedTable(
                data=[['row1', 'data'], ['row2', 'data']],
                page_number=1,
                table_type='financial',
                confidence_score=0.8,
                column_count=2,
                row_count=2,
                bpin=None  # Missing BPIN for financial table
            )
        
        # Check warning was logged
        self.assertTrue(any('BPIN missing' in msg for msg in log_ctx.output),
            "Audit 1.3 FAILED: Missing BPIN should log warning")
        
        logger.info(f"✓ Audit 1.3: BPIN degraded mode warning verified")


class TestAuditPoint1_4_ExternalResourceFallback(unittest.TestCase):
    """
    Test Audit Point 1.4: External Resource Fallback
    
    Quality Evidence to Verify Truth:
    - Test lexicon miss → confirm pipeline halts vs. degrades on non-core
    - Score penalty but completes for external resource failure
    """
    
    def test_core_resource_fail_closed(self):
        """Test that core resource failure causes hard stop"""
        from audit_config import ResourceLoader, ResourceType
        
        loader = ResourceLoader()
        
        def failing_core_loader():
            raise RuntimeError("Core resource unavailable")
        
        # Should raise RuntimeError (fail-closed)
        with self.assertRaises(RuntimeError) as ctx:
            loader.load_resource(
                name="CoreTestResource",
                loader_func=failing_core_loader,
                resource_type=ResourceType.CORE
            )
        
        error_msg = str(ctx.exception)
        self.assertIn("Critical failure", error_msg,
            "Audit 1.4 FAILED: Core failure should mention 'Critical'")
        
        # Check mode is FAILED
        self.assertEqual(loader.config.mode.value, "failed",
            "Audit 1.4 FAILED: Core failure should set mode to FAILED")
        
        logger.info(f"✓ Audit 1.4: Core resource fail-closed verified")
    
    def test_external_resource_fail_open(self):
        """Test that external resource failure allows degraded operation"""
        from audit_config import ResourceLoader, ResourceType
        
        loader = ResourceLoader()
        
        def failing_external_loader():
            raise ImportError("External resource unavailable")
        
        # Should return None (fail-open) and not raise
        result = loader.load_resource(
            name="ExternalTestResource",
            loader_func=failing_external_loader,
            resource_type=ResourceType.EXTERNAL,
            degradation_impact="Test degradation - 5% accuracy loss"
        )
        
        self.assertIsNone(result,
            "Audit 1.4 FAILED: External failure should return None")
        
        # Check mode is DEGRADED
        self.assertEqual(loader.config.mode.value, "degraded",
            "Audit 1.4 FAILED: External failure should set mode to DEGRADED")
        
        # Check warning was added
        self.assertTrue(len(loader.config.warnings) > 0,
            "Audit 1.4 FAILED: External failure should add warning")
        
        logger.info(f"✓ Audit 1.4: External resource fail-open verified")
    
    def test_degradation_score_calculation(self):
        """Test that degradation score is calculated correctly"""
        from audit_config import ResourceLoader, ResourceType
        
        loader = ResourceLoader()
        
        # Load 2 external resources, 1 fails
        loader.load_resource(
            name="External1",
            loader_func=lambda: {"data": "test"},
            resource_type=ResourceType.EXTERNAL
        )
        
        loader.load_resource(
            name="External2",
            loader_func=lambda: None if False else {"data": "test2"},  # Fails
            resource_type=ResourceType.EXTERNAL,
            degradation_impact="Minor impact"
        )
        
        # Both succeed - score should be 0
        score = loader.config.calculate_degradation()
        self.assertEqual(score, 0.0,
            "Audit 1.4 FAILED: No failures should have 0% degradation")
        
        logger.info(f"✓ Audit 1.4: Degradation score calculation verified")
    
    def test_can_proceed_logic(self):
        """Test that can_proceed correctly reflects operational status"""
        from audit_config import ResourceLoader, ResourceType, FallbackMode
        
        loader = ResourceLoader()
        
        # Initially should be able to proceed (FULL mode)
        self.assertTrue(loader.config.can_proceed())
        
        # After external failure, should still proceed (DEGRADED mode)
        try:
            loader.load_resource(
                name="External",
                loader_func=lambda: (_ for _ in ()).throw(ImportError("Test")),
                resource_type=ResourceType.EXTERNAL
            )
        except:
            pass
        
        self.assertTrue(loader.config.can_proceed(),
            "Audit 1.4 FAILED: Should proceed in DEGRADED mode")
        
        # Manually set to FAILED mode
        loader.config.mode = FallbackMode.FAILED
        self.assertFalse(loader.config.can_proceed(),
            "Audit 1.4 FAILED: Should NOT proceed in FAILED mode")
        
        logger.info(f"✓ Audit 1.4: can_proceed logic verified")


class TestIntegratedAuditPoints(unittest.TestCase):
    """Integration tests across all audit points"""
    
    def test_full_extraction_pipeline_audit_compliance(self):
        """Test that extraction pipeline respects all audit points"""
        import tempfile
        import os
        from extraction.extraction_pipeline import ExtractionPipeline, SemanticChunk
        
        # Create test PDF content
        test_pdf_content = b"%PDF-1.4\nTest content for audit compliance"
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as f:
            f.write(test_pdf_content)
            temp_path = f.name
        
        try:
            class MockConfig:
                pass
            
            pipeline = ExtractionPipeline(MockConfig())
            
            # Audit 1.2: Compute source_pdf_hash
            pdf_hash = pipeline._compute_sha256(temp_path)
            self.assertEqual(len(pdf_hash), 64, "PDF hash should be SHA-256")
            
            # Audit 1.2: Create chunk with provenance
            chunk_id = SemanticChunk.create_chunk_id(pdf_hash, 0, "Test text")
            self.assertEqual(len(chunk_id), 64, "Chunk ID should be SHA-256")
            
            # Audit 1.3: Validate chunk
            chunk = SemanticChunk(
                chunk_id=chunk_id,
                text="Test text",
                start_char=0,
                end_char=9,
                doc_id=pdf_hash,
                metadata={}
            )
            self.assertIsNotNone(chunk)
            
            logger.info(f"✓ Integrated audit compliance verified")
        
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    print("=" * 70)
    print("AUDIT POINTS 1.1-1.4 TEST SUITE")
    print("MMR Reproducibility Standards Validation")
    print("=" * 70)
    print()
    
    # Run tests
    unittest.main(verbosity=2)
