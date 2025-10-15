#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for IoR (Inference of Relations) Per-Link Explainability Payload.

Tests Audit Point 3.1 (Full Traceability) and Audit Point 3.2 (Credibility Reporting)
per SOTA requirements (Beach & Pedersen 2019, Doshi-Velez 2017, Gelman 2013).
"""

import json
import unittest
from datetime import datetime

# Mock numpy and scipy for testing without dependencies
class MockRandomState:
    def __init__(self, seed):
        self.seed_value = seed
    
    def beta(self, a, b, size):
        return [0.5] * size


class MockRandom:
    @staticmethod
    def seed(s):
        pass
    
    @staticmethod
    def randn(*args):
        return [0.5] * args[0] if args else [0.5]
    
    @staticmethod
    def RandomState(seed):
        return MockRandomState(seed)


class MockNumpy:
    random = MockRandom
    
    @staticmethod
    def array(data):
        return data
    
    @staticmethod
    def zeros(shape):
        if isinstance(shape, int):
            return [0.0] * shape
        return [[0.0] * shape[1] for _ in range(shape[0])]
    
    @staticmethod
    def sort(arr):
        return sorted(arr)
    
    @staticmethod
    def mean(arr):
        return sum(arr) / len(arr) if arr else 0.0
    
    @staticmethod
    def std(arr):
        if not arr:
            return 0.0
        mean_val = sum(arr) / len(arr)
        variance = sum((x - mean_val) ** 2 for x in arr) / len(arr)
        return variance ** 0.5
    
    @staticmethod
    def ceil(x):
        import math
        return math.ceil(x)
    
    @staticmethod
    def argmin(arr):
        return arr.index(min(arr))


# Mock scipy for testing
class MockScipy:
    class stats:
        @staticmethod
        def beta(*args, **kwargs):
            return None
    
    class spatial:
        class distance:
            @staticmethod
            def cosine(a, b):
                return 0.1  # Mock similarity


# Inject mocks before importing
import sys
sys.modules['numpy'] = MockNumpy
sys.modules['scipy'] = MockScipy
sys.modules['scipy.stats'] = MockScipy.stats
sys.modules['scipy.spatial'] = MockScipy.spatial
sys.modules['scipy.spatial.distance'] = MockScipy.spatial.distance

# Now import the real modules
from inference.bayesian_engine import (
    BayesianSamplingEngine,
    CausalLink,
    EvidenceChunk,
    InferenceExplainabilityPayload,
    NecessityTestResult,
    PosteriorDistribution,
    SamplingConfig,
)


class TestEvidenceChunkSHA256(unittest.TestCase):
    """Test SHA256 hashing for evidence chunks (Audit Point 3.1)"""
    
    def test_sha256_auto_computation(self):
        """Test that SHA256 is automatically computed for evidence chunks"""
        chunk = EvidenceChunk(
            chunk_id="chunk_001",
            text="This is evidence text for testing SHA256 computation",
            cosine_similarity=0.85
        )
        
        # SHA256 should be automatically computed
        self.assertIsNotNone(chunk.source_chunk_sha256)
        self.assertEqual(len(chunk.source_chunk_sha256), 64)  # SHA256 is 64 hex chars
        
    def test_sha256_deterministic(self):
        """Test that SHA256 is deterministic for same text"""
        text = "Deterministic test text"
        
        chunk1 = EvidenceChunk(
            chunk_id="chunk_001",
            text=text,
            cosine_similarity=0.85
        )
        
        chunk2 = EvidenceChunk(
            chunk_id="chunk_002",
            text=text,
            cosine_similarity=0.90
        )
        
        # Same text should produce same hash
        self.assertEqual(chunk1.source_chunk_sha256, chunk2.source_chunk_sha256)
    
    def test_sha256_different_for_different_text(self):
        """Test that different text produces different hashes"""
        chunk1 = EvidenceChunk(
            chunk_id="chunk_001",
            text="First evidence text",
            cosine_similarity=0.85
        )
        
        chunk2 = EvidenceChunk(
            chunk_id="chunk_002",
            text="Second evidence text",
            cosine_similarity=0.90
        )
        
        # Different text should produce different hashes
        self.assertNotEqual(chunk1.source_chunk_sha256, chunk2.source_chunk_sha256)


class TestPosteriorDistributionCredibleInterval(unittest.TestCase):
    """Test credible interval property (Audit Point 3.2)"""
    
    def test_credible_interval_95_property(self):
        """Test that credible_interval_95 property works correctly"""
        posterior = PosteriorDistribution(
            posterior_mean=0.75,
            posterior_std=0.10,
            confidence_interval=(0.55, 0.95)
        )
        
        # credible_interval_95 should return the confidence_interval
        ci_95 = posterior.credible_interval_95
        self.assertIsInstance(ci_95, tuple)
        self.assertEqual(len(ci_95), 2)
        self.assertLessEqual(ci_95[0], ci_95[1])


class TestInferenceExplainabilityPayload(unittest.TestCase):
    """Test IoR Explainability Payload (Audit Points 3.1 and 3.2)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.payload = InferenceExplainabilityPayload(
            cause_id="MP-001",
            effect_id="MR-002",
            link_type="producto→resultado",
            posterior_mean=0.75,
            posterior_std=0.10,
            credible_interval_95=(0.60, 0.90),
            convergence_diagnostic=True,
            necessity_passed=True,
            necessity_missing=[],
            evidence_snippets=[
                {
                    "chunk_id": "chunk_001",
                    "text": "Evidence text snippet...",
                    "cosine_similarity": 0.85,
                    "sha256": "abc123"
                }
            ],
            source_chunk_hashes=["abc123", "def456"],
            timestamp="2025-10-15T18:00:00Z"
        )
    
    def test_payload_required_fields(self):
        """Test that all required fields are present (Audit Point 3.1)"""
        # Link identification
        self.assertEqual(self.payload.cause_id, "MP-001")
        self.assertEqual(self.payload.effect_id, "MR-002")
        self.assertEqual(self.payload.link_type, "producto→resultado")
        
        # Bayesian metrics
        self.assertEqual(self.payload.posterior_mean, 0.75)
        self.assertEqual(self.payload.posterior_std, 0.10)
        self.assertEqual(self.payload.credible_interval_95, (0.60, 0.90))
        self.assertTrue(self.payload.convergence_diagnostic)
        
        # Necessity results
        self.assertTrue(self.payload.necessity_passed)
        self.assertEqual(self.payload.necessity_missing, [])
        
        # Evidence and hashes
        self.assertEqual(len(self.payload.evidence_snippets), 1)
        self.assertEqual(len(self.payload.source_chunk_hashes), 2)
    
    def test_payload_to_json_dict(self):
        """Test JSON serialization (Audit Point 3.1: Full Traceability)"""
        json_dict = self.payload.to_json_dict()
        
        # Verify all required fields are present
        required_fields = [
            "cause_id", "effect_id", "link_type",
            "posterior_mean", "posterior_std", "credible_interval_95",
            "convergence_diagnostic", "necessity_test",
            "evidence_snippets", "source_chunk_hashes",
            "quality_score", "timestamp"
        ]
        
        for field in required_fields:
            self.assertIn(field, json_dict, f"Missing required field: {field}")
        
        # Verify JSON serializability
        json_str = json.dumps(json_dict)
        self.assertIsInstance(json_str, str)
        
        # Verify round-trip
        loaded = json.loads(json_str)
        self.assertEqual(loaded["cause_id"], "MP-001")
        self.assertEqual(loaded["effect_id"], "MR-002")
    
    def test_quality_score_computation(self):
        """Test quality score with Bayesian metrics (Audit Point 3.2)"""
        quality = self.payload.compute_quality_score()
        
        # Verify required metrics
        self.assertIn("evidence_strength", quality)
        self.assertIn("epistemic_uncertainty", quality)
        self.assertIn("quality_score", quality)
        self.assertIn("credible_interval_95", quality)
        self.assertIn("credible_interval_width", quality)
        
        # Verify epistemic uncertainty computation
        expected_width = 0.90 - 0.60  # 0.30
        self.assertAlmostEqual(quality["credible_interval_width"], expected_width, places=6)
        
        # Evidence strength should equal posterior mean
        self.assertAlmostEqual(quality["evidence_strength"], 0.75, places=6)
        
        # Epistemic uncertainty should be computed
        self.assertGreater(quality["epistemic_uncertainty"], 0.0)
        self.assertLessEqual(quality["epistemic_uncertainty"], 1.0)
    
    def test_credible_interval_in_quality_score(self):
        """Test that credible_interval_95 is included in quality score (Audit Point 3.2)"""
        quality = self.payload.compute_quality_score()
        
        self.assertIn("credible_interval_95", quality)
        ci = quality["credible_interval_95"]
        self.assertEqual(len(ci), 2)
        self.assertEqual(ci[0], 0.60)
        self.assertEqual(ci[1], 0.90)


class TestBayesianSamplingEngineExplainability(unittest.TestCase):
    """Test BayesianSamplingEngine explainability payload creation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BayesianSamplingEngine(seed=42)
        
        # Create mock causal link
        self.link = CausalLink(
            cause_id="MP-001",
            effect_id="MR-002",
            cause_emb=[0.5] * 384,
            effect_emb=[0.6] * 384,
            cause_type="producto",
            effect_type="resultado"
        )
        
        # Create evidence chunks with SHA256
        self.evidence = [
            EvidenceChunk(
                chunk_id="chunk_001",
                text="Evidence supporting the causal mechanism",
                cosine_similarity=0.85,
                source_page=1
            ),
            EvidenceChunk(
                chunk_id="chunk_002",
                text="Additional evidence with high similarity",
                cosine_similarity=0.90,
                source_page=2
            ),
            EvidenceChunk(
                chunk_id="chunk_003",
                text="More supporting evidence for the link",
                cosine_similarity=0.75,
                source_page=3
            )
        ]
        
        # Create posterior distribution
        self.posterior = PosteriorDistribution(
            posterior_mean=0.80,
            posterior_std=0.08,
            confidence_interval=(0.65, 0.92),
            convergence_diagnostic=True
        )
        
        # Create necessity test result
        self.necessity_result = NecessityTestResult(
            passed=True,
            missing=[],
            severity=None,
            remediation=None
        )
    
    def test_create_explainability_payload(self):
        """Test creation of full explainability payload (Audit Points 3.1 and 3.2)"""
        payload = self.engine.create_explainability_payload(
            link=self.link,
            posterior=self.posterior,
            evidence=self.evidence,
            necessity_result=self.necessity_result,
            timestamp="2025-10-15T18:00:00Z"
        )
        
        # Verify payload structure
        self.assertIsInstance(payload, InferenceExplainabilityPayload)
        
        # Verify link information
        self.assertEqual(payload.cause_id, "MP-001")
        self.assertEqual(payload.effect_id, "MR-002")
        self.assertEqual(payload.link_type, "producto→resultado")
        
        # Verify Bayesian metrics
        self.assertEqual(payload.posterior_mean, 0.80)
        self.assertEqual(payload.posterior_std, 0.08)
        
        # Verify necessity results
        self.assertTrue(payload.necessity_passed)
        
        # Verify evidence snippets (top 5)
        self.assertGreater(len(payload.evidence_snippets), 0)
        self.assertLessEqual(len(payload.evidence_snippets), 5)
        
        # Verify SHA256 hashes are present
        self.assertGreater(len(payload.source_chunk_hashes), 0)
        for hash_val in payload.source_chunk_hashes:
            self.assertEqual(len(hash_val), 64)  # SHA256 is 64 hex chars
    
    def test_evidence_snippets_sorted_by_similarity(self):
        """Test that evidence snippets are sorted by similarity (highest first)"""
        payload = self.engine.create_explainability_payload(
            link=self.link,
            posterior=self.posterior,
            evidence=self.evidence
        )
        
        # Verify snippets are sorted by cosine_similarity (descending)
        similarities = [s["cosine_similarity"] for s in payload.evidence_snippets]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
        
        # First snippet should be the one with highest similarity (0.90)
        self.assertEqual(payload.evidence_snippets[0]["cosine_similarity"], 0.90)
    
    def test_evidence_snippet_contains_sha256(self):
        """Test that each evidence snippet contains SHA256 hash (Audit Point 3.1)"""
        payload = self.engine.create_explainability_payload(
            link=self.link,
            posterior=self.posterior,
            evidence=self.evidence
        )
        
        for snippet in payload.evidence_snippets:
            self.assertIn("sha256", snippet)
            self.assertIsNotNone(snippet["sha256"])
            self.assertEqual(len(snippet["sha256"]), 64)
    
    def test_payload_json_serialization(self):
        """Test that payload can be serialized to JSON (Audit Point 3.1)"""
        payload = self.engine.create_explainability_payload(
            link=self.link,
            posterior=self.posterior,
            evidence=self.evidence
        )
        
        json_dict = payload.to_json_dict()
        
        # Verify JSON serializability
        try:
            json_str = json.dumps(json_dict)
            self.assertIsInstance(json_str, str)
        except (TypeError, ValueError) as e:
            self.fail(f"Payload should be JSON serializable: {e}")
        
        # Verify all critical fields are present
        loaded = json.loads(json_str)
        self.assertIn("posterior_mean", loaded)
        self.assertIn("credible_interval_95", loaded)
        self.assertIn("source_chunk_hashes", loaded)
        self.assertIn("evidence_snippets", loaded)
        self.assertIn("quality_score", loaded)
    
    def test_default_necessity_result(self):
        """Test that default necessity result is created if not provided"""
        payload = self.engine.create_explainability_payload(
            link=self.link,
            posterior=self.posterior,
            evidence=self.evidence,
            necessity_result=None  # No necessity result provided
        )
        
        # Should have default values
        self.assertTrue(payload.necessity_passed)
        self.assertEqual(payload.necessity_missing, [])
        self.assertIsNone(payload.necessity_severity)


class TestSOTACompliance(unittest.TestCase):
    """Test compliance with SOTA requirements"""
    
    def test_audit_point_3_1_full_traceability(self):
        """
        Audit Point 3.1: Full Traceability Payload
        
        Check Criteria: Every link generates JSON payload with posterior, 
        necessity result, snippets, sha256.
        
        Quality Evidence: Sample link outputs; validate all fields present/matching 
        source hashes.
        """
        engine = BayesianSamplingEngine(seed=42)
        
        link = CausalLink(
            cause_id="MP-TEST",
            effect_id="MR-TEST",
            cause_emb=[0.5] * 384,
            effect_emb=[0.6] * 384,
            cause_type="producto",
            effect_type="resultado"
        )
        
        evidence = [
            EvidenceChunk(
                chunk_id=f"chunk_{i:03d}",
                text=f"Evidence chunk {i} for testing",
                cosine_similarity=0.8 + (i * 0.01)
            )
            for i in range(3)
        ]
        
        posterior = PosteriorDistribution(
            posterior_mean=0.85,
            posterior_std=0.07,
            confidence_interval=(0.72, 0.95),
            convergence_diagnostic=True
        )
        
        necessity = NecessityTestResult(
            passed=True,
            missing=[],
            severity=None
        )
        
        # Create payload
        payload = engine.create_explainability_payload(
            link=link,
            posterior=posterior,
            evidence=evidence,
            necessity_result=necessity
        )
        
        # Convert to JSON
        json_dict = payload.to_json_dict()
        
        # VERIFY: All required fields present
        required_fields = {
            "posterior_mean", "credible_interval_95", 
            "source_chunk_hashes", "evidence_snippets",
            "necessity_test"
        }
        
        for field in required_fields:
            self.assertIn(field, json_dict, 
                         f"Audit Point 3.1 FAILED: Missing {field}")
        
        # VERIFY: SHA256 hashes are valid
        for hash_val in json_dict["source_chunk_hashes"]:
            self.assertEqual(len(hash_val), 64,
                           "Audit Point 3.1 FAILED: Invalid SHA256 hash length")
        
        # VERIFY: Evidence snippets contain source hashes
        for snippet in json_dict["evidence_snippets"]:
            self.assertIn("sha256", snippet,
                         "Audit Point 3.1 FAILED: Evidence snippet missing SHA256")
    
    def test_audit_point_3_2_credibility_reporting(self):
        """
        Audit Point 3.2: Credibility Reporting
        
        Check Criteria: QualityScore includes credible_interval_95 and Bayesian 
        metrics, reflecting epistemic uncertainty.
        
        Quality Evidence: Review final scores; confirm intervals (e.g., 95% credible) 
        in evaluation logs.
        """
        payload = InferenceExplainabilityPayload(
            cause_id="MP-TEST",
            effect_id="MR-TEST",
            link_type="producto→resultado",
            posterior_mean=0.70,
            posterior_std=0.12,
            credible_interval_95=(0.50, 0.88),
            convergence_diagnostic=True,
            necessity_passed=True,
            necessity_missing=[]
        )
        
        # Compute quality score
        quality = payload.compute_quality_score()
        
        # VERIFY: Credible interval is present
        self.assertIn("credible_interval_95", quality,
                     "Audit Point 3.2 FAILED: Missing credible_interval_95")
        
        # VERIFY: Epistemic uncertainty is reported
        self.assertIn("epistemic_uncertainty", quality,
                     "Audit Point 3.2 FAILED: Missing epistemic_uncertainty")
        
        # VERIFY: Credible interval width is computed
        self.assertIn("credible_interval_width", quality,
                     "Audit Point 3.2 FAILED: Missing credible_interval_width")
        
        expected_width = 0.88 - 0.50
        self.assertAlmostEqual(
            quality["credible_interval_width"], 
            expected_width,
            places=6,
            msg="Audit Point 3.2 FAILED: Incorrect interval width"
        )
        
        # VERIFY: Uncertainty-aware quality score
        # Quality should decrease with higher uncertainty
        self.assertGreater(quality["evidence_strength"], 0,
                          "Audit Point 3.2 FAILED: Evidence strength not computed")
        self.assertLess(quality["quality_score"], quality["evidence_strength"],
                       "Audit Point 3.2 FAILED: Quality score should reflect uncertainty")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEvidenceChunkSHA256))
    suite.addTests(loader.loadTestsFromTestCase(TestPosteriorDistributionCredibleInterval))
    suite.addTests(loader.loadTestsFromTestCase(TestInferenceExplainabilityPayload))
    suite.addTests(loader.loadTestsFromTestCase(TestBayesianSamplingEngineExplainability))
    suite.addTests(loader.loadTestsFromTestCase(TestSOTACompliance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
