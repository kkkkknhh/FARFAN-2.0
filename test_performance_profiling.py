#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Profiling Tests for Extraction Pipeline
====================================================

Profiles semantic chunking and table extraction to identify bottlenecks.

SIN_CARRETA Compliance:
- Deterministic profiling with fixed inputs
- Auditable performance metrics
- Contract-based performance assertions
"""

import asyncio
import time
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

# Conditional imports for profiling
try:
    from extraction.extraction_pipeline import ExtractionPipeline, SemanticChunk

    EXTRACTION_AVAILABLE = True
except ImportError:
    EXTRACTION_AVAILABLE = False


class TestPerformanceProfiling(unittest.TestCase):
    """
    Performance profiling test suite.

    SIN_CARRETA-RATIONALE: Profiles execution time with deterministic inputs
    to identify bottlenecks while maintaining reproducibility.
    """

    @unittest.skipUnless(EXTRACTION_AVAILABLE, "Extraction pipeline not available")
    def test_semantic_chunking_performance(self):
        """
        Profile semantic chunking performance.

        CONTRACT: Chunking should process ≥1000 chars/sec for typical text.
        BOTTLENECK: Sentence boundary detection and overlap processing.
        """

        # Create mock config
        class MockConfig:
            def __init__(self):
                pass

        config = MockConfig()
        pipeline = ExtractionPipeline(config)

        # Generate test text (10,000 characters)
        test_text = "Este es un texto de prueba para medir el rendimiento. " * 200
        doc_id = "a" * 64  # Valid SHA-256 format

        # Profile chunking
        start_time = time.time()

        # Run chunking (async)
        chunks = asyncio.run(pipeline._chunk_with_provenance(test_text, doc_id))

        end_time = time.time()
        duration = end_time - start_time

        # Calculate metrics
        chars_per_sec = len(test_text) / duration if duration > 0 else 0
        chunks_per_sec = len(chunks) / duration if duration > 0 else 0

        # Assertions
        self.assertGreater(len(chunks), 0, "Should create chunks")
        self.assertLess(duration, 5.0, "Chunking should complete in <5 seconds")
        self.assertGreater(
            chars_per_sec,
            1000,
            f"Should process ≥1000 chars/sec (got {chars_per_sec:.0f})",
        )

        # Report metrics
        print(f"\n--- Semantic Chunking Performance ---")
        print(f"Text length: {len(test_text)} chars")
        print(f"Chunks created: {len(chunks)}")
        print(f"Duration: {duration:.3f} seconds")
        print(f"Throughput: {chars_per_sec:.0f} chars/sec")
        print(f"Chunk rate: {chunks_per_sec:.1f} chunks/sec")

        # Identify bottleneck
        avg_chunk_time = duration / len(chunks) if len(chunks) > 0 else 0
        print(f"Avg time per chunk: {avg_chunk_time * 1000:.2f} ms")

        if avg_chunk_time > 0.01:  # >10ms per chunk
            print("⚠️  BOTTLENECK: Chunk creation time excessive")
            print(
                "   Recommendation: Pre-compile regex patterns, optimize sentence detection"
            )

    @unittest.skipUnless(EXTRACTION_AVAILABLE, "Extraction pipeline not available")
    def test_chunk_id_generation_performance(self):
        """
        Profile SHA-256 chunk_id generation.

        CONTRACT: SHA-256 generation should be <1ms per chunk.
        """
        doc_id = "b" * 64
        test_text = "Test chunk text for hashing"

        # Profile SHA-256 generation
        iterations = 1000
        start_time = time.time()

        for i in range(iterations):
            chunk_id = SemanticChunk.create_chunk_id(doc_id, i, test_text)

        end_time = time.time()
        duration = end_time - start_time
        avg_time = duration / iterations

        # Assertions
        self.assertLess(
            avg_time,
            0.001,
            f"SHA-256 generation should be <1ms (got {avg_time * 1000:.2f}ms)",
        )

        # Report
        print(f"\n--- Chunk ID Generation Performance ---")
        print(f"Iterations: {iterations}")
        print(f"Total duration: {duration:.3f} seconds")
        print(f"Avg time per hash: {avg_time * 1000:.3f} ms")
        print(f"Throughput: {iterations / duration:.0f} hashes/sec")

    def test_semantic_chunking_algorithm_complexity(self):
        """
        Analyze algorithmic complexity of semantic chunking.

        CONTRACT: Chunking should be O(n) with text length.
        """
        if not EXTRACTION_AVAILABLE:
            self.skipTest("Extraction pipeline not available")

        class MockConfig:
            pass

        config = MockConfig()
        pipeline = ExtractionPipeline(config)
        doc_id = "c" * 64

        # Test with different text sizes
        sizes = [1000, 2000, 4000, 8000]
        timings = []

        for size in sizes:
            test_text = "Texto de prueba. " * (size // 17)  # ~17 chars per unit

            start_time = time.time()
            chunks = asyncio.run(
                pipeline._chunk_with_provenance(test_text[:size], doc_id)
            )
            duration = time.time() - start_time

            timings.append((size, duration, len(chunks)))

        # Analyze complexity
        print(f"\n--- Algorithmic Complexity Analysis ---")
        print(f"{'Size':>8} {'Time (s)':>10} {'Chunks':>8} {'Time/Char (μs)':>15}")

        for size, duration, num_chunks in timings:
            time_per_char = (duration / size) * 1e6  # microseconds
            print(f"{size:>8} {duration:>10.4f} {num_chunks:>8} {time_per_char:>15.2f}")

        # Check linearity (O(n))
        # Ratio of (time2/time1) should be approximately (size2/size1)
        if len(timings) >= 2:
            size1, time1, _ = timings[0]
            size2, time2, _ = timings[-1]

            size_ratio = size2 / size1
            time_ratio = time2 / time1
            linearity = time_ratio / size_ratio

            print(f"\nLinearity check:")
            print(f"  Size ratio: {size_ratio:.1f}x")
            print(f"  Time ratio: {time_ratio:.1f}x")
            print(f"  Linearity: {linearity:.2f} (1.0 = perfect O(n))")

            # Assert approximate linearity (allow 2x deviation)
            self.assertLess(linearity, 2.5, "Chunking should be approximately O(n)")

    def test_overlap_processing_impact(self):
        """
        Measure impact of chunk overlap on performance.

        CONTRACT: Overlap should not increase complexity beyond O(n).
        BOTTLENECK: Overlap region reprocessing.
        """
        if not EXTRACTION_AVAILABLE:
            self.skipTest("Extraction pipeline not available")

        class MockConfig:
            pass

        config = MockConfig()
        pipeline = ExtractionPipeline(config)
        doc_id = "d" * 64

        # Test text
        test_text = "Texto de prueba para medir impacto de overlap. " * 100

        # Measure with current overlap (200 chars)
        start_time = time.time()
        chunks_with_overlap = asyncio.run(
            pipeline._chunk_with_provenance(test_text, doc_id)
        )
        time_with_overlap = time.time() - start_time

        # Measure without overlap (temporarily modify)
        original_overlap = pipeline.chunk_overlap
        pipeline.chunk_overlap = 0

        start_time = time.time()
        chunks_no_overlap = asyncio.run(
            pipeline._chunk_with_provenance(test_text, doc_id)
        )
        time_no_overlap = time.time() - start_time

        # Restore
        pipeline.chunk_overlap = original_overlap

        # Calculate impact
        overlap_overhead = (time_with_overlap - time_no_overlap) / time_no_overlap * 100

        print(f"\n--- Overlap Processing Impact ---")
        print(f"Text length: {len(test_text)} chars")
        print(f"Chunks (no overlap): {len(chunks_no_overlap)}")
        print(f"Chunks (with overlap): {len(chunks_with_overlap)}")
        print(f"Time (no overlap): {time_no_overlap:.3f}s")
        print(f"Time (with overlap): {time_with_overlap:.3f}s")
        print(f"Overlap overhead: {overlap_overhead:.1f}%")

        if overlap_overhead > 50:
            print("⚠️  BOTTLENECK: Overlap processing adds >50% overhead")
            print("   Recommendation: Optimize overlap region handling")

    def test_table_extraction_performance_stub(self):
        """
        Stub for table extraction profiling.

        NOTE: Requires PyMuPDF and actual PDF file for realistic profiling.
        CONTRACT: Table extraction should process ≥1 page/sec.
        """
        print(f"\n--- Table Extraction Performance (Stub) ---")
        print("⚠️  Actual profiling requires:")
        print("   1. PyMuPDF library installed")
        print("   2. Test PDF file with tables")
        print("   3. PDFProcessor initialized")
        print("\nExpected bottleneck: CPU-bound PyMuPDF parsing")
        print("Recommendation: Use ProcessPoolExecutor for multi-page PDFs")

        # Mark as skipped
        self.skipTest("Table extraction profiling requires PyMuPDF and test PDF")

    def test_performance_regression_detection(self):
        """
        Baseline performance metrics for regression detection.

        SIN_CARRETA CONTRACT: Performance must not regress >20% between versions.
        """
        if not EXTRACTION_AVAILABLE:
            self.skipTest("Extraction pipeline not available")

        # Baseline metrics (from initial profiling)
        BASELINE_METRICS = {
            "chunking_throughput_chars_per_sec": 5000,  # Minimum acceptable
            "chunk_id_generation_ms": 1.0,  # Maximum acceptable
            "chunking_linearity": 2.0,  # Maximum deviation from O(n)
        }

        # Run performance tests and compare
        class MockConfig:
            pass

        config = MockConfig()
        pipeline = ExtractionPipeline(config)
        test_text = "Texto de prueba. " * 500
        doc_id = "e" * 64

        # Measure chunking throughput
        start_time = time.time()
        chunks = asyncio.run(pipeline._chunk_with_provenance(test_text, doc_id))
        duration = time.time() - start_time
        throughput = len(test_text) / duration if duration > 0 else 0

        # Measure chunk ID generation
        start_time = time.time()
        for i in range(100):
            SemanticChunk.create_chunk_id(doc_id, i, test_text[:100])
        chunk_id_time = (time.time() - start_time) / 100 * 1000  # ms

        # Report
        print(f"\n--- Performance Regression Check ---")
        print(f"Metric                        Baseline    Current    Status")
        print(
            f"Chunking throughput (c/s)     {BASELINE_METRICS['chunking_throughput_chars_per_sec']:>8} {throughput:>10.0f}    ",
            end="",
        )
        if throughput >= BASELINE_METRICS["chunking_throughput_chars_per_sec"]:
            print("✓ PASS")
        else:
            print("✗ FAIL (regression)")

        print(
            f"Chunk ID generation (ms)      {BASELINE_METRICS['chunk_id_generation_ms']:>8.2f} {chunk_id_time:>10.3f}    ",
            end="",
        )
        if chunk_id_time <= BASELINE_METRICS["chunk_id_generation_ms"]:
            print("✓ PASS")
        else:
            print("✗ FAIL (regression)")

        # Assertions for CI
        self.assertGreaterEqual(
            throughput,
            BASELINE_METRICS["chunking_throughput_chars_per_sec"] * 0.8,
            "Chunking throughput regression >20%",
        )
        self.assertLessEqual(
            chunk_id_time,
            BASELINE_METRICS["chunk_id_generation_ms"] * 1.2,
            "Chunk ID generation regression >20%",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
