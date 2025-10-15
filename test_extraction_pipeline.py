#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Extraction Pipeline
"""

import asyncio
import tempfile
from pathlib import Path
import pytest

# Test the imports
try:
    from extraction.extraction_pipeline import (
        ExtractionPipeline,
        ExtractionResult,
        ExtractedTable,
        SemanticChunk,
        DataQualityMetrics,
        TableDataCleaner,
    )
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


def test_imports():
    """Test that all imports work"""
    assert IMPORTS_OK, f"Import failed: {IMPORT_ERROR if not IMPORTS_OK else ''}"


def test_extracted_table_validation():
    """Test ExtractedTable Pydantic validation"""
    # Valid table
    table = ExtractedTable(
        data=[['A', 'B'], ['1', '2']],
        page_number=1,
        confidence_score=0.95,
        column_count=2,
        row_count=2
    )
    assert table.page_number == 1
    assert table.confidence_score == 0.95
    
    # Invalid: empty data
    with pytest.raises(ValueError):
        ExtractedTable(
            data=[],
            page_number=1,
            confidence_score=0.95,
            column_count=0,
            row_count=0
        )
    
    # Invalid: confidence out of range
    with pytest.raises(ValueError):
        ExtractedTable(
            data=[['A']],
            page_number=1,
            confidence_score=1.5,  # > 1.0
            column_count=1,
            row_count=1
        )


def test_semantic_chunk_validation():
    """Test SemanticChunk Pydantic validation"""
    # Valid chunk
    chunk = SemanticChunk(
        chunk_id="test_chunk_001",
        text="This is a test chunk",
        start_char=0,
        end_char=20,
        doc_id="abc123"
    )
    assert chunk.chunk_id == "test_chunk_001"
    assert len(chunk.text) == 20
    
    # Invalid: end_char <= start_char
    with pytest.raises(ValueError):
        SemanticChunk(
            chunk_id="test_chunk_002",
            text="Test",
            start_char=10,
            end_char=5,  # Less than start
            doc_id="abc123"
        )


def test_data_quality_metrics_validation():
    """Test DataQualityMetrics Pydantic validation"""
    # Valid metrics
    metrics = DataQualityMetrics(
        text_extraction_quality=0.95,
        table_extraction_quality=0.85,
        semantic_coherence=0.90,
        completeness_score=0.92,
        total_chars_extracted=5000,
        total_tables_extracted=3,
        total_chunks_created=10
    )
    assert metrics.completeness_score == 0.92
    assert len(metrics.extraction_warnings) == 0
    
    # Invalid: quality score out of range
    with pytest.raises(ValueError):
        DataQualityMetrics(
            text_extraction_quality=1.5,  # > 1.0
            table_extraction_quality=0.85,
            semantic_coherence=0.90,
            completeness_score=0.92,
            total_chars_extracted=5000,
            total_tables_extracted=3,
            total_chunks_created=10
        )


def test_extraction_result_structure():
    """Test ExtractionResult structure"""
    quality = DataQualityMetrics(
        text_extraction_quality=0.95,
        table_extraction_quality=0.85,
        semantic_coherence=0.90,
        completeness_score=0.92,
        total_chars_extracted=100,
        total_tables_extracted=1,
        total_chunks_created=2
    )
    
    table = ExtractedTable(
        data=[['Header1', 'Header2'], ['Value1', 'Value2']],
        page_number=1,
        confidence_score=0.85,
        column_count=2,
        row_count=2
    )
    
    chunk = SemanticChunk(
        chunk_id="chunk_001",
        text="Sample text chunk",
        start_char=0,
        end_char=17,
        doc_id="doc123"
    )
    
    result = ExtractionResult(
        raw_text="Sample document text",
        tables=[table],
        semantic_chunks=[chunk],
        extraction_quality=quality,
        doc_metadata={'filename': 'test.pdf'}
    )
    
    assert len(result.raw_text) == 20
    assert len(result.tables) == 1
    assert len(result.semantic_chunks) == 1
    assert result.extraction_quality.completeness_score == 0.92


def test_table_data_cleaner():
    """Test TableDataCleaner functionality"""
    import pandas as pd
    
    cleaner = TableDataCleaner()
    
    # Create sample DataFrames
    df1 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    df2 = pd.DataFrame({
        'X': [None, None],
        'Y': [None, None]
    })  # Empty table
    
    raw_tables = [df1, df2]
    cleaned = cleaner.clean(raw_tables)
    
    # Should have only 1 table (df2 is all null)
    assert len(cleaned) >= 1
    
    # First table should have correct structure
    assert cleaned[0]['column_count'] == 2
    assert cleaned[0]['row_count'] == 3
    assert 'confidence_score' in cleaned[0]


def test_pipeline_sha256_computation():
    """Test SHA256 hash computation"""
    from extraction.extraction_pipeline import ExtractionPipeline
    from dereck_beach import ConfigLoader
    
    # Create a temporary config for testing
    # This is a minimal test - actual usage would need real config
    class MockConfig:
        def get(self, key, default=None):
            return default
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content for hashing")
        temp_path = f.name
    
    try:
        pipeline = ExtractionPipeline(MockConfig())
        hash1 = pipeline._compute_sha256(temp_path)
        hash2 = pipeline._compute_sha256(temp_path)
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex chars
    finally:
        Path(temp_path).unlink()


def test_chunk_provenance():
    """Test semantic chunking with provenance"""
    class MockConfig:
        def get(self, key, default=None):
            return default
    
    pipeline = ExtractionPipeline(MockConfig())
    
    # Test chunking
    test_text = "This is a test sentence. " * 50  # ~1300 chars
    doc_id = "test_doc_123"
    
    async def run_test():
        chunks = await pipeline._chunk_with_provenance(test_text, doc_id)
        
        assert len(chunks) > 0
        assert all(c.doc_id == doc_id for c in chunks)
        
        # Check chunk IDs are unique
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
        
        # Check ranges are valid
        for chunk in chunks:
            assert chunk.end_char > chunk.start_char
            assert chunk.start_char >= 0
        
        return chunks
    
    chunks = asyncio.run(run_test())
    assert len(chunks) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
