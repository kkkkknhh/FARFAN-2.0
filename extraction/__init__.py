"""
Extraction module for CDAF Framework.

Provides unified extraction pipeline for Phase I processing with:
- Validated data structures (Pydantic models)
- Async I/O for parallel extraction
- Quality metrics and provenance tracking
"""

from extraction.extraction_pipeline import (
    ExtractionPipeline,
    ExtractionResult,
    ExtractedTable,
    SemanticChunk,
    DataQualityMetrics,
    TableDataCleaner,
)

__all__ = [
    'ExtractionPipeline',
    'ExtractionResult',
    'ExtractedTable',
    'SemanticChunk',
    'DataQualityMetrics',
    'TableDataCleaner',
]
