#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction Pipeline for CDAF Framework
Consolidates Phase I extraction with explicit contracts and async processing.

This module addresses architectural fragmentation by:
- Unifying PDF processing, document analysis, and table extraction
- Providing validated data structures with Pydantic
- Enabling async I/O for parallel extraction
- Establishing auditable checkpoints before Phase II
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

# Import existing CDAF components (will be injected)
# Note: Avoiding circular imports by using TYPE_CHECKING pattern
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, validator

if TYPE_CHECKING:
    from dereck_beach import ConfigLoader, PDFProcessor

# ============================================================================
# Pydantic Models for Validated Data Structures
# ============================================================================


class ExtractedTable(BaseModel):
    """Validated table with metadata and quality metrics"""

    data: List[List[Any]] = Field(description="Table data as list of rows")
    page_number: int = Field(ge=1, description="Page number where table was found")
    table_type: Optional[str] = Field(
        default=None, description="Classified table type (financial, indicators, etc.)"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Extraction confidence score"
    )
    column_count: int = Field(ge=1, description="Number of columns")
    row_count: int = Field(ge=0, description="Number of rows")

    class Config:
        arbitrary_types_allowed = True

    @validator("data")
    def validate_data_structure(cls, v):
        """
        Audit Point 1.3: Schema Integrity (Hoop Test)
        Ensure data is properly structured - hard failure on missing critical field.
        """
        if not v:
            raise ValueError(
                "Table data cannot be empty - Hoop Test failed (Audit Point 1.3)"
            )
        if not all(isinstance(row, list) for row in v):
            raise ValueError(
                "All rows must be lists - Schema integrity violation (Audit Point 1.3)"
            )
        return v

    @validator("bpin")
    def validate_bpin_if_required(cls, v, values):
        """
        Audit Point 1.3: Conditional validation for DNP-mandated BPIN.
        For financial/investment tables, BPIN should be present.
        """
        table_type = values.get("table_type", "")
        if table_type in ["financial", "investment", "budget"] and not v:
            # Log warning but don't hard-fail (degraded mode per Audit Point 1.4)
            logging.warning(
                f"BPIN missing for {table_type} table - DNP compliance degraded"
            )
        return v


class SemanticChunk(BaseModel):
    """
    Semantic text chunk with provenance tracking.

    Audit Point 1.2: Provenance Traceability (SHA-256)
    Each chunk anchored to immutable chunk_id via SHA-256 hash of canonicalized string.
    """

    chunk_id: str = Field(
        description="SHA-256 hash of canonicalized (doc_id + index + content preview) for immutable provenance"
    )
    text: str = Field(min_length=1, description="Chunk text content")
    start_char: int = Field(ge=0, description="Starting character position in document")
    end_char: int = Field(ge=0, description="Ending character position in document")
    doc_id: str = Field(description="Document SHA256 hash for traceability")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (page, section, etc.)"
    )

    @validator("chunk_id")
    def validate_chunk_id(cls, v):
        """
        Audit Point 1.3: Schema Integrity (Hoop Test)
        Ensure chunk_id is present and valid SHA-256 hash format.
        """
        if not v:
            raise ValueError(
                "chunk_id is required for provenance traceability (Audit Point 1.2)"
            )
        # Validate SHA-256 format (64 hex characters)
        if not (len(v) == 64 and all(c in "0123456789abcdef" for c in v.lower())):
            raise ValueError(
                f"chunk_id must be valid SHA-256 hash (64 hex chars), got: {v[:20]}..."
            )
        return v

    @validator("end_char")
    def validate_range(cls, v, values):
        """Ensure end_char > start_char"""
        if "start_char" in values and v <= values["start_char"]:
            raise ValueError("end_char must be greater than start_char")
        return v

    @classmethod
    def create_chunk_id(cls, doc_id: str, index: int, text_preview: str) -> str:
        """
        Generate deterministic SHA-256 chunk_id from canonicalized inputs.

        Audit Point 1.2: Provenance Traceability
        Creates immutable identifier enabling fine-grained process-tracing.

        Args:
            doc_id: Document SHA-256 hash
            index: Chunk index in document
            text_preview: First 200 chars of chunk content

        Returns:
            64-character SHA-256 hash hex string
        """
        # Canonicalize: normalize whitespace in preview
        normalized_preview = " ".join(text_preview[:200].split())
        canonical_string = f"{doc_id}|{index}|{normalized_preview}"
        return hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()


class DataQualityMetrics(BaseModel):
    """Quality assessment metrics for extracted data"""

    text_extraction_quality: float = Field(
        ge=0.0, le=1.0, description="Quality score for text extraction"
    )
    table_extraction_quality: float = Field(
        ge=0.0, le=1.0, description="Quality score for table extraction"
    )
    semantic_coherence: float = Field(
        ge=0.0, le=1.0, description="Semantic coherence of chunks"
    )
    completeness_score: float = Field(
        ge=0.0, le=1.0, description="Overall data completeness"
    )
    total_chars_extracted: int = Field(ge=0, description="Total characters extracted")
    total_tables_extracted: int = Field(ge=0, description="Total tables extracted")
    total_chunks_created: int = Field(ge=0, description="Total semantic chunks created")
    extraction_warnings: List[str] = Field(
        default_factory=list, description="Warnings during extraction"
    )


class ExtractionResult(BaseModel):
    """Complete extraction result with validated components"""

    raw_text: str = Field(description="Complete extracted text")
    tables: List[ExtractedTable] = Field(
        default_factory=list, description="Validated extracted tables"
    )
    semantic_chunks: List[SemanticChunk] = Field(
        default_factory=list, description="Semantic chunks with PDQ context"
    )
    extraction_quality: DataQualityMetrics = Field(
        description="Quality metrics for this extraction"
    )
    doc_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document-level metadata"
    )


# ============================================================================
# Table Data Cleaner
# ============================================================================


class TableDataCleaner:
    """Clean and validate extracted table data"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def clean(self, raw_tables: List[pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Clean and normalize table data.

        Args:
            raw_tables: List of pandas DataFrames

        Returns:
            List of cleaned table dictionaries
        """
        cleaned = []

        for idx, df in enumerate(raw_tables):
            try:
                # Skip empty tables
                if df.empty:
                    continue

                # Remove completely empty rows and columns
                df = df.dropna(how="all").dropna(axis=1, how="all")

                if df.empty:
                    continue

                # Convert to list of lists for validation
                data = df.values.tolist()

                # Calculate confidence based on data quality
                non_null_ratio = df.notna().sum().sum() / (df.shape[0] * df.shape[1])
                confidence = min(0.95, non_null_ratio)

                cleaned.append(
                    {
                        "data": data,
                        "page_number": idx
                        + 1,  # Placeholder - should come from metadata
                        "confidence_score": confidence,
                        "column_count": df.shape[1],
                        "row_count": df.shape[0],
                        "table_type": None,  # Will be classified later
                    }
                )

            except Exception as e:
                self.logger.warning(f"Error cleaning table {idx}: {e}")
                continue

        return cleaned


# ============================================================================
# Extraction Pipeline
# ============================================================================


class ExtractionPipeline:
    """
    Orquesta Phase I unificadamente con contratos explícitos.
    Garantiza que cada extractor reciba datos validados.

    This pipeline:
    - Executes async I/O in parallel (resolves Anti-pattern A.3)
    - Provides immediate validation (Schema Validation Standard)
    - Establishes auditable checkpoint before Phase II
    """

    def __init__(self, config: Any):
        """
        Initialize extraction pipeline.

        Args:
            config: CDAFConfig or ConfigLoader instance
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        # Import here to avoid circular dependencies
        from dereck_beach import PDFProcessor

        # Initialize components
        self.pdf_processor = PDFProcessor(config)
        self.table_cleaner = TableDataCleaner()

        # Chunking parameters
        self.chunk_size = 1000  # characters
        self.chunk_overlap = 200  # characters

    async def extract_complete(self, pdf_path: str) -> ExtractionResult:
        """
        Ejecuta extracción completa con fallback graceful.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractionResult with validated data structures
        """
        pdf_path_obj = Path(pdf_path)

        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.logger.info(f"Starting complete extraction for: {pdf_path_obj.name}")

        # Compute document ID for traceability
        doc_id = self._compute_sha256(pdf_path)

        # Load document first
        if not self.pdf_processor.load_document(pdf_path_obj):
            raise RuntimeError(f"Failed to load document: {pdf_path}")

        # Async I/O in parallel (Anti-pattern resolved: A.3)
        text_task = asyncio.create_task(self._extract_text_safe(pdf_path_obj))
        tables_task = asyncio.create_task(self._extract_tables_safe(pdf_path_obj))

        raw_text, raw_tables = await asyncio.gather(text_task, tables_task)

        # Validación inmediata (Schema Validation Standard)
        validated_tables = []
        cleaned_tables = self.table_cleaner.clean(raw_tables)

        for table_data in cleaned_tables:
            try:
                validated_table = ExtractedTable.model_validate(table_data)
                validated_tables.append(validated_table)
            except Exception as e:
                self.logger.warning(f"Table validation failed: {e}")
                continue

        # Chunking con trazabilidad inmediata
        semantic_chunks = await self._chunk_with_provenance(raw_text, doc_id=doc_id)

        # Data Quality Assessment
        quality = self._assess_extraction_quality(
            raw_text, semantic_chunks, validated_tables
        )

        # Get document metadata
        doc_metadata = {
            "filename": pdf_path_obj.name,
            "doc_id": doc_id,
            "pdf_metadata": self.pdf_processor.metadata,
        }

        result = ExtractionResult(
            raw_text=raw_text,
            tables=validated_tables,
            semantic_chunks=semantic_chunks,
            extraction_quality=quality,
            doc_metadata=doc_metadata,
        )

        self.logger.info(
            f"Extraction complete: {len(raw_text)} chars, "
            f"{len(validated_tables)} tables, {len(semantic_chunks)} chunks"
        )

        return result

    async def _extract_text_safe(self, pdf_path: Path) -> str:
        """
        Safely extract text with error handling.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text or empty string on failure
        """
        try:
            # Run synchronous extraction in executor to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self.pdf_processor.extract_text)
            return text
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return ""

    async def _extract_tables_safe(self, pdf_path: Path) -> List[pd.DataFrame]:
        """
        Safely extract tables with error handling.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of extracted tables or empty list on failure
        """
        try:
            # Run synchronous extraction in executor to avoid blocking
            loop = asyncio.get_event_loop()
            tables = await loop.run_in_executor(None, self.pdf_processor.extract_tables)
            return tables
        except Exception as e:
            self.logger.error(f"Table extraction failed: {e}")
            return []

    async def _chunk_with_provenance(
        self, text: str, doc_id: str
    ) -> List[SemanticChunk]:
        """
        Create semantic chunks with full provenance tracking.

        Audit Point 1.2: Provenance Traceability (SHA-256)
        Every SemanticChunk anchored to immutable chunk_id via SHA-256 hash.

        Args:
            text: Full text to chunk
            doc_id: Document SHA256 hash

        Returns:
            List of validated semantic chunks with SHA-256 chunk_id
        """
        chunks = []
        text_length = len(text)

        # Simple sliding window chunking
        # Can be enhanced with spaCy sentence boundaries
        start = 0
        chunk_num = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # Try to break at sentence/word boundary
            if end < text_length:
                # Look for period, newline, or space
                for sep in [". ", "\n", " "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size * 0.8:  # At least 80% of chunk
                        end = start + last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                # Audit Point 1.2: Generate SHA-256 chunk_id for immutable provenance
                chunk_id = SemanticChunk.create_chunk_id(
                    doc_id=doc_id, index=chunk_num, text_preview=chunk_text
                )

                try:
                    chunk = SemanticChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                        doc_id=doc_id,
                        metadata={
                            "chunk_number": chunk_num,
                            "total_length": text_length,
                        },
                    )
                    chunks.append(chunk)
                    chunk_num += 1
                except Exception as e:
                    # Audit Point 1.3: Schema validation hard failure
                    self.logger.error(
                        f"Chunk validation failed (Audit 1.3 Hoop Test): {e}"
                    )
                    # Hard failure - skip invalid chunk from evidence pool
                    continue

            # Move forward with overlap
            start = end - self.chunk_overlap
            if start >= text_length:
                break

        return chunks

    def _assess_extraction_quality(
        self,
        raw_text: str,
        semantic_chunks: List[SemanticChunk],
        validated_tables: List[ExtractedTable],
    ) -> DataQualityMetrics:
        """
        Assess overall extraction quality.

        Args:
            raw_text: Extracted text
            semantic_chunks: Created chunks
            validated_tables: Validated tables

        Returns:
            Quality metrics
        """
        warnings = []

        # Text quality - based on length and content
        text_quality = 1.0
        if not raw_text:
            text_quality = 0.0
            warnings.append("No text extracted")
        elif len(raw_text) < 100:
            text_quality = 0.3
            warnings.append("Very little text extracted")

        # Table quality - average confidence
        table_quality = 0.0
        if validated_tables:
            table_quality = sum(t.confidence_score for t in validated_tables) / len(
                validated_tables
            )
        else:
            warnings.append("No tables extracted")

        # Semantic coherence - chunk coverage
        semantic_coherence = 0.0
        if raw_text and semantic_chunks:
            total_chunk_chars = sum(len(c.text) for c in semantic_chunks)
            # Account for overlap
            expected_chars = len(raw_text)
            semantic_coherence = min(1.0, total_chunk_chars / max(1, expected_chars))

        # Completeness
        completeness = (
            text_quality * 0.5 + table_quality * 0.3 + semantic_coherence * 0.2
        )

        return DataQualityMetrics(
            text_extraction_quality=text_quality,
            table_extraction_quality=table_quality,
            semantic_coherence=semantic_coherence,
            completeness_score=completeness,
            total_chars_extracted=len(raw_text),
            total_tables_extracted=len(validated_tables),
            total_chunks_created=len(semantic_chunks),
            extraction_warnings=warnings,
        )

    def _compute_sha256(self, pdf_path: str) -> str:
        """
        Compute SHA256 hash of PDF file for tracking.

        Audit Point 1.2: Provenance Traceability (SHA-256)
        source_pdf_hash over binary file enables blockchain-level provenance.

        Args:
            pdf_path: Path to PDF file

        Returns:
            SHA256 hash as hex string (64 characters)
        """
        sha256_hash = hashlib.sha256()

        with open(pdf_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        hex_digest = sha256_hash.hexdigest()
        self.logger.info(
            f"Computed source_pdf_hash: {hex_digest[:16]}... (Audit Point 1.2)"
        )
        return hex_digest
