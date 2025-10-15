#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the Extraction Pipeline

This example demonstrates how to use the new consolidated extraction pipeline
to extract data from a PDF with proper validation and quality metrics.
"""

import asyncio
from pathlib import Path

# Import extraction pipeline components
from extraction import (
    ExtractionPipeline,
    ExtractionResult,
)

# Import CDAF config
from dereck_beach import ConfigLoader


async def extract_document_example(pdf_path: str, config_path: str):
    """
    Example of using the extraction pipeline.
    
    Args:
        pdf_path: Path to the PDF to extract
        config_path: Path to CDAF configuration file
    """
    print("=" * 60)
    print("EXTRACTION PIPELINE EXAMPLE")
    print("=" * 60)
    
    # Load configuration
    print(f"\n1. Loading configuration from: {config_path}")
    config = ConfigLoader(Path(config_path))
    
    # Create extraction pipeline
    print(f"\n2. Initializing extraction pipeline...")
    pipeline = ExtractionPipeline(config)
    
    # Perform extraction
    print(f"\n3. Extracting from PDF: {pdf_path}")
    print("   This runs async I/O in parallel for text and tables...")
    
    result: ExtractionResult = await pipeline.extract_complete(pdf_path)
    
    # Display results
    print(f"\n4. Extraction Results:")
    print(f"   Document ID: {result.doc_metadata.get('doc_id', 'N/A')[:16]}...")
    print(f"   Raw text: {len(result.raw_text)} characters")
    print(f"   Tables extracted: {len(result.tables)}")
    print(f"   Semantic chunks: {len(result.semantic_chunks)}")
    
    print(f"\n5. Quality Metrics:")
    quality = result.extraction_quality
    print(f"   Text quality: {quality.text_extraction_quality:.2%}")
    print(f"   Table quality: {quality.table_extraction_quality:.2%}")
    print(f"   Semantic coherence: {quality.semantic_coherence:.2%}")
    print(f"   Completeness: {quality.completeness_score:.2%}")
    
    if quality.extraction_warnings:
        print(f"\n   Warnings:")
        for warning in quality.extraction_warnings:
            print(f"   ⚠ {warning}")
    
    # Display sample chunks
    if result.semantic_chunks:
        print(f"\n6. Sample Semantic Chunks:")
        for i, chunk in enumerate(result.semantic_chunks[:3]):
            print(f"\n   Chunk {i+1} (ID: {chunk.chunk_id}):")
            print(f"   Position: {chunk.start_char}-{chunk.end_char}")
            preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            print(f"   Text: {preview}")
    
    # Display sample tables
    if result.tables:
        print(f"\n7. Sample Tables:")
        for i, table in enumerate(result.tables[:2]):
            print(f"\n   Table {i+1}:")
            print(f"   Page: {table.page_number}")
            print(f"   Dimensions: {table.row_count} rows x {table.column_count} cols")
            print(f"   Confidence: {table.confidence_score:.2%}")
            if table.table_type:
                print(f"   Type: {table.table_type}")
    
    print("\n" + "=" * 60)
    print("✓ EXTRACTION COMPLETE")
    print("=" * 60)
    
    return result


def main():
    """Main entry point for example"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python example_extraction_pipeline.py <pdf_path> <config_path>")
        print("\nExample:")
        print("  python example_extraction_pipeline.py plan.pdf cuestionario_canonico/config.yaml")
        return
    
    pdf_path = sys.argv[1]
    config_path = sys.argv[2]
    
    # Check files exist
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        return
    
    # Run extraction
    try:
        result = asyncio.run(extract_document_example(pdf_path, config_path))
        print(f"\n✓ Successfully extracted data from {pdf_path}")
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
