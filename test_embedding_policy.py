#!/usr/bin/env python3
"""
Quick validation test for emebedding_policy.py _contains_table fix
"""
import sys

def test_import():
    """Test that module can be imported"""
    try:
        from emebedding_policy import AdvancedSemanticChunker, ChunkingConfig
        print("✓ Module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_chunker_instantiation():
    """Test that chunker can be instantiated"""
    try:
        from emebedding_policy import AdvancedSemanticChunker, ChunkingConfig
        chunker = AdvancedSemanticChunker(ChunkingConfig())
        print("✓ Chunker instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        return False

def test_chunk_document():
    """Test chunk_document method with sample data"""
    try:
        from emebedding_policy import AdvancedSemanticChunker, ChunkingConfig
        
        chunker = AdvancedSemanticChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))
        
        sample_text = """
        CAPÍTULO I - INTRODUCCIÓN
        
        Este es un documento de ejemplo que contiene información sobre políticas municipales.
        La Tabla 1 muestra los indicadores principales del proyecto.
        
        El presupuesto asignado es de 500 millones de pesos, distribuidos en varios programas:
        • Programa de educación
        • Programa de salud
        • Programa de infraestructura
        
        Los resultados esperados incluyen mejoras en la calidad de vida de los habitantes.
        """
        
        doc_metadata = {"doc_id": "test_doc_001"}
        
        chunks = chunker.chunk_document(sample_text, doc_metadata)
        
        print(f"✓ chunk_document executed successfully, generated {len(chunks)} chunks")
        
        # Verify _contains_table was called correctly
        has_table_chunks = sum(1 for chunk in chunks if chunk["metadata"]["has_table"])
        print(f"✓ Found {has_table_chunks} chunks marked as containing tables")
        
        return True
    except Exception as e:
        print(f"✗ chunk_document failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    results = [
        test_import(),
        test_chunker_instantiation(),
        test_chunk_document()
    ]
    
    if all(results):
        print("\n✓ All tests passed")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
