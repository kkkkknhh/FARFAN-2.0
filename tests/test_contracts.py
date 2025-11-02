#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contract Tests
==============

Tests for core_contracts.py and contracts_runtime.py

Validates:
- All contracts are importable
- Pydantic models enforce invariants
- Strict mode rejects unknown fields
- Schema version patterns are validated
- Numeric bounds are enforced
"""

import pytest
from pydantic import ValidationError

from contracts_runtime import (
    SemanticAnalyzerInputModel,
    SemanticAnalyzerOutputModel,
    SemanticChunkModel,
    ContradictionDetectionInputModel,
    ContradictionDetectionOutputModel,
    ContradictionModel,
    FileReadRequestModel,
    FileWriteRequestModel,
    FileReadResultModel,
    FileWriteResultModel,
)


class TestSemanticContracts:
    """Test semantic analysis contracts."""
    
    def test_semantic_analyzer_input_valid(self):
        """Test valid semantic analyzer input."""
        data = {
            'text': 'Sample document text',
            'segments': ['segment1', 'segment2'],
            'ontology_params': {'param1': 'value1'},
            'schema_version': 'sem-1.3'
        }
        model = SemanticAnalyzerInputModel(**data)
        assert model.text == 'Sample document text'
        assert model.schema_version == 'sem-1.3'
    
    def test_semantic_analyzer_input_empty_text(self):
        """Test that empty text is rejected."""
        data = {
            'text': '',  # Invalid: empty
            'segments': [],
            'ontology_params': {},
            'schema_version': 'sem-1.3'
        }
        with pytest.raises(ValidationError) as exc_info:
            SemanticAnalyzerInputModel(**data)
        assert 'text' in str(exc_info.value)
    
    def test_semantic_analyzer_input_invalid_version(self):
        """Test that invalid schema version is rejected."""
        data = {
            'text': 'Sample text',
            'segments': [],
            'ontology_params': {},
            'schema_version': 'invalid-version'  # Invalid pattern
        }
        with pytest.raises(ValidationError) as exc_info:
            SemanticAnalyzerInputModel(**data)
        assert 'schema_version' in str(exc_info.value)
    
    def test_semantic_analyzer_input_strict_mode(self):
        """Test that unknown fields are rejected."""
        data = {
            'text': 'Sample text',
            'segments': [],
            'ontology_params': {},
            'schema_version': 'sem-1.3',
            'unknown_field': 'should fail'  # Unknown field
        }
        with pytest.raises(ValidationError) as exc_info:
            SemanticAnalyzerInputModel(**data)
        assert 'Extra inputs are not permitted' in str(exc_info.value)
    
    def test_semantic_chunk_valid(self):
        """Test valid semantic chunk."""
        data = {
            'id': 'chunk-1',
            'text': 'Chunk text content',
            'start_pos': 0,
            'end_pos': 100,
            'embedding': None,
            'metadata': {}
        }
        model = SemanticChunkModel(**data)
        assert model.id == 'chunk-1'
        assert model.end_pos == 100
    
    def test_semantic_chunk_invalid_positions(self):
        """Test that end_pos must be > start_pos."""
        data = {
            'id': 'chunk-1',
            'text': 'Chunk text',
            'start_pos': 100,
            'end_pos': 50,  # Invalid: <= start_pos
            'embedding': None,
            'metadata': {}
        }
        with pytest.raises(ValidationError) as exc_info:
            SemanticChunkModel(**data)
        assert 'end_pos' in str(exc_info.value).lower()
    
    def test_semantic_chunk_embedding_dimension(self):
        """Test that embedding dimension is validated."""
        # Valid 768-dim embedding
        data = {
            'id': 'chunk-1',
            'text': 'Chunk text',
            'start_pos': 0,
            'end_pos': 100,
            'embedding': [0.1] * 768,
            'metadata': {}
        }
        model = SemanticChunkModel(**data)
        assert len(model.embedding) == 768
        
        # Invalid dimension
        data['embedding'] = [0.1] * 500  # Invalid dimension
        with pytest.raises(ValidationError) as exc_info:
            SemanticChunkModel(**data)
        assert 'embedding' in str(exc_info.value).lower()


class TestContradictionContracts:
    """Test contradiction detection contracts."""
    
    def test_contradiction_severity_literal(self):
        """Test that severity must be a valid literal."""
        data = {
            'statement1': 'Statement A',
            'statement2': 'Statement B',
            'severity': 'critical',  # Valid
            'confidence': 0.9,
            'explanation': 'These contradict',
            'metadata': {}
        }
        model = ContradictionModel(**data)
        assert model.severity == 'critical'
        
        # Invalid severity
        data['severity'] = 'extreme'  # Not in Literal
        with pytest.raises(ValidationError):
            ContradictionModel(**data)
    
    def test_contradiction_detection_output_count_validation(self):
        """Test that total_count must equal len(contradictions)."""
        contradiction = {
            'statement1': 'Statement A',
            'statement2': 'Statement B',
            'severity': 'high',
            'confidence': 0.8,
            'explanation': 'Contradiction',
            'metadata': {}
        }
        
        data = {
            'contradictions': [contradiction],
            'total_count': 1,  # Valid: matches length
            'quality_grade': 'Bueno',
            'schema_version': 'cd-1.3'
        }
        model = ContradictionDetectionOutputModel(**data)
        assert model.total_count == 1
        
        # Mismatched count
        data['total_count'] = 2  # Invalid: doesn't match
        with pytest.raises(ValidationError) as exc_info:
            ContradictionDetectionOutputModel(**data)
        assert 'total_count' in str(exc_info.value)


class TestFileContracts:
    """Test file I/O contracts."""
    
    def test_file_read_request_defaults(self):
        """Test file read request with defaults."""
        data = {
            'path': '/path/to/file.txt',
            'encoding': 'utf-8'
        }
        model = FileReadRequestModel(**data)
        assert model.path == '/path/to/file.txt'
        assert model.encoding == 'utf-8'
    
    def test_file_write_request_create_dirs(self):
        """Test file write request with create_dirs."""
        data = {
            'path': '/path/to/file.txt',
            'content': 'File content',
            'encoding': 'utf-8',
            'create_dirs': True
        }
        model = FileWriteRequestModel(**data)
        assert model.create_dirs is True
    
    def test_file_result_error_optional(self):
        """Test that error is optional in results."""
        # Success case: no error
        data = {
            'content': 'File content',
            'path': '/path/to/file.txt',
            'success': True,
            'error': None
        }
        model = FileReadResultModel(**data)
        assert model.error is None
        
        # Failure case: with error
        data = {
            'content': '',
            'path': '/path/to/file.txt',
            'success': False,
            'error': 'File not found'
        }
        model = FileReadResultModel(**data)
        assert model.error == 'File not found'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
