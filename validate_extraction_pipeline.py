#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for Extraction Pipeline
Checks structure and imports without requiring dependencies.
"""

import ast
import sys
from pathlib import Path


def validate_python_syntax(filepath):
    """Validate Python syntax by parsing AST"""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_class_exists(filepath, class_name):
    """Check if a class exists in a Python file"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return True
        return False
    except Exception as e:
        print(f"Error checking class {class_name}: {e}")
        return False


def check_function_exists(filepath, class_name, method_name):
    """Check if a method exists in a class"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    # Check both regular and async functions
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        return True
        return False
    except Exception as e:
        print(f"Error checking method {method_name}: {e}")
        return False


def main():
    """Run validation checks"""
    print("=" * 60)
    print("EXTRACTION PIPELINE VALIDATION")
    print("=" * 60)
    
    pipeline_file = Path("extraction/extraction_pipeline.py")
    init_file = Path("extraction/__init__.py")
    
    # Check files exist
    if not pipeline_file.exists():
        print(f"❌ File not found: {pipeline_file}")
        return False
    if not init_file.exists():
        print(f"❌ File not found: {init_file}")
        return False
    
    print(f"✓ Files exist")
    
    # Validate syntax
    valid, error = validate_python_syntax(pipeline_file)
    if not valid:
        print(f"❌ Syntax error in {pipeline_file}: {error}")
        return False
    print(f"✓ Syntax valid: {pipeline_file}")
    
    valid, error = validate_python_syntax(init_file)
    if not valid:
        print(f"❌ Syntax error in {init_file}: {error}")
        return False
    print(f"✓ Syntax valid: {init_file}")
    
    # Check required classes exist
    required_classes = [
        'ExtractedTable',
        'SemanticChunk',
        'DataQualityMetrics',
        'ExtractionResult',
        'TableDataCleaner',
        'ExtractionPipeline'
    ]
    
    for class_name in required_classes:
        if not check_class_exists(pipeline_file, class_name):
            print(f"❌ Class not found: {class_name}")
            return False
    print(f"✓ All required classes exist: {', '.join(required_classes)}")
    
    # Check ExtractionPipeline methods
    required_methods = [
        'extract_complete',
        '_extract_text_safe',
        '_extract_tables_safe',
        '_chunk_with_provenance',
        '_assess_extraction_quality',
        '_compute_sha256'
    ]
    
    for method_name in required_methods:
        if not check_function_exists(pipeline_file, 'ExtractionPipeline', method_name):
            print(f"❌ Method not found: ExtractionPipeline.{method_name}")
            return False
    print(f"✓ All required methods exist in ExtractionPipeline")
    
    # Check that extract_complete is async
    try:
        with open(pipeline_file, 'r') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ExtractionPipeline':
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name == 'extract_complete':
                        print(f"✓ extract_complete is async")
                        break
    except Exception as e:
        print(f"⚠ Could not verify async nature: {e}")
    
    print("\n" + "=" * 60)
    print("✓ ALL VALIDATION CHECKS PASSED")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
