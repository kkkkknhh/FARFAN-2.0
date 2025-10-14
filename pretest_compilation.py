#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FARFAN 2.0 - Pre-Test Compilation Validator
Valida que todos los scripts compilen sin errores de sintaxis antes de ejecutar
"""

import sys
import py_compile
from pathlib import Path
from typing import List, Tuple
import subprocess


def compile_python_file(filepath: Path) -> Tuple[bool, str]:
    """
    Compila un archivo Python y retorna el resultado
    
    Args:
        filepath: Path al archivo Python
        
    Returns:
        Tuple (success: bool, error_msg: str)
    """
    try:
        py_compile.compile(str(filepath), doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, str(e)


def get_all_python_files(root_dir: Path) -> List[Path]:
    """
    Obtiene todos los archivos Python en el directorio
    
    Args:
        root_dir: Directorio raíz
        
    Returns:
        Lista de paths a archivos Python
    """
    python_files = []
    
    # Archivos .py
    python_files.extend(root_dir.glob("*.py"))
    
    # Ejecutables sin extensión que son Python
    for executable in ["dereck_beach", "contradiction_deteccion", "embeddings_policy",
                      "financiero_viabilidad_tablas", "guia_cuestionario",
                      "initial_processor_causal_policy", "teoria_cambio_validacion_monte_carlo"]:
        exec_path = root_dir / executable
        if exec_path.exists():
            python_files.append(exec_path)
    
    return sorted(python_files)


def run_test_suite(root_dir: Path) -> Tuple[bool, str]:
    """
    Ejecuta las suites de test existentes
    
    Args:
        root_dir: Directorio raíz
        
    Returns:
        Tuple (success: bool, output: str)
    """
    test_files = [
        "test_canonical_notation.py",
        "test_circuit_breaker.py",
        "test_risk_mitigation.py"
    ]
    
    results = []
    all_passed = True
    
    for test_file in test_files:
        test_path = root_dir / test_file
        if not test_path.exists():
            results.append(f"⚠️  Test {test_file} not found - SKIPPED")
            continue
        
        try:
            if test_file == "test_risk_mitigation.py":
                # Este test usa su propio main
                result = subprocess.run(
                    [sys.executable, str(test_path)],
                    cwd=str(root_dir),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
            else:
                # Estos tests usan unittest
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", test_file],
                    cwd=str(root_dir),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
            
            if result.returncode == 0:
                results.append(f"✅ {test_file} - PASSED")
            else:
                results.append(f"❌ {test_file} - FAILED")
                results.append(f"   Error: {result.stderr[:200]}")
                all_passed = False
                
        except subprocess.TimeoutExpired:
            results.append(f"⏱️  {test_file} - TIMEOUT")
            all_passed = False
        except Exception as e:
            results.append(f"❌ {test_file} - ERROR: {e}")
            all_passed = False
    
    return all_passed, "\n".join(results)


def main() -> int:
    """Entry point"""
    print("=" * 80)
    print("FARFAN 2.0 - PRE-TEST COMPILATION VALIDATOR")
    print("=" * 80)
    print()
    
    root_dir = Path(__file__).parent
    
    # Phase 1: Compilation
    print("Phase 1: Compilation Validation")
    print("-" * 80)
    
    python_files = get_all_python_files(root_dir)
    total_files = len(python_files)
    passed = 0
    failed = 0
    errors = []
    
    for filepath in python_files:
        filename = filepath.name
        success, error = compile_python_file(filepath)
        
        if success:
            print(f"✓ {filename}")
            passed += 1
        else:
            print(f"✗ {filename}")
            print(f"  Error: {error}")
            failed += 1
            errors.append((filename, error))
    
    print()
    print("-" * 80)
    print(f"Total files: {total_files}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    
    if failed > 0:
        print("❌ COMPILATION FAILED")
        print()
        print("Errors found:")
        for filename, error in errors:
            print(f"  • {filename}: {error[:100]}")
        print()
        return 1
    
    print("✅ ALL SCRIPTS COMPILE SUCCESSFULLY")
    print()
    
    # Phase 2: Test Execution
    print("=" * 80)
    print("Phase 2: Test Suite Execution")
    print("-" * 80)
    
    tests_passed, test_output = run_test_suite(root_dir)
    print(test_output)
    print()
    
    if not tests_passed:
        print("=" * 80)
        print("⚠️  COMPILATION PASSED BUT SOME TESTS FAILED")
        print("=" * 80)
        print()
        return 1
    
    # Success
    print("=" * 80)
    print("✅ ALL CHECKS PASSED - SCRIPTS CLEAN AND READY")
    print("=" * 80)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
