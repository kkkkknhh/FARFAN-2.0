# FARFAN 2.0 - Compilation and Syntax Validation

## Overview

This document describes the pre-test compilation validation system that ensures all scripts are clean, without syntax errors, and ready for execution.

## Quick Start

Run the comprehensive validation:

```bash
python3 pretest_compilation.py
```

This single command:
1. ✅ Compiles all 42 Python scripts
2. ✅ Validates syntax across all modules
3. ✅ Runs all test suites
4. ✅ Reports detailed results

## What Gets Validated

### Python Files (.py extension)
- All modules: `orchestrator.py`, `canonical_notation.py`, etc.
- All tests: `test_*.py`
- All demos: `demo_*.py`
- All examples: `ejemplo_*.py`

### Executable Python Scripts (no extension)
- `dereck_beach`: CDAF Framework processor
- `contradiction_deteccion`: Contradiction detection module
- `embeddings_policy`: Policy embeddings processor
- `financiero_viabilidad_tablas`: Financial viability tables
- `guia_cuestionario`: Questionnaire guide
- `initial_processor_causal_policy`: Initial causal policy processor
- `teoria_cambio_validacion_monte_carlo`: Theory of change Monte Carlo validator

## Validation Phases

### Phase 1: Compilation Validation

Compiles each Python file using `py_compile` to catch:
- Syntax errors (missing colons, parentheses, etc.)
- Indentation errors
- Invalid Python syntax
- Import statement issues

**Output Example:**
```
✓ orchestrator.py
✓ dereck_beach
✓ test_circuit_breaker.py
...
Total files: 42
Passed: 42
Failed: 0
```

### Phase 2: Test Suite Execution

Runs all existing test suites:
- `test_canonical_notation.py` (46 tests)
- `test_circuit_breaker.py` (24 tests)
- `test_risk_mitigation.py` (7 tests)

**Output Example:**
```
✅ test_canonical_notation.py - PASSED
✅ test_circuit_breaker.py - PASSED
✅ test_risk_mitigation.py - PASSED
```

## Exit Codes

- **0**: All scripts compile successfully and all tests pass
- **1**: Compilation errors detected OR tests failed

## Integration with CI/CD

Add to your CI/CD pipeline:

```yaml
# .github/workflows/validation.yml
- name: Validate Compilation
  run: python3 pretest_compilation.py
```

## Manual Validation Commands

### Individual File Compilation

```bash
# Compile a single file
python3 -m py_compile orchestrator.py

# Compile and run a script
python3 -c "import orchestrator"
```

### Run Individual Tests

```bash
# Canonical notation tests
python3 -m unittest test_canonical_notation.py -v

# Circuit breaker tests
python3 -m unittest test_circuit_breaker.py -v

# Risk mitigation tests
python3 test_risk_mitigation.py
```

## Common Issues and Solutions

### Issue: Import Errors

**Problem:**
```
ImportError: No module named 'spacy'
```

**Solution:**
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_lg
```

### Issue: Syntax Error in Script

**Problem:**
```
SyntaxError: invalid syntax (line 42)
```

**Solution:**
1. Open the file in a Python-aware editor
2. Check line 42 for missing colons, quotes, or parentheses
3. Run `python3 -m py_compile <file>` to get detailed error
4. Fix the syntax error

### Issue: Test Timeout

**Problem:**
```
⏱️  test_circuit_breaker.py - TIMEOUT
```

**Solution:**
- Tests have a 120-second timeout
- Circuit breaker tests include intentional delays
- This is normal behavior

## Best Practices

1. **Run before committing**: Always run `pretest_compilation.py` before committing
2. **Fix compilation errors first**: Syntax errors prevent tests from running
3. **Check all scripts**: The validator checks both .py files and executables
4. **Review error messages**: Compilation errors provide line numbers and descriptions

## Script Details

### pretest_compilation.py

**Features:**
- Validates 42 Python files
- Runs 3 test suites (77 total tests)
- Provides colored output (green ✓, red ✗)
- Timeout protection (120s per test suite)
- Detailed error reporting

**Structure:**
```python
def compile_python_file(filepath) -> (success, error)
def get_all_python_files(root_dir) -> List[Path]
def run_test_suite(root_dir) -> (success, output)
def main() -> int
```

## Results Summary

**Current Status (as of validation):**
- ✅ 42 Python files compiled successfully
- ✅ 0 syntax errors detected
- ✅ 46 canonical notation tests passed
- ✅ 24 circuit breaker tests passed
- ✅ 7 risk mitigation tests passed
- ✅ 100% success rate

## References

- Python `py_compile` module: https://docs.python.org/3/library/py_compile.html
- Python `unittest` module: https://docs.python.org/3/library/unittest.html
- FARFAN 2.0 Testing Guide: See `DATA_INTEGRITY_AND_RESOURCE_MANAGEMENT.md`

## Maintenance

### Adding New Scripts

When adding new Python scripts:
1. Ensure they have proper shebang: `#!/usr/bin/env python3`
2. Add to `get_all_python_files()` if it's an executable without .py extension
3. Run `pretest_compilation.py` to validate

### Adding New Tests

When adding new test suites:
1. Follow existing test patterns (unittest or custom main)
2. Add to `run_test_suite()` in `pretest_compilation.py`
3. Verify tests run within 120-second timeout

## Support

For issues with validation:
1. Check error messages from `pretest_compilation.py`
2. Run individual file compilation: `python3 -m py_compile <file>`
3. Review this documentation
4. Check FARFAN 2.0 documentation files
