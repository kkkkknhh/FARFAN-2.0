# Known Issues

## semantic_chunking_policy.py - Indentation Errors

**Status**: Not Fixed (Non-Critical)  
**Reason**: File has systematic indentation errors throughout and is not imported/used by any other module.

**Error**: IndentationError at multiple lines due to improper class and method indentation.

**Impact**: None - file is not used in the codebase (no imports found).

**Resolution Options**:
1. Keep as-is and exclude from compilation checks (current approach)
2. Complete rewrite with proper indentation structure
3. Remove file if confirmed unused

**Search Result**: 
```bash
$ grep -r "import semantic_chunking_policy\|from semantic_chunking_policy" --include="*.py" .
# (no results - file is not used)
```

## Fixed Issues

### orchestrator.py
- ✅ Added missing `_audit_store_dir` attribute initialization
- ✅ Added missing `_retention_years` attribute initialization (set to 7 for compliance)
- ✅ Both attributes are now properly initialized in `__init__` method
- ✅ Methods `append_audit_record` and `_calculate_retention_date` now work correctly

### contradiction_deteccion.py  
- ✅ Fixed indentation error at line 816 (recommendations assignment)
- ✅ Fixed indentation for lines 818-852 (audit summary and regulatory analysis)
- ✅ All code now properly indented within the `detect` method
- ✅ File compiles without errors

## Verification

All fixes verified with:
```bash
python -m py_compile orchestrator.py contradiction_deteccion.py
python orchestrator.py  # Runs validation - PASSED
```

Test of audit functionality:
```python
from orchestrator import create_orchestrator
o = create_orchestrator()
result = o.append_audit_record('test_run_001', {"test": "data"}, "Sample text")
verification = o.verify_audit_record(Path(result['record_path']))
# Both operations complete successfully
```
