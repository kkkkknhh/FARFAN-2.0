#!/bin/bash
# FARFAN 2.0 - Quick Validation Script
# Usage: ./validate.sh

set -e

echo "================================================================================"
echo "FARFAN 2.0 - Quick Validation"
echo "================================================================================"
echo ""

# Run the comprehensive validation
python3 pretest_compilation.py

echo ""
echo "================================================================================"
echo "✅ VALIDATION COMPLETE - All scripts are clean and ready for execution"
echo "================================================================================"
