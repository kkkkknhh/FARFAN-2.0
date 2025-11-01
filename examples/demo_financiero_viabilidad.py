#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Viability Demo
========================

Demonstrates financial viability analysis for Colombian Municipal Development
Plans using PDF table extraction and validation.

Run with:
    python -m examples.demo_financiero_viabilidad <path_to_pdf>
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from financiero_viabilidad_tablas import PDETMunicipalPlanAnalyzer


async def main():
    """Main function for financial viability analysis."""
    
    if len(sys.argv) < 2:
        print("Usage: python -m examples.demo_financiero_viabilidad <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # Initialize analyzer
    analyzer = PDETMunicipalPlanAnalyzer(use_gpu=True, confidence_threshold=0.7)

    # Execute analysis
    results = await analyzer.analyze_complete_plan(pdf_path)

    # Save results
    output_file = Path(pdf_path).stem + "_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
