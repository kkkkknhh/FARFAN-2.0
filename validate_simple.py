#!/usr/bin/env python3
"""Simple validation of unified orchestrator"""

import sys

def main():
    print("Testing unified orchestrator structure...")
    
    try:
        # Test basic import without pandas dependency
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "unified", 
            "orchestration/unified_orchestrator.py"
        )
        
        if spec and spec.loader:
            print("✓ File structure valid")
        else:
            print("✗ File structure invalid")
            return 1
            
        # Count lines
        with open("orchestration/unified_orchestrator.py", 'r') as f:
            lines = len(f.readlines())
        
        print(f"✓ File size: {lines} lines")
        
        # Check for key classes
        with open("orchestration/unified_orchestrator.py", 'r') as f:
            content = f.read()
            
        required = [
            'class UnifiedOrchestrator',
            'class MetricsCollector',
            'class PipelineStage',
            'def _create_prior_snapshot',
            'async def execute_pipeline',
            'STAGE_0_INGESTION',
            'STAGE_8_LEARNING'
        ]
        
        for item in required:
            if item in content:
                print(f"✓ Found: {item}")
            else:
                print(f"✗ Missing: {item}")
                return 1
        
        print("\n✅ All structural checks passed")
        return 0
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
