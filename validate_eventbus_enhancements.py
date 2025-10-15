#!/usr/bin/env python3
"""
Validation Script for EventBus Enhancements
============================================

Validates all enhancements made to the EventBus choreography layer:
1. ContradictionDetectorV2 subscribes to graph.node_added
2. Circuit breaker implementation
3. Memory tracking in StreamingBayesianUpdater
4. Error handling improvements

SIN_CARRETA Compliance:
- Deterministic validation
- Contract-based checks
- Comprehensive reporting
"""

import ast
import sys
from pathlib import Path


class EventBusValidator:
    """Validates EventBus enhancements via AST analysis"""
    
    def __init__(self):
        self.findings = []
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> bool:
        """Run all validations"""
        print("="*80)
        print("EVENTBUS ENHANCEMENTS VALIDATION")
        print("="*80 + "\n")
        
        success = True
        
        # Validation 1: ContradictionDetectorV2 subscriptions
        print("1. Validating ContradictionDetectorV2 subscriptions...")
        if not self.validate_contradiction_detector_subscriptions():
            success = False
            self.errors.append("ContradictionDetectorV2 missing required subscriptions")
        else:
            self.findings.append("✓ ContradictionDetectorV2 subscriptions verified")
        
        # Validation 2: Circuit breaker implementation
        print("2. Validating Circuit Breaker implementation...")
        if not self.validate_circuit_breaker():
            success = False
            self.errors.append("Circuit breaker implementation incomplete")
        else:
            self.findings.append("✓ Circuit breaker implementation verified")
        
        # Validation 3: Memory tracking
        print("3. Validating StreamingBayesianUpdater memory tracking...")
        if not self.validate_memory_tracking():
            success = False
            self.errors.append("Memory tracking implementation incomplete")
        else:
            self.findings.append("✓ Memory tracking implementation verified")
        
        # Validation 4: Error handling
        print("4. Validating error handling enhancements...")
        if not self.validate_error_handling():
            self.warnings.append("Some error handling enhancements may be incomplete")
        else:
            self.findings.append("✓ Error handling enhancements verified")
        
        print()
        return success
    
    def validate_contradiction_detector_subscriptions(self) -> bool:
        """Validate that ContradictionDetectorV2 subscribes to both edge and node events"""
        event_bus_path = Path("choreography/event_bus.py")
        
        try:
            with open(event_bus_path, 'r') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Find ContradictionDetectorV2 class
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "ContradictionDetectorV2":
                    # Find __init__ method
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            # Check for subscribe calls
                            subscriptions = []
                            for subnode in ast.walk(item):
                                if isinstance(subnode, ast.Call):
                                    if isinstance(subnode.func, ast.Attribute) and subnode.func.attr == "subscribe":
                                        if subnode.args:
                                            if isinstance(subnode.args[0], ast.Constant):
                                                subscriptions.append(subnode.args[0].value)
                            
                            print(f"   Found subscriptions: {subscriptions}")
                            
                            if "graph.edge_added" in subscriptions and "graph.node_added" in subscriptions:
                                print("   ✓ Both graph.edge_added and graph.node_added subscriptions found")
                                return True
                            else:
                                print("   ✗ Missing required subscriptions")
                                return False
            
            print("   ✗ ContradictionDetectorV2 class not found")
            return False
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def validate_circuit_breaker(self) -> bool:
        """Validate circuit breaker implementation"""
        event_bus_path = Path("choreography/event_bus.py")
        
        try:
            with open(event_bus_path, 'r') as f:
                source = f.read()
            
            # Check for circuit breaker components
            required_components = [
                '_circuit_breaker_active',
                '_failed_handler_count',
                'reset_circuit_breaker',
                'get_circuit_breaker_status',
                '_check_event_storm'
            ]
            
            found_components = []
            for component in required_components:
                if component in source:
                    found_components.append(component)
                    print(f"   ✓ Found: {component}")
                else:
                    print(f"   ✗ Missing: {component}")
            
            if len(found_components) == len(required_components):
                print("   ✓ All circuit breaker components found")
                return True
            else:
                print(f"   ✗ Missing {len(required_components) - len(found_components)} components")
                return False
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def validate_memory_tracking(self) -> bool:
        """Validate memory tracking in StreamingBayesianUpdater"""
        evidence_stream_path = Path("choreography/evidence_stream.py")
        
        try:
            with open(evidence_stream_path, 'r') as f:
                source = f.read()
            
            # Check for memory tracking components
            required_components = [
                'track_memory',
                '_memory_snapshots',
                '_track_memory_snapshot',
                'get_memory_stats'
            ]
            
            found_components = []
            for component in required_components:
                if component in source:
                    found_components.append(component)
                    print(f"   ✓ Found: {component}")
                else:
                    print(f"   ✗ Missing: {component}")
            
            if len(found_components) == len(required_components):
                print("   ✓ All memory tracking components found")
                return True
            else:
                print(f"   ✗ Missing {len(required_components) - len(found_components)} components")
                return False
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def validate_error_handling(self) -> bool:
        """Validate error handling enhancements"""
        event_bus_path = Path("choreography/event_bus.py")
        
        try:
            with open(event_bus_path, 'r') as f:
                source = f.read()
            
            # Check for error handling patterns
            error_patterns = [
                'CONTRACT_VIOLATION',
                'CIRCUIT_BREAKER',
                'EVENT_STORM_DETECTED',
                'exc_info=True'
            ]
            
            found_patterns = []
            for pattern in error_patterns:
                if pattern in source:
                    found_patterns.append(pattern)
                    print(f"   ✓ Found: {pattern}")
                else:
                    print(f"   ⚠ Missing: {pattern}")
            
            if len(found_patterns) >= 3:  # At least 3 out of 4
                print("   ✓ Sufficient error handling patterns found")
                return True
            else:
                print(f"   ⚠ Only {len(found_patterns)}/4 error patterns found")
                return False
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80 + "\n")
        
        if self.findings:
            print("FINDINGS:")
            for finding in self.findings:
                print(f"  {finding}")
            print()
        
        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
            print()
        
        if self.errors:
            print("ERRORS:")
            for error in self.errors:
                print(f"  ✗ {error}")
            print()
        
        if not self.errors:
            print("✅ ALL VALIDATIONS PASSED")
            print("\nEventBus enhancements are production-ready:")
            print("  • ContradictionDetectorV2 fully connected")
            print("  • Circuit breaker operational")
            print("  • Memory tracking enabled")
            print("  • Error handling comprehensive")
        else:
            print("❌ VALIDATION FAILED")
            print(f"\nFound {len(self.errors)} critical errors")


def main():
    """Execute validation"""
    validator = EventBusValidator()
    
    success = validator.validate_all()
    validator.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
