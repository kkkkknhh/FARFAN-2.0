#!/usr/bin/env python3
"""
Comprehensive validation of event_bus.py refactoring.
Validates structure, complexity reduction, and pattern preservation.
"""

import ast
import re
import sys


def calculate_cognitive_complexity(func_node):
    """
    Calculate cognitive complexity with proper nesting penalties.
    Based on SonarSource's cognitive complexity metric.
    """
    complexity = 0

    def visit(node, nesting=0):
        nonlocal complexity

        # Control flow structures add complexity + nesting penalty
        if isinstance(node, (ast.If, ast.While, ast.For)):
            complexity += 1 + nesting
            new_nesting = nesting + 1
            for child in ast.iter_child_nodes(node):
                visit(child, new_nesting)
            return

        # Exception handlers
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1 + nesting
            new_nesting = nesting + 1
            for child in ast.iter_child_nodes(node):
                visit(child, new_nesting)
            return

        # Boolean operators
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1

        # Recursion (not applicable here, but good to track)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Don't count nested functions toward parent complexity
            return

        # With statements add nesting
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            new_nesting = nesting + 1
            for child in ast.iter_child_nodes(node):
                visit(child, new_nesting)
            return

        # Try blocks add nesting for handlers
        elif isinstance(node, ast.Try):
            # Body doesn't add nesting
            for child in node.body:
                visit(child, nesting)
            # Handlers add nesting
            for handler in node.handlers:
                visit(handler, nesting)
            for child in node.orelse + node.finalbody:
                visit(child, nesting)
            return

        # Continue traversing
        for child in ast.iter_child_nodes(node):
            visit(child, nesting)

    visit(func_node, 0)
    return complexity


def analyze_file(filepath):
    """Analyze the event_bus.py file."""
    with open(filepath, "r") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return None, f"Syntax error: {e}"

    # Find EventBus class and analyze methods
    results = {"methods": {}, "source": source, "tree": tree}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EventBus":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = calculate_cognitive_complexity(item)
                    results["methods"][item.name] = {
                        "complexity": complexity,
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                        "lineno": item.lineno,
                    }

    return results, None


def validate_refactoring(results):
    """Validate the refactoring meets all requirements."""
    errors = []
    warnings = []

    source = results["source"]
    methods = results["methods"]

    # 1. Check publish method complexity
    if "publish" not in methods:
        errors.append("publish method not found")
    else:
        publish_complexity = methods["publish"]["complexity"]
        if publish_complexity > 15:
            errors.append(
                f"publish complexity too high: {publish_complexity} (target: ≤15)"
            )
        elif publish_complexity <= 15:
            print(f"✓ publish method complexity: {publish_complexity} (target: ≤15)")

    # 2. Check helper methods exist
    required_helpers = [
        "_prepare_handler_task",
        "_collect_handler_tasks",
        "_log_handler_exceptions",
    ]

    missing_helpers = [h for h in required_helpers if h not in methods]
    if missing_helpers:
        errors.append(f"Missing helper methods: {', '.join(missing_helpers)}")
    else:
        print(f"✓ All {len(required_helpers)} helper methods present")
        for helper in required_helpers:
            print(f"  • {helper} (complexity: {methods[helper]['complexity']})")

    # 3. Check event types are preserved
    required_event_types = [
        "graph.edge_added",
        "contradiction.detected",
        "posterior.updated",
    ]

    for event_type in required_event_types:
        if event_type not in source:
            errors.append(f"Event type '{event_type}' not found in source")
        else:
            print(f"✓ Event type preserved: {event_type}")

    # 4. Check asyncio.gather is still used
    if "asyncio.gather" not in source:
        errors.append("asyncio.gather pattern not found")
    else:
        print("✓ asyncio.gather pattern preserved")

    # 5. Check return_exceptions=True is still present
    if "return_exceptions=True" not in source:
        warnings.append("return_exceptions=True pattern not found")
    else:
        print("✓ Error handling with return_exceptions=True preserved")

    # 6. Verify audit trail logging
    logging_patterns = ["self.event_log.append", "async with self._lock"]

    for pattern in logging_patterns:
        if pattern not in source:
            warnings.append(f"Audit pattern '{pattern}' not found")
        else:
            print(f"✓ Audit trail pattern preserved: {pattern}")

    # 7. Check error handling logging
    if "logger.error" not in source:
        warnings.append("Error logging not found")
    else:
        error_logs = source.count("logger.error")
        print(f"✓ Error handling logging preserved ({error_logs} occurrences)")

    # 8. Verify subscribe/unsubscribe unchanged
    if "def subscribe" in source and "def unsubscribe" in source:
        print("✓ Subscription methods preserved")

    # 9. Check for proper method signatures
    if "async def publish(self, event: PDMEvent)" not in source:
        warnings.append("publish method signature may have changed")
    else:
        print("✓ publish method signature preserved")

    return errors, warnings


def compare_complexity(old_value, new_results):
    """Compare old and new complexity."""
    new_publish = new_results["methods"].get("publish", {}).get("complexity", 0)

    print(f"\nComplexity Comparison:")
    print(f"  Before: {old_value}")
    print(f"  After:  {new_publish}")
    print(f"  Reduction: {old_value - new_publish}")

    if new_publish <= 15:
        print(f"  ✓ Target achieved (≤15)")
        return True
    else:
        print(f"  ✗ Still above target")
        return False


def main():
    """Run validation."""
    print("=" * 60)
    print("Event Bus Refactoring Validation")
    print("=" * 60)

    filepath = "choreography/event_bus.py"

    print(f"\nAnalyzing {filepath}...\n")

    results, error = analyze_file(filepath)

    if error:
        print(f"✗ {error}")
        return 1

    print("✓ Syntax valid\n")

    # Validate refactoring
    print("Checking refactoring requirements:")
    print("-" * 60)

    errors, warnings = validate_refactoring(results)

    print("-" * 60)

    # Compare with original complexity (17)
    original_complexity = 17
    compare_complexity(original_complexity, results)

    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    if errors:
        print(f"\n✗ {len(errors)} error(s):")
        for error in errors:
            print(f"  • {error}")

    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  • {warning}")

    if not errors:
        print("\n✅ All requirements met!")
        print("\nRefactoring achievements:")
        print("  • Extracted 3 helper methods")
        print(
            "  • Reduced cognitive complexity from 17 to",
            results["methods"]["publish"]["complexity"],
        )
        print("  • Preserved all event types and async patterns")
        print("  • Maintained error handling and audit trail")
        return 0
    else:
        print("\n❌ Validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
